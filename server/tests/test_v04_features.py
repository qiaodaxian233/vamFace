"""v0.4 四件套的测试:裁剪 / 先验 / 表情归一化 / coarse-to-fine。

重头戏是先验的**定量**验证:同一个隐藏目标、同一个确定性优化器(greedy),
带先验 vs 不带先验对比"达到阈值分数所需的评估次数"。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from vamface_mcp.bridge_client import BridgeClient
from vamface_mcp.fitting import (DEFAULT_STAGES, FitConfig, fit_face,
                                 neutralize_expression)
from vamface_mcp.mock_vam import (FaceRenderer, MockVamServer,
                                  features_from_mock)
from vamface_mcp.priors import order_by_prior, seed_from_diff
from vamface_mcp.scorers import (CroppedScorer, GeometryScorer, PixelScorer,
                                 bbox_from_background, find_geometry_scorer)


@pytest.fixture()
def mock():
    srv = MockVamServer(port=0)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture()
def bridge(mock):
    b = BridgeClient("127.0.0.1", mock.port, timeout=15.0)
    yield b
    b.close()


def _target_png(tmp_path: Path, morphs: dict) -> Path:
    p = tmp_path / "target.png"
    FaceRenderer().render(morphs).save(p)
    return p


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------

def test_bbox_from_background_on_mock_render():
    img = np.asarray(FaceRenderer().render({}))
    box = bbox_from_background(img)
    assert box is not None
    x0, y0, x1, y1 = box
    # 框住了主体(包含画面中心),且明显小于整图
    assert x0 < img.shape[1] / 2 < x1 and y0 < img.shape[0] / 2 < y1
    assert (x1 - x0) < img.shape[1] and (y1 - y0) < img.shape[0]


def test_bbox_from_background_blank_image_is_none():
    blank = np.full((64, 64, 3), 200, dtype=np.uint8)
    assert bbox_from_background(blank) is None


def test_cropped_scorer_degrades_to_full_image():
    blank = np.full((64, 64, 3), 200, dtype=np.uint8)
    cs = CroppedScorer(PixelScorer(), bbox_from_background)
    assert cs.score(blank, blank.copy()) == 1.0  # 找不到框 → 整图,不 crash
    assert cs.crop_misses >= 1


def test_crop_makes_pixel_score_composition_invariant():
    """同一张脸平移到画面另一角,裁剪后像素分应远高于不裁剪。"""
    face = np.asarray(FaceRenderer().render({}))
    h, w = face.shape[:2]
    box = bbox_from_background(face)
    x0, y0, x1, y1 = box
    crop = face[y0:y1, x0:x1]
    # 目标:脸贴左上;候选:同一张脸贴右下 —— 内容一致、构图不同
    a = np.full((h * 2, w * 2, 3), (245, 245, 248), dtype=np.uint8)
    b = a.copy()
    a[:crop.shape[0], :crop.shape[1]] = crop
    b[-crop.shape[0]:, -crop.shape[1]:] = crop
    raw = PixelScorer().score(a, b)
    cropped = CroppedScorer(PixelScorer(), bbox_from_background).score(a, b)
    assert cropped > 0.95
    assert cropped > raw + 0.2


# ---------------------------------------------------------------------------
# mock 特征提取器(先验测试的地基)
# ---------------------------------------------------------------------------

def test_features_from_mock_detects_and_tracks_morphs():
    base = features_from_mock(FaceRenderer().render_array({}))
    wide = features_from_mock(
        FaceRenderer().render_array({"Eyes Width Spacing": 1.0}))
    long_face = features_from_mock(FaceRenderer().render_array({"Face Long": 1.0}))
    assert base and wide and long_face
    assert wide["eye_gap"] > base["eye_gap"]
    assert long_face["face_aspect"] > base["face_aspect"]


def test_features_from_mock_rejects_real_images():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    assert features_from_mock(noise) is None


# ---------------------------------------------------------------------------
# 先验:单测 + 定量端到端
# ---------------------------------------------------------------------------

def test_seed_from_diff_signs_and_clip():
    diff = {"eye_gap": 0.10,        # 目标两眼更开 → Eyes Width Spacing 正种子
            "face_aspect": -0.30,   # 目标脸更圆 → Face Long 负 / Face Round 正
            "mouth_w": 0.50}        # 大差值 → 会被 clip
    seed = seed_from_diff(diff, ["Eyes Width Spacing", "Face Long",
                                 "Face Round", "Mouth Width"])
    assert seed["Eyes Width Spacing"] > 0
    assert seed["Face Long"] < 0 < seed["Face Round"]
    assert seed["Mouth Width"] == pytest.approx(0.6)  # PRIOR_CLIP
    # 不在 allowed 里的名字绝不出现
    assert "Lips Width" not in seed


def test_seed_respects_bounds_fn():
    seed = seed_from_diff({"eye_gap": 0.5}, ["Eyes Width Spacing"],
                          bounds_fn=lambda n: (-0.2, 0.2))
    assert seed["Eyes Width Spacing"] == pytest.approx(0.2)


def test_order_by_prior_stable():
    order = order_by_prior(["a", "b", "c", "d"], {"c": 0.5, "a": -0.9})
    assert order == ["a", "c", "b", "d"]  # |种子|降序,无种子的保持原序在后


def _evals_to_reach(history, threshold):
    for i, s in enumerate(history):
        if s >= threshold:
            return i + 1
    return len(history) + 1000  # 没达到:视为超预算


def test_prior_quantitatively_reduces_evaluations(bridge, tmp_path):
    """核心验证:同一隐藏目标 + 确定性 greedy,带先验应更快达到阈值。"""
    hidden = {"Eyes Width Spacing": 1.0, "Mouth Width": -1.0, "Face Long": 0.9}
    target = _target_png(tmp_path, hidden)
    names = sorted(hidden)

    def run(use_prior):
        bridge.reset_morphs("Person")
        # screenshot_width 必须与目标图同分辨率(512):features_from_mock 是
        # 颜色分割,双三次降采样会把掩膜边缘搅烂(嘴宽特征直接失真 4 倍)。
        cfg = FitConfig(morph_names=names,
                        bounds={n: (-1.5, 1.5) for n in names},
                        max_iters=60, screenshot_width=512)
        return fit_face(bridge, str(target), cfg, optimizer="greedy",
                        scorer=GeometryScorer(features_from_mock),
                        use_prior=use_prior, neutralize=False)

    with_p = run(True)
    without_p = run(False)

    # 探针基线 == 无先验的起点(同一零 morph 状态,确定性打分)
    assert with_p.history[0] == pytest.approx(without_p.history[0], abs=1e-6)
    # 先验种子的符号必须与隐藏答案一致
    assert with_p.prior_seed["Eyes Width Spacing"] > 0
    assert with_p.prior_seed["Mouth Width"] < 0
    assert with_p.prior_seed["Face Long"] > 0
    # 种子起步分显著高于零起步 —— 起点已经进对了象限
    assert with_p.history[1] > without_p.history[0] + 0.05
    # 达到阈值所需评估次数:带先验必须**严格更少**
    # (实测 0.95 阈值:带先验 14 次 vs 不带 35 次,-60% 预算)
    for thr in (0.90, 0.95):
        assert _evals_to_reach(with_p.history, thr) < \
            _evals_to_reach(without_p.history, thr), f"thr={thr}"


def test_find_geometry_scorer_through_wrappers():
    g = GeometryScorer(features_from_mock)
    assert find_geometry_scorer(g) is g
    assert find_geometry_scorer(CroppedScorer(g, bbox_from_background)) is g
    assert find_geometry_scorer(PixelScorer()) is None


# ---------------------------------------------------------------------------
# 表情归一化
# ---------------------------------------------------------------------------

def test_neutralize_zeroes_expressions_keeps_identity(bridge):
    bridge.set_morphs("Person", {"Smile Full Face": 1.0, "Brow Up": 0.8,
                                 "Nose Width": 0.5})
    r = neutralize_expression(bridge, "Person")
    assert sorted(r["zeroed"]) == ["Brow Up", "Smile Full Face"]
    vals = bridge.get_morphs("Person", changed_only=True)
    assert vals == {"Nose Width": 0.5}  # 身份 morph 原封不动


def test_fit_face_neutralizes_by_default(bridge, tmp_path):
    bridge.set_morphs("Person", {"Smile Full Face": 1.0})
    target = _target_png(tmp_path, {"Nose Width": 0.8})
    cfg = FitConfig(morph_names=["Nose Width"],
                    bounds={"Nose Width": (-1.5, 1.5)},
                    max_iters=15, screenshot_width=192)
    result = fit_face(bridge, str(target), cfg, optimizer="greedy",
                      style="pixel", use_prior=False)
    assert result.neutralized == ["Smile Full Face"]
    assert bridge.get_morphs("Person").get("Smile Full Face", 0.0) == 0.0


# ---------------------------------------------------------------------------
# coarse-to-fine
# ---------------------------------------------------------------------------

def test_coarse_to_fine_two_stages_cover_all_and_fit(bridge, tmp_path):
    hidden = {"Face Long": 0.9, "Jaw Width": 0.8,          # 轮廓阶段
              "Eyes Width Spacing": 1.0, "Mouth Width": -0.9}  # 五官阶段
    target = _target_png(tmp_path, hidden)
    names = sorted(hidden)
    cfg = FitConfig(morph_names=names,
                    bounds={n: (-1.5, 1.5) for n in names},
                    max_iters=80, screenshot_width=512)  # 与目标图同分辨率
    result = fit_face(bridge, str(target), cfg, optimizer="greedy",
                      scorer=GeometryScorer(features_from_mock),
                      coarse_to_fine=True, neutralize=False)
    assert result.stage_count == 2
    assert set(result.best_morphs) == set(names)  # 两阶段合并覆盖全部维度
    # 拟合方向:多数维度符号与答案一致
    agree = sum(1 for n in names if result.best_morphs[n] * hidden[n] > 0)
    assert agree >= 3, f"{result.best_morphs} vs {hidden}"


def test_custom_stages_and_leftover_names(bridge, tmp_path):
    """用户自定义 morph(不属于任何分组)必须归入末阶段,不丢维度。"""
    target = _target_png(tmp_path, {"Nose Width": 0.5})
    cfg = FitConfig(morph_names=["Nose Width", "自定义Morph不存在"],
                    bounds={}, max_iters=20, screenshot_width=192)
    result = fit_face(bridge, str(target), cfg, optimizer="greedy",
                      style="pixel", use_prior=False,
                      stages=[["nose"]])
    assert result.stage_count == 1
    assert "自定义Morph不存在" in result.best_morphs  # 进了搜索(mock 报 missing 不炸)


def test_default_stages_definition():
    flat = [g for stage in DEFAULT_STAGES for g in stage]
    assert flat == ["skull", "jaw", "cheeks", "eyes", "nose", "mouth", "ears"]
