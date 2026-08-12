"""打分器单测:不依赖任何重型模型。

GeometryScorer 的提取器是可注入的,所以这里用合成特征函数直接验证:
分数的单调性、交集键比较、方向性提示的正负号。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vamface_mcp.scorers import (CompositeScorer, CroppedScorer, GeometryScorer,
                                 NullScorer, PixelScorer, build_scorer_stack)


def _img(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# PixelScorer
# ---------------------------------------------------------------------------

def test_pixel_identity_is_one():
    a = _img(0)
    assert PixelScorer().score(a, a.copy()) == 1.0


def test_pixel_monotonic_in_difference():
    s = PixelScorer()
    base = np.full((64, 64, 3), 128, dtype=np.uint8)
    near = np.clip(base.astype(int) + 10, 0, 255).astype(np.uint8)
    far = np.clip(base.astype(int) + 80, 0, 255).astype(np.uint8)
    assert s.score(base, near) > s.score(base, far)


# ---------------------------------------------------------------------------
# GeometryScorer(注入合成提取器)
# ---------------------------------------------------------------------------

def _extractor_from_table(table):
    """用 id(img) 查表返回预设特征 —— 模拟检测器。"""
    return lambda img: table.get(id(img))


def test_geometry_perfect_match_scores_one():
    t, c = _img(1), _img(2)
    feats = {"eye_gap": 0.45, "mouth_w": 0.3, "face_aspect": 1.3}
    g = GeometryScorer(_extractor_from_table({id(t): dict(feats), id(c): dict(feats)}))
    assert g.score(t, c) == 1.0
    assert g.hints() == []  # 没差就没提示


def test_geometry_monotonic_and_intersection_only():
    t, near, far = _img(1), _img(2), _img(3)
    g = GeometryScorer(_extractor_from_table({
        id(t): {"eye_gap": 0.45, "mouth_w": 0.30},
        id(near): {"eye_gap": 0.47, "mouth_w": 0.30, "jaw_len": 0.2},  # 多的键被忽略
        id(far): {"eye_gap": 0.60, "mouth_w": 0.40},
    }))
    assert g.score(t, near) > g.score(t, far)


def test_geometry_detect_miss_returns_zero_and_counts():
    t, c = _img(1), _img(2)
    g = GeometryScorer(_extractor_from_table({id(t): {"eye_gap": 0.4}}))  # c 检不出
    assert g.score(t, c) == 0.0
    assert g.detect_misses == 1


def test_geometry_hints_direction():
    """目标两眼间距更大(Δ>0)→ 提示应该指向 Eyes Width Spacing ↑。"""
    t, c = _img(1), _img(2)
    g = GeometryScorer(_extractor_from_table({
        id(t): {"eye_gap": 0.55, "mouth_w": 0.30},
        id(c): {"eye_gap": 0.40, "mouth_w": 0.30},
    }))
    g.score(t, c)
    hints = g.hints()
    assert len(hints) == 1
    assert "Eyes Width Spacing ↑" in hints[0]
    # 反过来:目标间距更小 → ↓
    g2 = GeometryScorer(_extractor_from_table({
        id(t): {"eye_gap": 0.40},
        id(c): {"eye_gap": 0.55},
    }))
    g2.score(t, c)
    assert "Eyes Width Spacing ↓" in g2.hints()[0]


# ---------------------------------------------------------------------------
# CompositeScorer / build_scorer_stack
# ---------------------------------------------------------------------------

class _Const:
    def __init__(self, v, name="const"):
        self.v, self.name = v, name

    def score(self, t, c):
        return self.v

    def hints(self):
        return [f"hint-{self.name}"]


def test_composite_weighted_mean_and_hint_merge():
    comp = CompositeScorer([(_Const(1.0, "a"), 3.0), (_Const(0.0, "b"), 1.0)])
    assert abs(comp.score(None, None) - 0.75) < 1e-9
    assert comp.hints() == ["hint-a", "hint-b"]


def test_build_pixel_style():
    b = build_scorer_stack("pixel")
    # v0.4: pixel 打分器带主体框裁剪,防背景/构图污染分数
    assert isinstance(b.scorer, CroppedScorer)
    assert isinstance(b.scorer.inner, PixelScorer)
    assert b.style == "pixel"


def test_build_never_raises_and_degrades_with_warning():
    """构建必须永不抛异常:装了重依赖就用真打分器,没装就带 warning 降级。"""
    import importlib.util

    b_real = build_scorer_stack("real")
    b_anime = build_scorer_stack("anime")
    assert b_real.scorer is not None and b_anime.scorer is not None
    if importlib.util.find_spec("insightface") is None:
        assert isinstance(b_real.scorer, NullScorer)
        assert b_real.warning
    if importlib.util.find_spec("animeface") is None:
        assert isinstance(b_anime.scorer, CroppedScorer)  # anime 降到 pixel(带裁剪)
        assert isinstance(b_anime.scorer.inner, PixelScorer)
        assert b_anime.warning


def test_build_auto_without_target_falls_to_pixel():
    b = build_scorer_stack("auto", target=None)
    assert b.style == "pixel"
    assert b.warning


# ---------------------------------------------------------------------------
# v0.7.1:onnxruntime 后端优选(打分器上 GPU)
# ---------------------------------------------------------------------------

def test_preferred_providers_gpu_first(monkeypatch):
    import sys, types
    from vamface_mcp import scorers

    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: [
        "CPUExecutionProvider", "DmlExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    provs = scorers.preferred_providers()
    assert provs[0] == "DmlExecutionProvider"      # GPU 排前
    assert provs[-1] == "CPUExecutionProvider"     # CPU 永远兜底


def test_preferred_providers_cpu_only(monkeypatch):
    import sys, types
    from vamface_mcp import scorers

    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    assert scorers.preferred_providers() == ["CPUExecutionProvider"]


def test_preferred_providers_no_ort(monkeypatch):
    import sys
    from vamface_mcp import scorers

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert scorers.preferred_providers() == []


def test_actual_provider_reads_session(monkeypatch):
    from vamface_mcp.scorers import _actual_provider

    class _Sess:
        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    class _Model:
        session = _Sess()

    class _App:
        models = {"detection": _Model()}

    assert _actual_provider(_App()) == "DmlExecutionProvider"
    assert _actual_provider(object()) is None  # 没 session 结构就别硬猜


def test_backend_report_states(monkeypatch):
    import sys, types
    from vamface_mcp import scorers

    fake = types.ModuleType("onnxruntime")
    fake.__version__ = "1.23.0"
    fake.get_available_providers = lambda: ["DmlExecutionProvider",
                                            "CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    r = scorers.backend_report(deep=False)
    assert "✅" in r and "DirectML" in r

    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    r = scorers.backend_report(deep=False)
    assert "⚠️" in r and "force-reinstall" in r

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    r = scorers.backend_report(deep=False)
    assert "❌" in r and "残骸" in r
