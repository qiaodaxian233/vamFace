"""v0.7 本地校准模型:实测斜率 → 解方程,拟合从瞎蒙变定向。"""
import numpy as np
import pytest

from vamface_mcp import calibrate
from vamface_mcp.calibrate import (jacobian_polish, load_jacobian,
                                   probe_jacobian, save_jacobian, solve_step)
from vamface_mcp.fitting import FitConfig, fit_face, make_evaluator
from vamface_mcp.mock_vam import FaceRenderer, features_from_mock
from vamface_mcp.scorers import GeometryScorer


# ---------------------------------------------------------------------------
# solve_step:欠定系统上岭回归解出正确方向,步长有夹
# ---------------------------------------------------------------------------

def test_solve_step_recovers_direction_and_clips():
    J = {"M1": {"eye_gap": 0.05}, "M2": {"mouth_w": 0.04},
         "M3": {"eye_gap": 0.01, "mouth_w": -0.01}}
    residual = {"eye_gap": 0.10, "mouth_w": -0.08}  # 目标眼距更宽、嘴更窄
    d = solve_step(J, residual, ["M1", "M2", "M3"])
    assert d["M1"] > 0 and d["M2"] < 0                 # 方向对
    assert all(abs(v) <= 0.5 + 1e-9 for v in d.values())  # 步长夹在 step_clip 内


def test_solve_step_empty_when_nothing_shared():
    assert solve_step({"M": {"eye_gap": 1.0}}, {"jaw_len": 0.1}, ["M"]) == {}


# ---------------------------------------------------------------------------
# 缓存往返
# ---------------------------------------------------------------------------

def test_jacobian_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "CACHE_DIR", tmp_path)
    J = {"Nose Size": {"nose_len": 0.033}}
    save_jacobian(J, ["Nose Size"], 512, basis_tag="B=1")
    assert load_jacobian(["Nose Size"], 512, basis_tag="B=1") == J
    assert load_jacobian(["Nose Size"], 512, basis_tag="") is None   # 基底不同
    assert load_jacobian(["Nose Size"], 256, basis_tag="B=1") is None  # 宽度不同


# ---------------------------------------------------------------------------
# mock 全链路:探针量出真斜率;解方程拿小预算逼近隐藏目标
# ---------------------------------------------------------------------------

class _MockFitBridge:
    """直连 FaceRenderer 的桥接(不开 TCP),隐藏目标由测试指定。"""

    def __init__(self):
        self.renderer = FaceRenderer()
        self.state = {}

    def list_morphs(self, atom, filter="", region="", limit=200):
        from vamface_mcp.mock_vam import MORPH_DEFS
        rows = [{"name": n, "uid": f"mock/{n}", "region": g,
                 "value": self.state.get(n, 0.0), "min": lo, "max": hi}
                for n, (g, lo, hi) in MORPH_DEFS.items()]
        return {"count": len(rows), "total": len(rows), "morphs": rows}

    def set_morphs(self, atom, values, clamp=True):
        self.state.update({k: float(v) for k, v in values.items()})
        return {"ok": True, "applied": len(values), "missing": []}

    def get_morphs(self, atom, changed_only=True):
        return dict(self.state)

    def screenshot(self, max_width=512):
        import base64
        import io
        img = self.renderer.render(self.state)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"png_base64": base64.b64encode(buf.getvalue()).decode()}


HIDDEN = {"Eyes Width Spacing": 0.55, "Mouth Width": -0.45, "Nose Size": 0.4}
PROBE_NAMES = ["Eyes Width Spacing", "Mouth Width", "Nose Size", "Jaw Size"]


def _target_png(tmp_path):
    p = tmp_path / "target.png"
    FaceRenderer().render(HIDDEN).save(p)
    return str(p)


def _geo_scorer():
    return GeometryScorer(features_from_mock)


def test_probe_jacobian_measures_real_slopes(tmp_path):
    bridge = _MockFitBridge()
    cfg = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                    max_iters=99, use_cache=False, screenshot_width=512)
    geo = _geo_scorer()
    from vamface_mcp.fitting import load_image
    target = load_image(_target_png(tmp_path))
    ev = make_evaluator(bridge, target, geo, cfg)

    J, used, hist = probe_jacobian(ev, geo, {n: 0.0 for n in PROBE_NAMES},
                                   PROBE_NAMES, lambda n: (-1.0, 1.0))
    assert used == 1 + len(PROBE_NAMES)
    # mock 里 Eyes Width Spacing 直接驱动 eye_gap,斜率必须显著非零
    assert "Eyes Width Spacing" in J and "eye_gap" in J["Eyes Width Spacing"]
    assert abs(J["Eyes Width Spacing"]["eye_gap"]) > 1e-3
    # 探针结束场景要回到基线
    assert all(abs(v) < 1e-9 for v in bridge.state.values())


def test_jacobian_polish_converges_fast(tmp_path):
    """解方程 vs 黑盒:同一隐藏目标,校准 + 少量 GN 步就把几何残差打穿。"""
    bridge = _MockFitBridge()
    cfg = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                    max_iters=99, use_cache=False, screenshot_width=512)
    geo = _geo_scorer()
    from vamface_mcp.fitting import load_image
    target = load_image(_target_png(tmp_path))
    ev = make_evaluator(bridge, target, geo, cfg)

    base = {n: 0.0 for n in PROBE_NAMES}
    s_base = ev(dict(base))
    J, _, _ = probe_jacobian(ev, geo, base, PROBE_NAMES,
                             lambda n: (-1.0, 1.0))
    r0 = dict(geo.last_diff)  # 起点残差(evaluate(base) 之后的)
    ev(dict(base))             # 回到基线,拿干净的起点残差
    r0 = dict(geo.last_diff)
    bx, bs, used, _ = jacobian_polish(ev, geo, J, base, PROBE_NAMES,
                                      lambda n: (-1.0, 1.0), iters=6)
    ev(dict(bx))               # 在最优点上量收敛后残差
    r1 = dict(geo.last_diff)
    shrink = (abs(r1.get("eye_gap", 0)) + abs(r1.get("mouth_w", 0))) / \
             max(1e-9, abs(r0.get("eye_gap", 0)) + abs(r0.get("mouth_w", 0)))
    assert shrink < 0.4, f"主残差没打下去: 收敛比 {shrink:.2f}"
    assert bs >= s_base
    assert used <= 7
    # 隐藏目标的主方向要被解出来(符号正确)
    assert bx["Eyes Width Spacing"] > 0.25
    assert bx["Mouth Width"] < -0.15


def test_fit_face_with_jacobian_end_to_end(tmp_path, monkeypatch):
    """fit_face(use_jacobian=True) 全链路:校准落盘 + note 上报 + 分数达标。"""
    monkeypatch.setattr(calibrate, "CACHE_DIR", tmp_path / "cache")
    bridge = _MockFitBridge()
    scorer = GeometryScorer(features_from_mock)

    cfg = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                    max_iters=30, use_cache=False, screenshot_width=512)
    res = fit_face(bridge, _target_png(tmp_path), cfg, optimizer="greedy",
                   scorer=scorer, use_prior=True, neutralize=False,
                   use_jacobian=True)
    assert "新测" in res.jacobian_note
    assert res.best_score > 0.8

    # 第二跑:缓存命中,探针评估全免
    bridge2 = _MockFitBridge()
    res2 = fit_face(bridge2, _target_png(tmp_path), cfg, optimizer="greedy",
                    scorer=GeometryScorer(features_from_mock),
                    use_prior=True, neutralize=False, use_jacobian=True)
    assert "缓存复用" in res2.jacobian_note


def test_probe_baseline_retries_through_detection_flicker():
    """检测帧间抖动:基线前两拍检不出,第三拍成功 —— 探针不能一拍放弃。"""
    class _FlickerGeo:
        WEIGHTS = GeometryScorer.WEIGHTS

        def __init__(self):
            self.calls = 0
            self.last_diff = {}

    class _Ev:
        def __init__(self, geo):
            self.geo = geo
            self._bridge = type("B", (), {"set_morphs":
                                          staticmethod(lambda *a, **k: {})})()
            self._cfg = type("C", (), {"atom": "P"})()
            self.n = 0

        def __call__(self, vals):
            self.n += 1
            self.geo.calls += 1
            # 前两拍检不出脸,之后稳定
            self.geo.last_diff = ({} if self.geo.calls <= 2
                                  else {"eye_gap": 0.05})
            return 0.5

        def bump_epoch(self):
            pass

    geo = _FlickerGeo()
    ev = _Ev(geo)
    J, used, _ = probe_jacobian(ev, geo, {"M": 0.0}, ["M"],
                                lambda n: (-1.0, 1.0))
    assert used >= 3            # 重试消耗了评估
    assert "M" in J or J == {}  # 基线活了之后探针正常走(M 可能斜率为零)
    assert geo.calls >= 4       # 3 次基线尝试 + 至少 1 次探针


def test_fit_face_health_warning_when_detection_mostly_fails(tmp_path):
    """检不出脸的评估占比过高 → result.health 必须大声说话。"""
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tpath)

    bridge = _MockFitBridge()
    calls = {"n": 0}

    def flaky(img):
        calls["n"] += 1
        if calls["n"] == 1:                 # 目标照检得出
            return {"eye_gap": 0.3, "mouth_w": 0.2}
        return None                         # 渲染 candidate 全检不出(暗光场景)

    geo = GeometryScorer(extractor=flaky)
    cfg = FitConfig(atom="Person", morph_names=["Nose Size"],
                    max_iters=6, use_cache=False)
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=geo, use_prior=False, neutralize=False)
    assert res.health and "检不出人脸" in res.health


def test_fit_face_health_when_target_photo_undetectable(tmp_path):
    """目标照片本身检不出 → 最响的警报,一切分数无效。"""
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tpath)

    bridge = _MockFitBridge()
    geo = GeometryScorer(extractor=lambda img: None)
    cfg = FitConfig(atom="Person", morph_names=["Nose Size"],
                    max_iters=4, use_cache=False)
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=geo, use_prior=False, neutralize=False)
    assert "目标照片本身检不出" in res.health


# ---------------------------------------------------------------------------
# v0.7.3:增量校准累积 + 预算守门 + 基底扫描前清残留
# ---------------------------------------------------------------------------

def test_calibration_accumulates_across_starved_runs(tmp_path, monkeypatch):
    """预算饿死只测了一部分 → 下一跑接着测没测的,累计落盘(真机第七跑:
    默认预算 60 被基底吃光,43 个滑块只测了 1 个)。"""
    monkeypatch.setattr(calibrate, "CACHE_DIR", tmp_path / "cache")
    tp = _target_png(tmp_path)

    # 第一跑:预算掐到刚好只能测一部分
    bridge = _MockFitBridge()
    cfg1 = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                     max_iters=14, use_cache=False, screenshot_width=512)
    res1 = fit_face(bridge, tp, cfg1, optimizer="greedy",
                    scorer=GeometryScorer(features_from_mock),
                    use_prior=False, neutralize=False, use_jacobian=True)
    assert "累计" in res1.jacobian_note

    # 第二跑:足额预算,只补没测的,最终 4/4 全齐
    bridge2 = _MockFitBridge()
    cfg2 = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                     max_iters=40, use_cache=False, screenshot_width=512)
    res2 = fit_face(bridge2, tp, cfg2, optimizer="greedy",
                    scorer=GeometryScorer(features_from_mock),
                    use_prior=False, neutralize=False, use_jacobian=True)
    assert ("4/4" in res2.jacobian_note) or ("缓存复用(4/4" in res2.jacobian_note)


def test_calibration_budget_guard_recommends_number(tmp_path, monkeypatch):
    """预算连几个滑块都测不起 → 不白烧,直接给出确切的推荐预算。"""
    monkeypatch.setattr(calibrate, "CACHE_DIR", tmp_path / "cache")
    bridge = _MockFitBridge()
    cfg = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                    max_iters=12, use_cache=False, screenshot_width=512)
    res = fit_face(bridge, _target_png(tmp_path), cfg, optimizer="greedy",
                   scorer=GeometryScorer(features_from_mock),
                   use_prior=True, neutralize=False, use_jacobian=True)
    assert "预算不够校准" in res.jacobian_note
    assert "≥" in res.jacobian_note


def test_basis_baseline_clears_leftover_state(tmp_path):
    """上一跑的基底残留在场景里会抬高 baseline —— 扫描前必须清零全部候选。"""
    from tests.test_basis import _HeadBridge, _BrightScorer
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(tpath)

    bridge = _HeadBridge()
    bridge.state["TestFace"] = 1.0  # 残留:上次拟合留下的基底
    cfg = FitConfig(atom="Person", max_iters=14, use_cache=False,
                    morph_names=["Nose Size"])
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=_BrightScorer(), use_prior=False, neutralize=False,
                   use_basis=True)
    # 残留被清零后重新公平选拔,TestFace 仍然当选(它确实最像)
    assert res.basis == {"TestFace": 1.0}
    assert res.best_score >= 0.99


def test_final_gn_polish_recovers_after_cma_drift(tmp_path, monkeypatch):
    """CMA 末段漂走的几何残差,收尾 GN 要能压回去(分数不劣于 CMA 结果)。"""
    monkeypatch.setattr(calibrate, "CACHE_DIR", tmp_path / "cache")
    bridge = _MockFitBridge()
    cfg = FitConfig(atom="Person", morph_names=list(PROBE_NAMES),
                    max_iters=40, use_cache=False, screenshot_width=512)
    res = fit_face(bridge, _target_png(tmp_path), cfg, optimizer="greedy",
                   scorer=GeometryScorer(features_from_mock),
                   use_prior=False, neutralize=False, use_jacobian=True)
    assert res.best_score > 0.9
    assert "43" not in res.jacobian_note  # sanity: mock 只有 4 个滑块


def test_saturated_sliders_reported(tmp_path):
    """滑块顶到边界要点名 —— 连续几跑同一残差压不掉,用户该知道是物理极限。"""
    from PIL import Image
    from tests.test_resolution import _FlatScorer

    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (127, 127, 127)).save(tpath)

    class _B:
        def list_morphs(self, atom, filter="", region="", limit=200):
            rows = [{"name": "Nose Size", "uid": "x", "region": "nose",
                     "value": 0, "min": 0, "max": 0.3}]
            return {"count": 1, "total": 1, "morphs": rows}

        def set_morphs(self, atom, values, clamp=True):
            return {"ok": True, "applied": len(values), "missing": []}

        def screenshot(self, max_width=512):
            import base64, io
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (127, 127, 127)).save(buf, format="PNG")
            return {"png_base64": base64.b64encode(buf.getvalue()).decode()}

    class _WantMore(_FlatScorer):
        """分数 = 值本身:优化器必然把 Nose Size 推到上界 0.3。"""

        def __init__(self):
            self.last = 0.0

        def score(self, target, candidate):
            return self.last

    class _TrackBridge(_B):
        def __init__(self, sc):
            self.sc = sc

        def set_morphs(self, atom, values, clamp=True):
            if "Nose Size" in values:
                self.sc.last = float(values["Nose Size"])
            return super().set_morphs(atom, values, clamp)

    sc = _WantMore()
    bridge = _TrackBridge(sc)
    cfg = FitConfig(atom="Person", morph_names=["Nose Size"],
                    max_iters=12, use_cache=False)
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=sc, use_prior=False, neutralize=False)
    assert any(s.startswith("Nose Size=0.3") for s in res.saturated), res.saturated
