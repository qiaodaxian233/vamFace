"""FitResult.missing:目标 VaM 缺的精选 morph 名要被收集并带回。"""
import numpy as np

from vamface_mcp.fitting import Evaluator, FitConfig


class _FakeBridge:
    """set_morphs 回执带 missing 的假桥接。"""

    def __init__(self):
        self.calls = 0

    def set_morphs(self, atom, values, clamp=True):
        self.calls += 1
        missing = [n for n in values if n.startswith("Bogus")]
        return {"ok": True, "applied": len(values) - len(missing),
                "missing": missing}

    def screenshot(self, max_width=512):
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (127, 127, 127)).save(buf, format="PNG")
        return {"png_base64": base64.b64encode(buf.getvalue()).decode()}


class _FlatScorer:
    name = "flat"

    def score(self, target, candidate):
        return 0.5


def test_evaluator_collects_missing_union():
    bridge = _FakeBridge()
    cfg = FitConfig(atom="Person", use_cache=False)
    ev = Evaluator(bridge, np.zeros((8, 8, 3), dtype=np.uint8), _FlatScorer(), cfg)
    ev({"Nose Size": 0.1, "Bogus One": 0.2})
    ev({"Nose Size": 0.3, "Bogus Two": 0.4})
    assert ev.missing == {"Bogus One", "Bogus Two"}


def test_evaluator_missing_empty_when_all_present():
    bridge = _FakeBridge()
    cfg = FitConfig(atom="Person", use_cache=False)
    ev = Evaluator(bridge, np.zeros((8, 8, 3), dtype=np.uint8), _FlatScorer(), cfg)
    ev({"Nose Size": 0.1})
    assert ev.missing == set()

