"""v0.5.4 别名解析:概念槽位 → 目标 VaM 实际 morph 名。

真机第二/三跑的两个实锤问题:
  1. 44 个精选名缺 16 个,36% 搜索维度是死区 → 解析后剔除死维度;
  2. hints 推荐 missing 列表里的 morph(推荐一根不存在的滑块)→
     提示按实际可用性过滤/改名。
"""
import base64
import io

import numpy as np

from vamface_mcp.fitting import FitConfig, fit_face
from vamface_mcp.morph_presets import MORPH_ALIASES, norm_name, resolve_names
from vamface_mcp.priors import FEATURE_TO_MORPHS
from vamface_mcp.scorers import _HINT_MAP, GeometryScorer, Scorer


# ---------------------------------------------------------------------------
# resolve_names 单测
# ---------------------------------------------------------------------------

def test_resolve_exact_and_normalized():
    r = resolve_names(["Nose Size", "Face Long"],
                      ["nose size", "Face-Long", "Other"])
    assert r.mapping == {"Nose Size": "nose size", "Face Long": "Face-Long"}
    assert r.renamed == {"Nose Size": "nose size", "Face Long": "Face-Long"}
    assert r.unresolved == []


def test_resolve_identical_name_not_in_renamed():
    r = resolve_names(["Nose Size"], ["Nose Size"])
    assert r.mapping == {"Nose Size": "Nose Size"}
    assert r.renamed == {}


def test_resolve_via_alias():
    # "Eyes Width Spacing" 缺失,但别名 "Eye Spacing" 在
    r = resolve_names(["Eyes Width Spacing"], ["Eye Spacing", "Nose Size"])
    assert r.mapping["Eyes Width Spacing"] == "Eye Spacing"
    assert r.renamed["Eyes Width Spacing"] == "Eye Spacing"


def test_resolve_unresolved_and_actual_passthrough():
    r = resolve_names(["Cranium Shape"], ["Nose Size"])
    assert r.unresolved == ["Cranium Shape"]
    assert r.actual("Cranium Shape") == "Cranium Shape"  # 原样返回
    assert r.to_actual(["Cranium Shape", "Nose Size"]) == ["Nose Size"]  # 死维度剔除


def test_resolve_no_substring_false_positive():
    # 归一化只做精确等值:"Nose Size" 绝不能命中 "Nose Size Depth"
    r = resolve_names(["Nose Size"], ["Nose Size Depth"])
    assert r.unresolved == ["Nose Size"]


def test_resolve_one_actual_one_slot():
    # 同一实际 morph 不能被两个槽位占用(先到先得)
    r = resolve_names(["Lips Width", "Mouth Width"], ["Lip Width"])
    taken = list(r.mapping.values())
    assert len(taken) == len(set(taken))
    assert len(r.mapping) + len(r.unresolved) == 2


def test_alias_table_selfconsistent():
    # 别名不产生歧义:任何归一化后的候选名只属于一个概念槽位
    seen = {}
    for canon, aliases in MORPH_ALIASES.items():
        for cand in [canon] + aliases:
            key = norm_name(cand)
            assert seen.setdefault(key, canon) == canon, \
                f"别名 {cand!r} 同时属于 {seen[key]!r} 和 {canon!r}"


# ---------------------------------------------------------------------------
# hints:方向从 FEATURE_TO_MORPHS 推导 + 按可用性过滤/改名
# ---------------------------------------------------------------------------

def _geo_with_diff(diff):
    g = GeometryScorer(extractor=lambda img: None)
    g.last_diff = dict(diff)
    return g


def test_hint_map_and_feature_map_in_sync():
    # 单一方向真相源:每个提示特征都必须能从 FEATURE_TO_MORPHS 拿到 morph 方向
    assert set(_HINT_MAP) == set(FEATURE_TO_MORPHS)


def test_hints_direction_derived_from_gains():
    g = _geo_with_diff({"face_aspect": -0.07})
    (h,) = g.hints()
    # face_aspect: [(Face Long,+),(Face Round,-)],Δ<0 → Face Long ↓ / Face Round ↑
    assert "Face Long ↓" in h and "Face Round ↑" in h


def test_hints_filter_unavailable_morph():
    g = _geo_with_diff({"eye_gap": -0.05, "nose_len": -0.04})
    g.set_morph_availability(["Nose Size", "Nose Height"])  # 没有 Eyes Width Spacing
    hs = g.hints()
    joined = "\n".join(hs)
    assert "Eyes Width Spacing ↓" not in joined          # 不再推荐不存在的滑块
    assert "缺失" in joined                               # 但特征差本身仍然报告
    assert "Nose Size ↓" in joined and "Nose Height ↓" in joined


def test_hints_show_renamed_actual_name():
    g = _geo_with_diff({"eye_gap": +0.05})
    g.set_morph_availability(["Eye Spacing"],
                             rename={"Eyes Width Spacing": "Eye Spacing"})
    (h,) = g.hints()
    assert "Eye Spacing ↑" in h


# ---------------------------------------------------------------------------
# fit_face 端到端:解析生效(改名参与拟合、死维度剔除进 missing)
# ---------------------------------------------------------------------------

class _AliasBridge:
    """一个装了 'Eye Spacing'(别名)和 'Nose Size',缺 'Cranium Shape' 的假 VaM。"""

    AVAILABLE = ["Eye Spacing", "Nose Size"]

    def __init__(self):
        self.set_names = set()

    def list_morphs(self, atom, filter="", region="", limit=200):
        rows = [{"name": n, "uid": f"fake/{n}", "region": "face",
                 "value": 0.0, "min": -1, "max": 1} for n in self.AVAILABLE]
        return {"count": len(rows), "total": len(rows), "morphs": rows}

    def set_morphs(self, atom, values, clamp=True):
        self.set_names.update(values)
        missing = [n for n in values if n not in self.AVAILABLE]
        return {"ok": True, "applied": len(values) - len(missing),
                "missing": missing}

    def get_morphs(self, atom, changed_only=True):
        return {}

    def screenshot(self, max_width=512):
        buf = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (8, 8), (127, 127, 127)).save(buf, format="PNG")
        return {"png_base64": base64.b64encode(buf.getvalue()).decode()}


class _FlatScorer(Scorer):
    name = "flat"

    def score(self, target, candidate):
        return 0.5


def test_fit_face_resolution_end_to_end(tmp_path):
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (127, 127, 127)).save(tpath)

    bridge = _AliasBridge()
    cfg = FitConfig(atom="Person", max_iters=6, use_cache=False,
                    morph_names=["Eyes Width Spacing", "Nose Size",
                                 "Cranium Shape"])
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=_FlatScorer(), use_prior=False, neutralize=False)

    # 别名解析:概念名 Eyes Width Spacing → 实际名 Eye Spacing
    assert res.renamed == {"Eyes Width Spacing": "Eye Spacing"}
    # 解析不到的概念是死维度:不进搜索、进 missing
    assert "Cranium Shape" in res.missing
    assert "Cranium Shape" not in bridge.set_names
    # 优化循环用的是实际名,产出的 .vap 键也是实际名
    assert "Eye Spacing" in bridge.set_names
    assert set(res.best_morphs) <= {"Eye Spacing", "Nose Size"}


def test_fit_face_without_list_morphs_still_works(tmp_path):
    """list_morphs 缺席(老插件/测试桩)时解析静默跳过,不阻塞拟合。"""
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (127, 127, 127)).save(tpath)

    class _NoListBridge(_AliasBridge):
        AVAILABLE = ["Nose Size"]

        def list_morphs(self, *a, **kw):  # 模拟老插件:命令不存在
            raise RuntimeError("unknown command")

    bridge = _NoListBridge()
    cfg = FitConfig(atom="Person", max_iters=4, use_cache=False,
                    morph_names=["Nose Size", "Bogus Thing"])
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=_FlatScorer(), use_prior=False, neutralize=False)
    assert res.renamed == {}
    assert "Bogus Thing" in res.missing  # 回执兜底路径仍然工作
