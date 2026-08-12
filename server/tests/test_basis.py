"""v0.6 角色基底粗定位:先扫整头角色 morph 跨大距离,再精修。

背景:精选 morph 是增量滑块,从默认脸出发几百次评估爬不到脸型差异大
的目标(真机实测 0.2x 分封顶)。用户 Head 区装了几十个整头角色 morph,
先扫一遍找最像的当起点。
"""
import base64
import io

import numpy as np

from vamface_mcp.fitting import (FitConfig, basis_search,
                                 character_head_candidates, fit_face)
from vamface_mcp.scorers import Scorer


# ---------------------------------------------------------------------------
# 候选识别:Head 区、名字无特征词
# ---------------------------------------------------------------------------

def _row(name, region="Head"):
    return {"name": name, "uid": f"x/{name}", "region": region,
            "value": 0, "min": -1, "max": 1}


def test_character_head_candidates_heuristic():
    rows = [
        _row("Aiko 6 Head"), _row("Sumiko Head"), _row("Kimi (REN)"),
        _row("Mei Lin 6 Head"),
        # 特征滑块要被排除
        _row("Head Scale"), _row("Head Width"), _row("Cranium Size"),
        _row("ForeHead Top Width"), _row("Neck Width"), _row("Lower Mouth Puffy"),
        # 非 Head 区一律不算
        _row("Nose Size", region="Nose"), _row("Tara Face", region="Face"),
    ]
    cand = character_head_candidates(rows)
    assert cand == ["Aiko 6 Head", "Sumiko Head", "Kimi (REN)", "Mei Lin 6 Head"]


def test_character_head_candidates_rejects_part_prefixed_features():
    """用户新装的包把特征滑块混进了 Head 区(真机实锤)——不能进基底扫描,
    不然 'Face Sag'=1.0 这种都可能被当成角色头采纳。"""
    rows = [_row(n) for n in
            ["Eye Bags", "Eye Sag Under", "Eye Socket", "Eyelid Size",
             "Face Sag", "Nostrils Lower Depth", "fab 2",
             "Carmen Face", "Tara Face B"]]
    cand = character_head_candidates(rows)
    assert cand == ["fab 2", "Carmen Face", "Tara Face B"]


# ---------------------------------------------------------------------------
# basis_search 状态机:清上一个、选冠军、权重微调、落定/放弃
# ---------------------------------------------------------------------------

class _StateEval:
    """有状态的假评估器:按增量语义累积写入,B=1.0 时分数最高。"""

    def __init__(self, scores):
        self.state = {}
        self.scores = scores  # (名字, 权重) -> 分数
        self.epoch_bumps = 0

    def __call__(self, vals):
        self.state.update(vals)
        active = [(n, v) for n, v in sorted(self.state.items()) if abs(v) > 1e-9]
        if len(active) != 1:
            return 0.05  # 两个头叠加 = 鬼脸,分数必须垫底
        return self.scores.get(active[0], 0.1)

    def bump_epoch(self):
        self.epoch_bumps += 1


class _RecordingBridge:
    def __init__(self):
        self.settled = None

    def set_morphs(self, atom, values, clamp=True):
        self.settled = dict(values)
        return {"ok": True, "applied": len(values), "missing": []}


def test_basis_search_picks_winner_and_zeroes_losers():
    ev = _StateEval({("A", 1.0): 0.3, ("B", 1.0): 0.8, ("B", 0.6): 0.6,
                     ("C", 1.0): 0.4})
    bridge = _RecordingBridge()
    basis, used, hist, invalid = basis_search(ev, bridge, "Person",
                                              ["A", "B", "C"], baseline=0.2)
    assert basis == {"B": 1.0}          # 冠军 @ 最优权重(0.6 没赢过 1.0)
    assert used == 6                     # 3 候选 + top3 各 1 次复赛
    assert invalid == []
    assert bridge.settled["A"] == 0.0 and bridge.settled["C"] == 0.0
    assert bridge.settled["B"] == 1.0
    assert ev.epoch_bumps == 1           # 落定绕过 evaluate,必须作废缓存


def test_basis_search_weight_refinement_wins():
    ev = _StateEval({("A", 1.0): 0.5, ("A", 0.6): 0.9})
    bridge = _RecordingBridge()
    basis, used, _, _ = basis_search(ev, bridge, "Person", ["A"], baseline=0.2)
    assert basis == {"A": 0.6}
    assert bridge.settled["A"] == 0.6


def test_basis_search_topk_rescues_noisy_runnerup():
    """首轮亚军在复赛权重上反超 —— top-k 复赛就是为噪声 argmax 兜底的。"""
    ev = _StateEval({("A", 1.0): 0.50, ("B", 1.0): 0.45, ("C", 1.0): 0.10,
                     ("A", 0.6): 0.40, ("B", 0.6): 0.80, ("C", 0.6): 0.10})
    bridge = _RecordingBridge()
    basis, used, _, _ = basis_search(ev, bridge, "Person", ["A", "B", "C"],
                                     baseline=0.2)
    assert basis == {"B": 0.6}
    assert bridge.settled["B"] == 0.6 and bridge.settled["A"] == 0.0


def test_basis_search_invalidates_unsettable_candidate():
    """真机怪相:morph 列表里有但 set 被拒(Izarra/Lilith 6 Head 实锤)——
    那次评估拍到的是没变化的脸,分数是假的,候选必须作废。"""

    class _GhostEval(_StateEval):
        """'Ghost' 写不进去:状态不更新 + 记 missing;全零脸给假高分。"""

        def __init__(self, scores, ghost):
            super().__init__(scores)
            self.ghost = ghost
            self.missing = set()

        def __call__(self, vals):
            applied = dict(vals)
            if applied.pop(self.ghost, 0.0):
                self.missing.add(self.ghost)
            self.state.update(applied)
            active = [(n, v) for n, v in sorted(self.state.items())
                      if abs(v) > 1e-9]
            if not active:
                return 0.99  # 没写进任何东西 = 拍到基线脸的假高分
            if len(active) != 1:
                return 0.05
            return self.scores.get(active[0], 0.1)

    ev = _GhostEval({("A", 1.0): 0.4, ("A", 0.6): 0.3}, ghost="Ghost")
    bridge = _RecordingBridge()
    basis, used, _, invalid = basis_search(ev, bridge, "Person",
                                           ["Ghost", "A"], baseline=0.2)
    assert invalid == ["Ghost"]
    assert basis == {"A": 1.0}           # 假高分 0.99 没能让 Ghost 当选


def test_basis_search_declines_when_nobody_beats_baseline():
    ev = _StateEval({("A", 1.0): 0.1, ("B", 1.0): 0.15})
    bridge = _RecordingBridge()
    basis, used, _, _ = basis_search(ev, bridge, "Person", ["A", "B"],
                                     baseline=0.5)
    assert basis == {}                             # 基底只能帮忙不能帮倒忙
    assert set(bridge.settled.items()) == {("A", 0.0), ("B", 0.0)}


def test_basis_search_respects_budget():
    ev = _StateEval({(f"C{i}", 1.0): 0.3 for i in range(10)})
    bridge = _RecordingBridge()
    _, used, _, _ = basis_search(ev, bridge, "Person",
                                 [f"C{i}" for i in range(10)],
                                 baseline=0.0, budget=4)
    assert used <= 4


# ---------------------------------------------------------------------------
# fit_face 集成:基底进 best_morphs(.vap 要带脸型),没候选时静默跳过
# ---------------------------------------------------------------------------

class _HeadBridge:
    """带一个角色头 'TestFace' 的假 VaM;截图亮度 = TestFace 权重。"""

    def __init__(self):
        self.state = {}

    def list_morphs(self, atom, filter="", region="", limit=200):
        rows = [
            {"name": "TestFace", "uid": "x/TestFace", "region": "Head",
             "value": 0, "min": 0, "max": 1},
            {"name": "Nose Size", "uid": "x/Nose Size", "region": "Nose",
             "value": 0, "min": -1, "max": 1},
        ]
        return {"count": len(rows), "total": len(rows), "morphs": rows}

    def set_morphs(self, atom, values, clamp=True):
        self.state.update(values)
        return {"ok": True, "applied": len(values), "missing": []}

    def get_morphs(self, atom, changed_only=True):
        return dict(self.state)

    def screenshot(self, max_width=512):
        from PIL import Image
        v = int(255 * max(0.0, min(1.0, self.state.get("TestFace", 0.0))))
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (v, v, v)).save(buf, format="PNG")
        return {"png_base64": base64.b64encode(buf.getvalue()).decode()}


class _BrightScorer(Scorer):
    """目标是白图:越亮越像 → 应该把 TestFace 推到 1.0。"""

    name = "bright"

    def score(self, target, candidate):
        return float(candidate.mean()) / 255.0


def test_fit_face_adopts_basis_into_vap(tmp_path):
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(tpath)

    bridge = _HeadBridge()
    cfg = FitConfig(atom="Person", max_iters=14, use_cache=False,
                    morph_names=["Nose Size"])
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=_BrightScorer(), use_prior=False, neutralize=False,
                   use_basis=True)
    assert res.basis == {"TestFace": 1.0}
    assert res.best_morphs.get("TestFace") == 1.0  # 基底进 .vap
    assert res.best_score >= 0.99


def test_fit_face_basis_skips_gracefully_without_candidates(tmp_path):
    from PIL import Image
    tpath = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (127, 127, 127)).save(tpath)

    from tests.test_resolution import _AliasBridge, _FlatScorer
    bridge = _AliasBridge()  # 名单里没有 Head 区
    cfg = FitConfig(atom="Person", max_iters=6, use_cache=False,
                    morph_names=["Nose Size"])
    res = fit_face(bridge, str(tpath), cfg, optimizer="greedy",
                   scorer=_FlatScorer(), use_prior=False, neutralize=False,
                   use_basis=True)
    assert res.basis == {}
