"""Face fitting: drive VaM morphs toward a target photo.

This is the heart of the automation. Given a target face photo, we search
the morph space so the rendered VaM face matches. The approach is black-box
optimization with a live VaM in the loop:

    propose morph vector
        -> bridge.set_morphs
        -> bridge.screenshot
        -> crop/align both faces
        -> score = identity similarity (higher is better)
    repeat, keeping the best.

Two scorers are provided:
  * ArcFaceScorer  — cosine similarity of face-recognition embeddings.
    Requires `insightface` + a face detector. This is the quantitative,
    "no LLM in the loop" path. Preferred when available.
  * NullScorer     — placeholder returning 0.0, so the module imports and
    the optimizer wiring can be tested without heavy deps installed.

Two optimizers:
  * greedy_coordinate  — cheap, dependency-free, good for smoke tests and
    for a small hand-picked set of "big lever" morphs.
  * cma_optimize       — CMA-ES over a chosen morph subset (needs `cma`).
    This is the real workhorse for a ~30–80 dim morph subset.

IMPORTANT design choice (see 对话记忆 错误5 — no hard gates on guesses):
if a scorer's dependency is missing we DEGRADE to NullScorer with a loud
warning rather than crashing. The MCP tool surfaces the warning to the
caller so they know the score is not meaningful yet.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bridge_client import BridgeClient

log = logging.getLogger("vamface.fit")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decode_png_b64(b64: str) -> np.ndarray:
    """Decode base64 PNG to an HxWx3 uint8 RGB array (needs Pillow)."""
    from PIL import Image  # local import so module loads without Pillow

    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img)


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Scorers —— 已迁到 scorers.py(v0.3:可切换的 real/anime/pixel 打分器栈)。
# 这里 re-export 旧名字,老的调用方(server.py / 外部脚本)不用改。
# ---------------------------------------------------------------------------

from .scorers import (ArcFaceScorer, NullScorer, PixelScorer,  # noqa: F401
                      Scorer, build_scorer_stack)


def build_scorer() -> Tuple[Scorer, Optional[str]]:
    """兼容旧签名:等价于 build_scorer_stack("real")。新代码请用后者。"""
    b = build_scorer_stack("real")
    return b.scorer, b.warning


# ---------------------------------------------------------------------------
# Fitting configuration and result
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    atom: str = "Person"
    morph_names: List[str] = field(default_factory=list)  # subset to optimize
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    default_bound: Tuple[float, float] = (-1.0, 1.0)
    max_iters: int = 60
    screenshot_width: int = 512
    seed: Optional[Dict[str, float]] = None  # initial morph values (from route A)
    # Called after every evaluation: (eval_count, score, rendered_image).
    # Used by the GUI for live preview; exceptions in the callback are
    # swallowed so a UI bug can't kill a long-running fit.
    on_eval: Optional[Callable[[int, float, np.ndarray], None]] = None


@dataclass
class FitResult:
    best_score: float
    best_morphs: Dict[str, float]
    history: List[float]
    warning: Optional[str] = None
    style: Optional[str] = None          # 实际生效的打分风格(auto 解析后)
    scorer_name: Optional[str] = None    # 打分器组合的可读名
    hints: List[str] = field(default_factory=list)  # 几何差 → 方向性提示


# ---------------------------------------------------------------------------
# The evaluation closure: morphs -> score, via the live bridge
# ---------------------------------------------------------------------------

def make_evaluator(bridge: BridgeClient, target: np.ndarray, scorer: Scorer,
                   cfg: FitConfig) -> Callable[[Dict[str, float]], float]:
    counter = {"n": 0}

    def evaluate(morphs: Dict[str, float]) -> float:
        bridge.set_morphs(cfg.atom, morphs, clamp=True)
        shot = bridge.screenshot(max_width=cfg.screenshot_width)
        candidate = decode_png_b64(shot["png_base64"])
        score = scorer.score(target, candidate)
        counter["n"] += 1
        if cfg.on_eval is not None:
            try:
                cfg.on_eval(counter["n"], score, candidate)
            except Exception:
                log.exception("on_eval callback failed (ignored)")
        return score

    return evaluate


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def _bounds_for(cfg: FitConfig, name: str) -> Tuple[float, float]:
    return cfg.bounds.get(name, cfg.default_bound)


def greedy_coordinate(evaluate: Callable[[Dict[str, float]], float],
                      cfg: FitConfig, step: float = 0.25) -> FitResult:
    """Dependency-free coordinate ascent. Good for smoke tests / few morphs."""
    current = dict(cfg.seed or {name: 0.0 for name in cfg.morph_names})
    for name in cfg.morph_names:
        current.setdefault(name, 0.0)

    best_score = evaluate(current)
    history = [best_score]
    iters = 0

    while iters < cfg.max_iters:
        improved = False
        for name in cfg.morph_names:
            lo, hi = _bounds_for(cfg, name)
            for delta in (step, -step):
                if iters >= cfg.max_iters:
                    break
                trial = dict(current)
                trial[name] = float(np.clip(trial[name] + delta, lo, hi))
                if abs(trial[name] - current[name]) < 1e-9:
                    continue
                s = evaluate(trial)
                iters += 1
                history.append(max(best_score, s))
                if s > best_score:
                    best_score, current, improved = s, trial, True
        if not improved:
            step *= 0.5
            if step < 0.02:
                break
    return FitResult(best_score=best_score, best_morphs=current, history=history)


def cma_optimize(evaluate: Callable[[Dict[str, float]], float],
                 cfg: FitConfig, sigma0: float = 0.3) -> FitResult:
    """CMA-ES over the chosen morph subset. Needs the `cma` package."""
    import cma  # lazy

    names = list(cfg.morph_names)
    if not names:
        raise ValueError("cma_optimize requires cfg.morph_names to be non-empty")

    seed = cfg.seed or {}
    x0 = [float(seed.get(n, 0.0)) for n in names]
    lows = [_bounds_for(cfg, n)[0] for n in names]
    highs = [_bounds_for(cfg, n)[1] for n in names]

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"bounds": [lows, highs], "maxfevals": cfg.max_iters, "verbose": -9},
    )

    best_score = -1e9
    best_vec: Optional[Sequence[float]] = None
    history: List[float] = []

    while not es.stop():
        solutions = es.ask()
        costs = []
        for vec in solutions:
            morphs = {n: float(v) for n, v in zip(names, vec)}
            s = evaluate(morphs)          # similarity, higher better
            costs.append(-s)              # CMA minimizes
            if s > best_score:
                best_score, best_vec = s, vec
            history.append(best_score)
        es.tell(solutions, costs)

    # 注意:best_vec 是 numpy 数组,`best_vec or x0` 会抛 ValueError(数组真值
    # 歧义)。这是 v0.1 的遗留 bug,mock 端到端测试抓出来的。
    final_vec = x0 if best_vec is None else best_vec
    best_morphs = {n: float(v) for n, v in zip(names, final_vec)}
    return FitResult(best_score=best_score, best_morphs=best_morphs, history=history)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def fit_face(bridge: BridgeClient, target_image_path: str, cfg: FitConfig,
             optimizer: str = "cma", style: str = "auto",
             anime_onnx: Optional[str] = None) -> FitResult:
    """style: auto | real | anime | pixel(见 scorers.build_scorer_stack)。"""
    target = load_image(target_image_path)
    build = build_scorer_stack(style, target=target, anime_onnx=anime_onnx)
    scorer = build.scorer
    if isinstance(scorer, ArcFaceScorer):
        scorer.set_target(target)

    evaluate = make_evaluator(bridge, target, scorer, cfg)

    if optimizer == "greedy":
        result = greedy_coordinate(evaluate, cfg)
    elif optimizer == "cma":
        result = cma_optimize(evaluate, cfg)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    # leave VaM showing the best result
    bridge.set_morphs(cfg.atom, result.best_morphs, clamp=True)
    result.warning = build.warning
    result.style = build.style
    result.scorer_name = scorer.name
    # 让最优解成为"最近一次评估",这样 hints 描述的是最终结果的残差
    try:
        shot = bridge.screenshot(max_width=cfg.screenshot_width)
        scorer.score(target, decode_png_b64(shot["png_base64"]))
        result.hints = scorer.hints()
    except Exception:
        log.exception("final hints pass failed (ignored)")
    return result
