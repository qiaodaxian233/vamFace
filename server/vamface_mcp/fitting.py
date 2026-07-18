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
# Scorers
# ---------------------------------------------------------------------------

class Scorer:
    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        raise NotImplementedError


class NullScorer(Scorer):
    """Fallback that always returns 0.0. Lets everything wire up headless."""

    reason = "no identity scorer available (install insightface)"

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        return 0.0


class ArcFaceScorer(Scorer):
    """Cosine similarity between ArcFace embeddings of the two faces."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # heavy, lazy

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._target_cache: Optional[np.ndarray] = None

    def _embed(self, img: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(img[:, :, ::-1])  # insightface wants BGR
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                   reverse=True)
        emb = faces[0].normed_embedding
        return np.asarray(emb, dtype=np.float32)

    def set_target(self, target: np.ndarray) -> bool:
        self._target_cache = self._embed(target)
        return self._target_cache is not None

    def score(self, target: np.ndarray, candidate: np.ndarray) -> float:
        if self._target_cache is None:
            if not self.set_target(target):
                return 0.0
        cand = self._embed(candidate)
        if cand is None:
            return 0.0
        return float(np.dot(self._target_cache, cand))


def build_scorer() -> Tuple[Scorer, Optional[str]]:
    """Return (scorer, warning). Never raises on missing deps."""
    try:
        return ArcFaceScorer(), None
    except Exception as e:  # ImportError or model download failure
        log.warning("ArcFace unavailable, degrading to NullScorer: %s", e)
        return NullScorer(), f"{NullScorer.reason}: {e}"


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

    best_morphs = {n: float(v) for n, v in zip(names, best_vec or x0)}
    return FitResult(best_score=best_score, best_morphs=best_morphs, history=history)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def fit_face(bridge: BridgeClient, target_image_path: str, cfg: FitConfig,
             optimizer: str = "cma") -> FitResult:
    target = load_image(target_image_path)
    scorer, warning = build_scorer()
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
    result.warning = warning
    return result
