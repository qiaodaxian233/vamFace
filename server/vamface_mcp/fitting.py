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
                      Scorer, build_scorer_stack, find_geometry_scorer)


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
    use_cache: bool = True  # 重访相同 morph 向量时复用分数,省真机截图来回
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
    prior_seed: Dict[str, float] = field(default_factory=dict)  # 先验给出的初始种子
    neutralized: List[str] = field(default_factory=list)  # 拟合前清零的表情 morph
    stage_count: int = 1                 # coarse-to-fine 的阶段数
    cache_hits: int = 0                  # 截图缓存命中次数(省下的真机来回)


# ---------------------------------------------------------------------------
# The evaluation closure: morphs -> score, via the live bridge
# ---------------------------------------------------------------------------

class Evaluator:
    """morphs -> score,经由活桥接。可选缓存:重访相同向量直接复用分数。

    缓存 key = (epoch, 量化到 1e-4 的完整入参向量)。set_morphs 是**增量**语义,
    所以任何绕过 evaluate 的 set_morphs(表情归一化、阶段冻结)都会改变底下的
    真实状态 —— 调用方必须在那之后 bump_epoch(),让旧缓存作废。这就是为什么
    缓存只做"精确重访去重",不做近邻插值:宁可少省,不能错。
    """

    def __init__(self, bridge: BridgeClient, target: np.ndarray, scorer: Scorer,
                 cfg: FitConfig) -> None:
        self._bridge = bridge
        self._target = target
        self._scorer = scorer
        self._cfg = cfg
        self._n = 0
        self._epoch = 0
        self._cache: Dict[tuple, float] = {}
        self.cache_hits = 0

    def bump_epoch(self) -> None:
        """底层 morph 状态被 evaluate 之外的写动过之后调用,作废旧缓存。"""
        self._epoch += 1

    def _key(self, morphs: Dict[str, float]) -> tuple:
        return (self._epoch,
                tuple(sorted((n, round(float(v), 4)) for n, v in morphs.items())))

    def __call__(self, morphs: Dict[str, float]) -> float:
        cfg = self._cfg
        if cfg.use_cache:
            key = self._key(morphs)
            hit = self._cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit
        self._bridge.set_morphs(cfg.atom, morphs, clamp=True)
        shot = self._bridge.screenshot(max_width=cfg.screenshot_width)
        candidate = decode_png_b64(shot["png_base64"])
        score = self._scorer.score(self._target, candidate)
        self._n += 1
        if cfg.use_cache:
            self._cache[key] = score
        if cfg.on_eval is not None:
            try:
                cfg.on_eval(self._n, score, candidate)
            except Exception:
                log.exception("on_eval callback failed (ignored)")
        return score


def make_evaluator(bridge: BridgeClient, target: np.ndarray, scorer: Scorer,
                   cfg: FitConfig) -> Evaluator:
    """兼容旧签名:返回可调用的 Evaluator(带 .cache_hits / .bump_epoch)。"""
    return Evaluator(bridge, target, scorer, cfg)


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

def neutralize_expression(bridge: BridgeClient, atom: str) -> Dict[str, object]:
    """拟合前清零表情类 morph,防"用表情凑相似度"的作弊解。

    只动**已改动**且名字命中表情模式的 morph(get_morphs(changed_only=True)),
    身份 morph 一律不碰。眼球朝向/头部姿态的归正是 storable 参数不是 morph,
    属于真机验证清单(protocol.md)—— 这里不猜参数名。
    """
    from .morph_presets import is_expression_morph

    changed = bridge.get_morphs(atom, changed_only=True)
    targets = {n: 0.0 for n in changed if is_expression_morph(n)}
    if targets:
        bridge.set_morphs(atom, targets, clamp=True)
    return {"zeroed": sorted(targets), "checked": len(changed)}


# coarse-to-fine 的默认阶段划分:先定轮廓,冻结,再修五官
DEFAULT_STAGES: List[List[str]] = [
    ["skull", "jaw", "cheeks"],
    ["eyes", "nose", "mouth", "ears"],
]


def _stage_name_lists(cfg: FitConfig, stages: List[List[str]]) -> List[List[str]]:
    """分组名 → morph 名列表,并与 cfg.morph_names 求交(尊重用户的 groups 选择)。

    cfg.morph_names 里不属于任何所选分组的名字(用户自定义 morph)归入末阶段。
    """
    from .morph_presets import FACE_MORPH_GROUPS

    allowed = set(cfg.morph_names)
    out: List[List[str]] = []
    used: set = set()
    for groups in stages:
        names = [n for g in groups for n in FACE_MORPH_GROUPS.get(g, [])
                 if n in allowed and n not in used]
        used.update(names)
        if names:
            out.append(names)
    leftovers = [n for n in cfg.morph_names if n not in used]
    if leftovers:
        if out:
            out[-1].extend(leftovers)
        else:
            out.append(leftovers)
    return out


def fit_face(bridge: BridgeClient, target_image_path: str, cfg: FitConfig,
             optimizer: str = "cma", style: str = "auto",
             anime_onnx: Optional[str] = None,
             scorer: Optional[Scorer] = None,
             use_prior: bool = True, neutralize: bool = True,
             coarse_to_fine: bool = False,
             stages: Optional[List[List[str]]] = None) -> FitResult:
    """v0.4 拟合入口:表情归一化 → 先验探针 → (分阶段)黑盒优化 → 提示。

    style: auto | real | anime | pixel(见 scorers.build_scorer_stack)
    scorer: 直接注入打分器(测试/自定义用);给了就跳过 build_scorer_stack
    use_prior: 打分器里有 GeometryScorer 时,先做一次基线评估拿特征差,
               换算成初始种子(CMA x0 / greedy 起点+坐标顺序)。探针那次
               评估计入预算和 history[0]。
    neutralize: 拟合前清零表情类 morph(失败只警告,不阻塞)
    coarse_to_fine: 按 DEFAULT_STAGES 两阶段(轮廓→五官),预算按维数分摊
    stages: 自定义阶段(分组名列表的列表),给了则覆盖 coarse_to_fine
    """
    from .priors import order_by_prior, seed_from_diff

    target = load_image(target_image_path)
    if scorer is None:
        build = build_scorer_stack(style, target=target, anime_onnx=anime_onnx)
        scorer, eff_style, warning = build.scorer, build.style, build.warning
    else:
        eff_style, warning = getattr(scorer, "name", "custom"), None
    if isinstance(scorer, ArcFaceScorer):
        scorer.set_target(target)

    neutralized: List[str] = []
    if neutralize:
        try:
            neutralized = list(neutralize_expression(bridge, cfg.atom)["zeroed"])
        except Exception:
            log.warning("expression neutralization failed (ignored)", exc_info=True)

    evaluate = make_evaluator(bridge, target, scorer, cfg)

    # ---- 先验探针:一次基线评估 → 特征差 → 初始种子 ------------------------
    seed: Dict[str, float] = dict(cfg.seed or {})
    prior_seed: Dict[str, float] = {}
    history_all: List[float] = []
    budget = cfg.max_iters
    geo = find_geometry_scorer(scorer) if use_prior else None
    if geo is not None and budget > 1:
        base_vals = {n: seed.get(n, 0.0) for n in cfg.morph_names}
        history_all.append(evaluate(base_vals))
        budget -= 1
        if geo.last_diff:
            prior_seed = seed_from_diff(
                geo.last_diff, cfg.morph_names,
                bounds_fn=lambda n: _bounds_for(cfg, n))
            for k, v in prior_seed.items():
                seed.setdefault(k, v)  # 用户显式给的种子优先于先验

    # ---- 阶段划分与预算分摊 --------------------------------------------------
    if stages is None and coarse_to_fine:
        stages = DEFAULT_STAGES
    stage_names = (_stage_name_lists(cfg, stages) if stages
                   else [list(cfg.morph_names)])
    total_dims = sum(len(ns) for ns in stage_names) or 1

    best_morphs: Dict[str, float] = {}
    best_score = -1e9
    for i, names in enumerate(stage_names):
        stage_budget = max(8, int(round(budget * len(names) / total_dims))) \
            if len(stage_names) > 1 else budget
        sub = FitConfig(
            atom=cfg.atom,
            morph_names=order_by_prior(names, seed),
            bounds=cfg.bounds, default_bound=cfg.default_bound,
            max_iters=stage_budget, screenshot_width=cfg.screenshot_width,
            use_cache=cfg.use_cache,
            seed={n: seed.get(n, 0.0) for n in names},
            on_eval=cfg.on_eval)
        if optimizer == "greedy":
            res = greedy_coordinate(evaluate, sub)
        elif optimizer == "cma":
            res = cma_optimize(evaluate, sub)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")
        history_all.extend(res.history)
        best_morphs.update(res.best_morphs)
        best_score = res.best_score
        # 冻结本阶段最优:让后续阶段在它之上评估(set_morphs 是增量语义)
        bridge.set_morphs(cfg.atom, res.best_morphs, clamp=True)
        evaluate.bump_epoch()  # 绕过 evaluate 写了状态,旧缓存作废
        # 阶段间刷新先验:用最新残差修正后续阶段的种子与顺序
        if geo is not None and geo.last_diff and i + 1 < len(stage_names):
            for k, v in seed_from_diff(geo.last_diff, cfg.morph_names,
                                       bounds_fn=lambda n: _bounds_for(cfg, n)).items():
                seed.setdefault(k, v)

    # leave VaM showing the best result
    bridge.set_morphs(cfg.atom, best_morphs, clamp=True)
    result = FitResult(best_score=best_score, best_morphs=best_morphs,
                       history=history_all, warning=warning,
                       style=eff_style, scorer_name=scorer.name,
                       prior_seed=prior_seed, neutralized=neutralized,
                       stage_count=len(stage_names),
                       cache_hits=evaluate.cache_hits)
    # 让最优解成为"最近一次评估",这样 hints 描述的是最终结果的残差
    try:
        shot = bridge.screenshot(max_width=cfg.screenshot_width)
        scorer.score(target, decode_png_b64(shot["png_base64"]))
        result.hints = scorer.hints()
    except Exception:
        log.exception("final hints pass failed (ignored)")
    return result
