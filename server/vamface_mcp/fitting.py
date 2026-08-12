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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    missing: List[str] = field(default_factory=list)  # 目标 VaM 缺的精选 morph 名
    renamed: Dict[str, str] = field(default_factory=dict)  # 别名解析:概念名→实际名
    basis: Dict[str, float] = field(default_factory=dict)  # 角色基底 {整头morph: 权重}
    basis_missing: List[str] = field(default_factory=list)  # 列表里有但 set 被拒的基底候选
    jacobian_note: str = ""              # 本地校准模型状态(新测/缓存/失败)


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
        # 目标 VaM 里不存在的 morph 名(set_morphs 回执的 missing 并集)。
        # 这是校准 morph_presets 的原料 —— 精选名单是按 Genesis 2 惯例猜的,
        # 必须跟用户实际装的 morph 包对账(对话记忆 第五节)。
        self.missing: set = set()

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
        reply = self._bridge.set_morphs(cfg.atom, morphs, clamp=True)
        miss = reply.get("missing") if isinstance(reply, dict) else None
        if miss:
            self.missing.update(str(m) for m in miss)
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


# ---------------------------------------------------------------------------
# 角色基底粗定位(v0.6)
# ---------------------------------------------------------------------------
# 精选 morph 是**增量**滑块:从默认脸出发,几百次评估爬不到脸型差异大的
# 目标(实测:默认 G2 脸 → 东亚脸,0.2x 分封顶)。但用户装的 morph 包里
# 通常有一堆"整头角色" morph(Aiko 6 Head、Sumiko Head……),每个都是
# 一张完整的脸 —— 先扫一遍找最像的当起点,再用精选 morph 精修,等于
# 白捡一个 route A 的低配版:基底负责跨过大距离,增量滑块负责收尾。

# 名字里带这些词的 Head 区 morph 是特征滑块,不是整头角色
_CHARACTER_HEAD_EXCLUDE = (
    "scale", "width", "length", "height", "size", "define", "round",
    "flat", "slope", "wrinkle", "puffy", "neck", "smooth", "shape",
)


def character_head_candidates(rows: List[Dict[str, Any]]) -> List[str]:
    """从 list_morphs 行里挑'整头角色' morph:region==Head 且名字无特征词。

    启发式(按仓库主人真机清单调过):Head 区 51 个里能留下 42 个角色头,
    误杀为零。别的机器上宁可漏挑(候选少)也别错挑(把 Head Scale 当
    角色头扫会毁掉基线状态)。
    """
    out: List[str] = []
    for r in rows:
        if str(r.get("region", "")).strip().lower() != "head":
            continue
        name = str(r.get("name", "")).strip()
        low = name.lower()
        if not name or any(w in low for w in _CHARACTER_HEAD_EXCLUDE):
            continue
        out.append(name)
    return out


def basis_search(evaluate, bridge, atom: str, candidates: List[str],
                 baseline: float, weights: Tuple[float, ...] = (1.0, 0.6),
                 budget: int = 60, topk: int = 3
                 ) -> Tuple[Dict[str, float], int, List[float], List[str]]:
    """扫描候选整头 morph,找最像目标的当拟合起点。

    每个候选打 weights[0] 评估一次(顺手清掉上一个候选);**前 topk 名**
    再各试其余权重 —— 截图有噪声,单次评估的 argmax 容易选错人,给前几名
    复赛机会。谁都赢不过 baseline(不加基底的脸)就全部归零、空手而归。

    真机怪相防御:个别 morph 列表里有但 set 被拒(回执 missing)——那次
    评估拍的是没变化的脸,分数是 baseline 的伪装,候选当场作废,进
    invalid 名单带回去(修插件的线索)。

    返回 (basis {名字: 权重} 或 {}, 用掉的评估数, 分数序列, invalid 名单)。
    结束时场景状态 = 采纳的结果(其余候选已归零),缓存 epoch 已 bump。
    """
    history: List[float] = []
    used = 0
    prev: Optional[str] = None
    touched: set = set()
    scores: Dict[str, float] = {}
    invalid: List[str] = []
    seen_missing = set(getattr(evaluate, "missing", set()))

    for c in candidates:
        if used >= budget:
            break
        vals: Dict[str, float] = {c: weights[0]}
        if prev is not None:
            vals[prev] = 0.0
        s = evaluate(vals)
        used += 1
        history.append(s)
        touched.add(c)
        now_missing = set(getattr(evaluate, "missing", set()))
        if c in now_missing - seen_missing:
            invalid.append(c)
            seen_missing = now_missing
        else:
            scores[c] = s
        prev = c

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:max(1, topk)]
    best_name: Optional[str] = None
    best_w = 0.0
    best_score = baseline
    for name, s in ranked:
        if s > best_score:
            best_name, best_w, best_score = name, weights[0], s

    for w in weights[1:]:
        for name, _ in ranked:
            if used >= budget:
                break
            vals = {name: w}
            if prev is not None and prev != name:
                vals[prev] = 0.0
            s = evaluate(vals)
            used += 1
            history.append(s)
            if s > best_score:
                best_name, best_w, best_score = name, w, s
            prev = name

    # 落定:所有摸过的候选归零,采纳的基底(若有)设回最优权重
    settle = {c: 0.0 for c in touched}
    basis: Dict[str, float] = {}
    if best_name is not None:
        basis = {best_name: best_w}
        settle[best_name] = best_w
    if settle:
        bridge.set_morphs(atom, settle, clamp=True)
        evaluate.bump_epoch()  # 绕过 evaluate 写了状态,旧缓存作废
    return basis, used, history, invalid


def _stage_name_lists(cfg: FitConfig, stages: List[List[str]],
                      rename: Optional[Dict[str, str]] = None) -> List[List[str]]:
    """分组名 → morph 名列表,并与 cfg.morph_names 求交(尊重用户的 groups 选择)。

    cfg.morph_names 里不属于任何所选分组的名字(用户自定义 morph)归入末阶段。
    rename: 概念名 → 实际名(别名解析产物)。分组表存的是概念名,
    cfg.morph_names 解析后是实际名,交集前先翻译。
    """
    from .morph_presets import FACE_MORPH_GROUPS

    ren = rename or {}
    allowed = set(cfg.morph_names)
    out: List[List[str]] = []
    used: set = set()
    for groups in stages:
        names = []
        for g in groups:
            for n in FACE_MORPH_GROUPS.get(g, []):
                a = ren.get(n, n)
                if a in allowed and a not in used:
                    names.append(a)
                    used.add(a)
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
             stages: Optional[List[List[str]]] = None,
             use_basis: bool = False,
             basis_candidates: Optional[List[str]] = None,
             use_jacobian: bool = False) -> FitResult:
    """v0.4 拟合入口:表情归一化 → 先验探针 → (分阶段)黑盒优化 → 提示。

    style: auto | real | anime | pixel(见 scorers.build_scorer_stack)
    scorer: 直接注入打分器(测试/自定义用);给了就跳过 build_scorer_stack
    use_prior: 打分器里有 GeometryScorer 时,先做一次基线评估拿特征差,
               换算成初始种子(CMA x0 / greedy 起点+坐标顺序)。探针那次
               评估计入预算和 history[0]。
    neutralize: 拟合前清零表情类 morph(失败只警告,不阻塞)
    coarse_to_fine: 按 DEFAULT_STAGES 两阶段(轮廓→五官),预算按维数分摊
    stages: 自定义阶段(分组名列表的列表),给了则覆盖 coarse_to_fine
    use_basis: v0.6 角色基底粗定位 —— 先扫整头角色 morph 找最像的当起点,
               再精修(候选默认从 list_morphs 的 Head 区自动挑,也可用
               basis_candidates 显式给)。每个候选吃一次评估,计入预算。
    use_jacobian: v0.7 本地校准模型 —— 逐 morph 实测"这根滑块动脸多少"
               (首跑 ~1 评估/滑块,结果落盘,同配置复跑免费),然后按
               残差解方程走 Gauss-Newton,几何维度直接收敛,CMA 只收尾。
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

    # ---- 别名解析(v0.5.4):概念名 → 目标 VaM 实际 morph 名 -----------------
    # 解析不到的概念是死维度,进搜索只烧预算不产出 —— 从优化范围里剔除,
    # 记进 missing。list_morphs 失败或名单被 limit 截断时跳过解析,按原名
    # 硬试(missing 仍由 set_morphs 回执兜底收集)—— 猜测值不当阻塞器。
    canonical_names = list(cfg.morph_names)
    resolution = None
    rows: List[Dict[str, Any]] = []
    real_bounds: Dict[str, Tuple[float, float]] = {}
    try:
        reply = bridge.list_morphs(cfg.atom, limit=1_000_000)
        rows = reply.get("morphs") or []
        available = [str(r.get("name")) for r in rows if r.get("name")]
        total = int(reply.get("total", len(available)))
        if available and len(available) >= total:
            from .morph_presets import resolve_names
            resolution = resolve_names(canonical_names, available)
            for r in rows:
                try:
                    real_bounds[str(r["name"])] = (float(r["min"]), float(r["max"]))
                except (KeyError, TypeError, ValueError):
                    pass  # 行缺 min/max 就不收口这一个,不因脏数据放弃全部
    except Exception:
        log.warning("别名解析跳过(list_morphs 不可用),按原名硬试", exc_info=True)

    to_act = resolution.actual if resolution is not None else (lambda n: n)
    unresolved: List[str] = list(resolution.unresolved) if resolution else []
    renamed: Dict[str, str] = dict(resolution.renamed) if resolution else {}
    if resolution is not None and (unresolved or renamed):
        from dataclasses import replace as _dc_replace
        cfg = _dc_replace(
            cfg,
            morph_names=resolution.to_actual(canonical_names),
            bounds={to_act(k): v for k, v in cfg.bounds.items()},
            seed=({to_act(k): v for k, v in cfg.seed.items()}
                  if cfg.seed else cfg.seed))
    # 搜索边界 ∩ 真机 morph 的 min/max:超出插件会 clamp 的区域是平坦地形,
    # 优化器在里面走一步就是白烧一次评估。交集为空(配置错)时用真机范围。
    if real_bounds:
        from dataclasses import replace as _dc_replace
        tightened = dict(cfg.bounds)
        for n in cfg.morph_names:
            rb = real_bounds.get(n)
            if rb is None:
                continue
            lo, hi = _bounds_for(cfg, n)
            tlo, thi = max(lo, rb[0]), min(hi, rb[1])
            if tlo >= thi:
                tlo, thi = rb
            tightened[n] = (tlo, thi)
        cfg = _dc_replace(cfg, bounds=tightened)
    # hints 按实际可用性过滤/改名(修真机第三跑的 bug:提示推荐不存在的滑块)
    if resolution is not None:
        geo_h = find_geometry_scorer(scorer)
        if geo_h is not None:
            geo_h.set_morph_availability(available, rename=renamed)

    neutralized: List[str] = []
    if neutralize:
        try:
            neutralized = list(neutralize_expression(bridge, cfg.atom)["zeroed"])
        except Exception:
            log.warning("expression neutralization failed (ignored)", exc_info=True)

    evaluate = make_evaluator(bridge, target, scorer, cfg)
    evaluate.missing.update(unresolved)  # 解析阶段就确认缺失的,不用等回执

    # ---- 先验探针:一次基线评估 → 特征差 → 初始种子 ------------------------
    # FEATURE_TO_MORPHS 用概念名,产出的种子键要过 to_act 翻译成实际名。
    allowed_actual = set(cfg.morph_names)

    def _prior(diff: Dict[str, float]) -> Dict[str, float]:
        raw = seed_from_diff(diff, canonical_names,
                             bounds_fn=lambda n: _bounds_for(cfg, to_act(n)))
        return {to_act(k): v for k, v in raw.items()
                if to_act(k) in allowed_actual}

    seed: Dict[str, float] = dict(cfg.seed or {})
    prior_seed: Dict[str, float] = {}
    history_all: List[float] = []
    budget = cfg.max_iters
    geo = find_geometry_scorer(scorer) if use_prior else None

    # ---- 角色基底粗定位(v0.6):先跨大距离,再精修 --------------------------
    # 采纳基底后,下面的先验探针会在基底之上重测残差,种子据此重算。
    basis: Dict[str, float] = {}
    basis_missing: List[str] = []
    if use_basis and budget > 3:
        cand = (list(basis_candidates) if basis_candidates is not None
                else character_head_candidates(rows))
        if cand:
            base_vals = {n: seed.get(n, 0.0) for n in cfg.morph_names}
            baseline = evaluate(base_vals)
            history_all.append(baseline)
            budget -= 1
            bb = min(len(cand) + 3, max(0, budget - 8))  # 至少给精修留 8 次
            basis, used, bh, basis_missing = basis_search(
                evaluate, bridge, cfg.atom, cand, baseline, budget=bb)
            history_all.extend(bh)
            budget -= used
            if basis:
                log.info("basis 采纳: %s(扫描 %d 个候选,用 %d 次评估)",
                         basis, len(cand), used)
        else:
            log.info("basis 扫描跳过:没找到整头角色 morph 候选")

    if geo is not None and budget > 1:
        base_vals = {n: seed.get(n, 0.0) for n in cfg.morph_names}
        history_all.append(evaluate(base_vals))
        budget -= 1
        if geo.last_diff:
            prior_seed = _prior(geo.last_diff)
            for k, v in prior_seed.items():
                seed.setdefault(k, v)  # 用户显式给的种子优先于先验

    # ---- 本地校准模型(v0.7):量斜率 → 解方程,不瞎猜 -----------------------
    jacobian_note = ""
    pre_best: Optional[Tuple[Dict[str, float], float]] = None
    if use_jacobian and geo is not None and budget > 6:
        from .calibrate import (jacobian_polish, load_jacobian,
                                probe_jacobian, save_jacobian)
        basis_tag = ",".join(f"{k}={v:g}" for k, v in sorted(basis.items()))
        J = load_jacobian(cfg.morph_names, cfg.screenshot_width, basis_tag)
        if J is not None:
            jacobian_note = f"缓存复用({len(J)} 个滑块已测)"
        else:
            base_vals = {n: seed.get(n, 0.0) for n in cfg.morph_names}
            probe_cap = max(0, budget - 10)  # 至少给后面留 10 次
            J, used, ph = probe_jacobian(evaluate, geo, base_vals,
                                         cfg.morph_names,
                                         lambda n: _bounds_for(cfg, n),
                                         budget=probe_cap)
            history_all.extend(ph)
            budget -= used
            if J:
                save_jacobian(J, cfg.morph_names, cfg.screenshot_width,
                              basis_tag)
                jacobian_note = f"新测 {len(J)} 个滑块(已落盘,下次免费)"
            else:
                jacobian_note = "校准失败(基线检不出脸),退回黑盒"
        if J and budget > 2:
            x0 = {n: seed.get(n, 0.0) for n in cfg.morph_names}
            bx, bs, used, ph = jacobian_polish(
                evaluate, geo, J, x0, cfg.morph_names,
                lambda n: _bounds_for(cfg, n),
                iters=6, budget=max(2, budget // 3))
            history_all.extend(ph)
            budget -= used
            seed.update(bx)          # CMA 从解出来的点起步
            pre_best = (bx, bs)      # CMA 收尾没超过它就用它

    # ---- 阶段划分与预算分摊 --------------------------------------------------
    if stages is None and coarse_to_fine:
        stages = DEFAULT_STAGES
    stage_names = (_stage_name_lists(cfg, stages, rename=renamed) if stages
                   else [list(cfg.morph_names)])
    total_dims = sum(len(ns) for ns in stage_names) or 1

    best_morphs: Dict[str, float] = dict(basis)  # 基底进 .vap,不然预设丢脸型
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
            for k, v in _prior(geo.last_diff).items():
                seed.setdefault(k, v)

    if pre_best is not None and pre_best[1] > best_score:
        best_morphs = dict(basis)
        best_morphs.update(pre_best[0])
        best_score = pre_best[1]

    # leave VaM showing the best result
    bridge.set_morphs(cfg.atom, best_morphs, clamp=True)
    result = FitResult(best_score=best_score, best_morphs=best_morphs,
                       history=history_all, warning=warning,
                       style=eff_style, scorer_name=scorer.name,
                       prior_seed=prior_seed, neutralized=neutralized,
                       stage_count=len(stage_names),
                       cache_hits=evaluate.cache_hits,
                       missing=sorted(set(evaluate.missing) - set(basis_missing)),
                       renamed=renamed, basis=basis,
                       basis_missing=sorted(basis_missing),
                       jacobian_note=jacobian_note)
    # 让最优解成为"最近一次评估",这样 hints 描述的是最终结果的残差
    try:
        shot = bridge.screenshot(max_width=cfg.screenshot_width)
        scorer.score(target, decode_png_b64(shot["png_base64"]))
        result.hints = scorer.hints()
    except Exception:
        log.exception("final hints pass failed (ignored)")
    return result
