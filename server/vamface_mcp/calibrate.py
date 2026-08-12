"""本地滑块校准模型(v0.7)—— 从"瞎蒙"到"解方程"。

用户的原话点破了问题:"做一个专门为这台 VaM 的模型,知道怎么调是什么样,
现在就是乱弄"。他说得对:CMA-ES 在 43 维里黑盒乱撞,每一步都不知道
滑块和脸的关系。但这个关系**可以直接量出来**:

  校准(一次性,~44 次评估,结果落盘复用):
    逐个 morph 拨一下 → 截图 → 量几何特征动了多少 → 斜率表 J
    J[morph][feature] = ∂feature/∂morph,就是"这台 VaM 上这根滑块干什么"

  拟合(每步 1 次评估):
    残差 r = 目标特征 - 当前特征(GeometryScorer.last_diff 白送)
    解岭回归  (JᵀWJ + λI) Δ = JᵀW r  → 一步该拨哪些滑块、拨多少
    走几步 Gauss-Newton,几何维度直接收敛;ArcFace 管不到的细节
    再交给 CMA 收尾。

对比先验(priors.py):那张表是在 mock 上手标的十来条斜率;这里是在
**用户真机、当前基底之上**逐滑块实测的全量斜率 —— 这就是他要的
"专门为这个 VaM 的本地模型"。

缓存:~/.vamface/jacobian_<key>.json,key = morph 名单 + 截图宽 + 基底。
换 morph 包 / 换基底会自然换 key 重测;同配置复跑白捡 ~44 次评估。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("vamface.calibrate")

CACHE_DIR = Path.home() / ".vamface"

# 特征权重沿用 GeometryScorer.WEIGHTS(解方程时重要的特征话语权大)
from .scorers import GeometryScorer  # noqa: E402

Jacobian = Dict[str, Dict[str, float]]  # morph -> {feature: slope}


def _cache_key(morph_names: List[str], width: int, basis_tag: str) -> str:
    blob = "|".join(sorted(morph_names)) + f"@{width}#{basis_tag}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"jacobian_{key}.json"


def load_jacobian(morph_names: List[str], width: int,
                  basis_tag: str = "") -> Optional[Jacobian]:
    p = _cache_path(_cache_key(morph_names, width, basis_tag))
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return {str(k): {str(f): float(s) for f, s in v.items()}
                    for k, v in data.get("jacobian", {}).items()}
    except Exception:
        log.warning("校准缓存读取失败,忽略并重测", exc_info=True)
    return None


def save_jacobian(J: Jacobian, morph_names: List[str], width: int,
                  basis_tag: str = "") -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _cache_path(_cache_key(morph_names, width, basis_tag))
        p.write_text(json.dumps({"created": time.time(), "width": width,
                                 "basis": basis_tag, "jacobian": J},
                                ensure_ascii=False, indent=1),
                     encoding="utf-8")
    except Exception:
        log.warning("校准缓存写入失败(不阻塞)", exc_info=True)


def _probe_delta(name: str, bounds_fn: Callable[[str], Tuple[float, float]],
                 base_v: float, want: float = 0.5) -> float:
    """选探测步长:优先 +want,越界就试负向;两头都没空间返回 0(跳过)。"""
    lo, hi = bounds_fn(name)
    if base_v + want <= hi:
        return want
    if base_v - want >= lo:
        return -want
    up, down = hi - base_v, base_v - lo
    if max(up, down) < 0.05:
        return 0.0
    return up if up >= down else -down


def probe_jacobian(evaluate, geo, base_vals: Dict[str, float],
                   morph_names: List[str],
                   bounds_fn: Callable[[str], Tuple[float, float]],
                   delta: float = 0.5, budget: int = 10 ** 9,
                   ) -> Tuple[Jacobian, int, List[float]]:
    """逐 morph 实测斜率。走 evaluate(计预算/进曲线/吃缓存),特征从
    geo.last_diff 反推:diff = 目标 - 当前,所以斜率 = (diff0 - diff_i)/d。

    返回 (J, 用掉的评估数, 分数序列)。结束时场景回到 base_vals 状态。
    """
    history: List[float] = []
    used = 0

    diff0: Dict[str, float] = {}
    for attempt in range(3):  # 检测会帧间抖动,基线多拍两张再放弃
        s0 = evaluate(dict(base_vals))
        history.append(s0)
        used += 1
        diff0 = dict(geo.last_diff)
        if diff0:
            break
        evaluate.bump_epoch()  # 相同向量会命中缓存,bump 强制重拍
    if not diff0:
        log.warning("校准基线连拍 3 次都检不出脸,放弃(灯光/机位问题)")
        return {}, used, history

    J: Jacobian = {}
    for name in morph_names:
        if used >= budget:
            break
        d = _probe_delta(name, bounds_fn, base_vals.get(name, 0.0), delta)
        if d == 0.0:
            continue
        vals = dict(base_vals)
        vals[name] = base_vals.get(name, 0.0) + d
        s = evaluate(vals)
        history.append(s)
        used += 1
        diff_i = dict(geo.last_diff)
        if not diff_i:
            continue  # 这一拨把脸拨没了(检测失败),该滑块不入模型
        slopes = {}
        for k in set(diff0) & set(diff_i):
            slope = (diff0[k] - diff_i[k]) / d
            if abs(slope) > 1e-4:
                slopes[k] = round(float(slope), 6)
        if slopes:
            J[name] = slopes

    # 回到基线(绕过 evaluate 的写要作废缓存 —— 老规矩)
    evaluate_bridge_restore(evaluate, base_vals)
    return J, used, history


def evaluate_bridge_restore(evaluate, base_vals: Dict[str, float]) -> None:
    """探针结束把场景恢复到 base_vals。Evaluator 持有 bridge/cfg,借用之。"""
    try:
        evaluate._bridge.set_morphs(evaluate._cfg.atom, dict(base_vals),
                                    clamp=True)
        evaluate.bump_epoch()
    except Exception:
        log.warning("校准后状态恢复失败(下一次评估会覆盖,继续)",
                    exc_info=True)


def solve_step(J: Jacobian, residual: Dict[str, float],
               morph_names: List[str],
               lam: float = 0.1, step_clip: float = 0.5
               ) -> Dict[str, float]:
    """一步 Gauss-Newton:解 (JᵀWJ + λ·s·I)Δ = JᵀW r。

    λ 是**相对**岭正则(s = trace(JᵀWJ)/n 做尺度归一 —— 斜率量级只有
    ~0.05,固定 λ 会把解压扁十倍,实测踩过);9 个特征 vs 40+ 滑块严重
    欠定,正则不能省。Δ 逐坐标夹在 ±step_clip(线性模型只在局部可信)。
    """
    names = [n for n in morph_names if n in J]
    feats = sorted(set(residual) & {f for m in names for f in J[m]})
    if not names or not feats:
        return {}
    W = np.array([GeometryScorer.WEIGHTS.get(f, 1.0) for f in feats])
    A = np.array([[J[n].get(f, 0.0) for f in feats] for n in names]).T
    r = np.array([residual[f] for f in feats])
    Aw = A * W[:, None]
    n = len(names)
    AtA = Aw.T @ A
    scale = float(np.trace(AtA)) / n
    if scale <= 0:
        return {}
    try:
        delta = np.linalg.solve(AtA + lam * scale * np.eye(n), Aw.T @ r)
    except np.linalg.LinAlgError:
        return {}
    delta = np.clip(delta, -step_clip, step_clip)
    return {names[i]: float(delta[i]) for i in range(n)
            if abs(delta[i]) > 1e-4}


def jacobian_polish(evaluate, geo, J: Jacobian, x0: Dict[str, float],
                    morph_names: List[str],
                    bounds_fn: Callable[[str], Tuple[float, float]],
                    iters: int = 5, budget: int = 10 ** 9,
                    ) -> Tuple[Dict[str, float], float, int, List[float]]:
    """从 x0 出发走 Gauss-Newton:每步 = 拿残差 → 解方程 → 拨滑块 → 评估。

    返回 (最优向量, 最优分, 用掉的评估数, 分数序列)。
    """
    history: List[float] = []
    used = 0
    x = {n: x0.get(n, 0.0) for n in morph_names}

    s = evaluate(dict(x))
    history.append(s)
    used += 1
    best_x, best_s = dict(x), s

    for _ in range(iters):
        if used >= budget or not geo.last_diff:
            break
        delta = solve_step(J, geo.last_diff, morph_names)
        if not delta:
            break
        moved = False
        for n, dv in delta.items():
            lo, hi = bounds_fn(n)
            nv = float(np.clip(x.get(n, 0.0) + dv, lo, hi))
            if abs(nv - x.get(n, 0.0)) > 1e-4:
                x[n] = nv
                moved = True
        if not moved:
            break
        s = evaluate(dict(x))
        history.append(s)
        used += 1
        if s > best_s:
            best_x, best_s = dict(x), s

    return best_x, best_s, used, history
