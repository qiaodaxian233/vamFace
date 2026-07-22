"""先验:把几何特征差反哺给优化器 —— hints 从"给人看"升级成"给优化器用"。

v0.3 的 GeometryScorer 已经能算出"目标 vs 当前渲染"的特征差(eye_gap、
mouth_w、face_aspect……),并映射成人类可读的方向提示。这些特征和 morph
几乎一一对应,所以同一张映射表可以直接产出:

  1. **初始种子** seed_from_diff():CMA 的 x0 / greedy 的起点,不再从全零
     开始摸;
  2. **坐标顺序** order_by_prior():greedy 先动"差得最多"的维度,前几步
     就是最有效的几步。

真机上每次评估是一整个"set_morphs → 截图 → 打分"往返,种子省下的评估
次数就是省下的真实挂机时间。价值在 mock 上可定量验证(见
tests/test_priors.py:同一隐藏目标,带/不带先验对比达阈值所需评估数)。

设计注意:
  - 增益是**粗略经验值**,目的只是把起点丢进正确的象限,精修交给优化器;
    所以每个种子都夹在 PRIOR_CLIP 内,宁可保守不可过冲。
  - 一个特征摊到多个 morph 时增益按份拆,避免同向叠加过冲。
  - 表里的 morph 名沿用精选子集(morph_presets),不在 cfg.morph_names
    里的名字不会出现在种子里 —— 先验永远不会引入优化范围之外的维度。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

# 特征差(target - candidate,正 = 目标该维更大)→ [(morph 名, 增益)]
# 增益符号约定:seed[morph] = clip(gain * delta)。
# 与 scorers._HINT_MAP 的方向一致 —— 两张表必须同步改(grep 提醒)。
#
# 增益幅度**按 mock 实测斜率校准**(morph +1.0 → 特征动多少,取 1/slope 的
# 一半:一步走一半路,防过冲):eye_gap 1/slope≈16,mouth_w≈20,eye_w≈37,
# eye_y≈-31,mouth_y≈-37,face_aspect≈3.9。真机的斜率同数量级(morph=1 改
# 变眼距约百分之几的脸宽),且有 PRIOR_CLIP 兜底 —— 先验只负责进对象限。
FEATURE_TO_MORPHS: Dict[str, List[Tuple[str, float]]] = {
    "eye_gap": [("Eyes Width Spacing", 8.0)],
    "eye_w": [("Eyes Size", 10.0)],
    "eye_h": [("Eyes Size", 6.0), ("Eyelids Height", -6.0)],
    "eye_y": [("Eyes Height", -15.0)],         # 特征是"眼在脸框内的相对深度",
                                                # 目标更靠下(Δ>0)→ Eyes Height ↓
    "nose_len": [("Nose Size", 5.0), ("Nose Height", 5.0)],
    "mouth_w": [("Mouth Width", 10.0), ("Lips Width", 4.0)],
    "mouth_y": [("Mouth Height", -18.0)],
    "jaw_len": [("Chin Height", 4.0), ("Jaw Size", 3.0)],
    "face_aspect": [("Face Long", 2.0), ("Face Round", -1.5)],
}

# 种子的绝对上限:先验只负责指方向,不负责一步到位
PRIOR_CLIP = 0.6


def seed_from_diff(diff: Dict[str, float],
                   allowed_names: Sequence[str],
                   bounds_fn: Callable[[str], Tuple[float, float]] | None = None,
                   clip: float = PRIOR_CLIP) -> Dict[str, float]:
    """特征差 → 初始 morph 种子。只产出 allowed_names 里的名字。

    diff: GeometryScorer.last_diff(正 = 目标该特征更大)
    bounds_fn: 每个 morph 的 (lo, hi),种子最终还会夹到该范围内
    """
    allowed = set(allowed_names)
    seed: Dict[str, float] = {}
    for feat, delta in diff.items():
        for name, gain in FEATURE_TO_MORPHS.get(feat, []):
            if name not in allowed:
                continue
            v = seed.get(name, 0.0) + gain * float(delta)
            seed[name] = v
    out: Dict[str, float] = {}
    for name, v in seed.items():
        v = max(-clip, min(clip, v))
        if bounds_fn is not None:
            lo, hi = bounds_fn(name)
            v = max(lo, min(hi, v))
        if abs(v) > 1e-4:  # 幅度太小的种子没有信息量,不如留 0
            out[name] = round(v, 4)
    return out


def order_by_prior(names: Sequence[str], seed: Dict[str, float]) -> List[str]:
    """greedy 的坐标顺序:|种子| 大的先动(差得多的维度最先修),其余保持原序。

    sorted 是稳定排序,种子里没有的名字相对顺序不变、排在后面。
    """
    return sorted(names, key=lambda n: -abs(seed.get(n, 0.0)))
