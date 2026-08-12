"""A curated set of high-leverage Genesis 2 face morphs.

These are the "big levers" for identity — the dimensions that move a face
the most per unit change. Optimizing over this ~40-morph subset instead of
the full ~1000+ installed morphs keeps the search tractable (CMA-ES scales
poorly past ~100 dims) while still covering the shape of the face.

NAMES ARE PROVISIONAL. They follow VaM/Genesis 2 built-in display-name
conventions, but installed morph packs vary between setups. At runtime the
MCP `list_morphs` tool should be used to reconcile these against what the
target VaM actually has; unknown names are reported back as `missing` by
set_morphs rather than causing a failure.

Grouped by region so a caller can optimize coarse-to-fine (e.g. skull +
jaw first, then eyes/nose/mouth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# region -> [morph display names]
FACE_MORPH_GROUPS: Dict[str, List[str]] = {
    "skull": [
        "Head Big", "Head Scale", "Face Flat", "Face Round",
        "Face Long", "Cranium Shape", "Brow Height",
    ],
    "jaw": [
        "Jaw Size", "Jaw Height", "Jaw Width", "Jaw Angle",
        "Chin Height", "Chin Width", "Chin Depth", "Chin Forward",
    ],
    "cheeks": [
        "Cheekbones Size", "Cheekbones Width", "Cheeks Depth",
        "Cheeks Sink",
    ],
    "eyes": [
        "Eyes Size", "Eyes Height", "Eyes Depth", "Eyes Width Spacing",
        "Eyes Slant", "Eyelids Height", "Eye Fold Depth",
    ],
    "nose": [
        "Nose Size", "Nose Width", "Nose Height", "Nose Bridge Width",
        "Nose Tip Height", "Nose Tip Width", "Nostrils Width",
        "Nose Bump",
    ],
    "mouth": [
        "Mouth Size", "Mouth Width", "Mouth Height",
        "Lips Thickness", "Upper Lip Thickness", "Lower Lip Thickness",
        "Lips Width", "Mouth Corners",
    ],
    "ears": [
        "Ears Size", "Ears Height",
    ],
}


def default_face_morph_names() -> List[str]:
    names: List[str] = []
    for group in FACE_MORPH_GROUPS.values():
        names.extend(group)
    return names


def default_bounds(name: str) -> Tuple[float, float]:
    """Most Genesis 2 head morphs behave well in [-1, 1]; a few want wider.

    Scale-type morphs are given a tighter range to avoid grotesque results
    during search.
    """
    lname = name.lower()
    if "scale" in lname or "size" in lname:
        return (-0.6, 0.8)
    return (-1.0, 1.0)


# ---------------------------------------------------------------------------
# 表情归一化(v0.4)
# ---------------------------------------------------------------------------
# 拟合比的是**身份**,不是表情。目标图咧嘴笑时,优化器很乐意用"Smile"类
# morph 去凑相似度 —— 这是作弊解:预设存下来的是一张永远在笑的脸。
# 所以拟合前把表情类 morph 清零。匹配走小写子串,宁可多匹配也别漏
# (身份 morph 命中这些词的概率极低,精选子集里一个都没有)。
EXPRESSION_PATTERNS = [
    "smile", "frown", "grin", "laugh", "cry", "pout", "kiss",
    "angry", "anger", "sad", "happy", "fear", "afraid", "disgust",
    "surprise", "shock", "concentrate", "flirt", "desire", "pain",
    "blink", "wink", "squint", "eyes closed", "eye closed",
    "brow up", "brow down", "brows up", "brows down",
    "mouth open", "open mouth", "jaw open", "tongue",
    "snarl", "sneer", "smirk", "scream", "yawn",
]


def is_expression_morph(name: str) -> bool:
    lname = name.lower()
    return any(p in lname for p in EXPRESSION_PATTERNS)


# ---------------------------------------------------------------------------
# 别名解析(v0.5.4)—— 概念槽位 → 目标 VaM 实际 morph 名
# ---------------------------------------------------------------------------
# 真机第二/三跑实锤:44 个精选名在用户 VaM 上缺 16 个(36% 搜索维度是死区)。
# 精选名单从"名字表"升级成"概念槽位表":每个槽位一串候选别名,运行时对
# list_morphs 返回的实际名单解析。匹配只做**归一化后的精确等值**
# (小写 + 去掉全部非字母数字),绝不做子串模糊 ——"Nose Size"命中
# "Nose Size Depth" 这种假阳性会把优化器带沟里,宁可解析不到进 missing。
#
# ALIAS 表是**数据**,机制不依赖它的完备性:候选名按优先级排,现在填的是
# Genesis 2 / 常见 morph 包的命名变体(仍是猜的,对话记忆第五节继续有效);
# 等用户导出的 morph 清单 JSON 到手,按实际装的包回填,一劳永逸。

def norm_name(name: str) -> str:
    """morph 名归一化:小写 + 剥掉所有非字母数字(空格/连字符/下划线/点)。"""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# 概念槽位(= FACE_MORPH_GROUPS 里的规范名)→ 候选别名(按优先级)
# 规范名自身永远是第一候选,不用重复写进表里。
#
# 2026-08-12 已按仓库主人导出的真机清单(1190 个 morph)校准:
# 标 [真机] 的候选是在他机器上实际命中的名字;前面的泛型候选留给其他
# morph 包。标 [近义] 的是语义近似兜底(找不到同义杠杆时用最接近的),
# 语义有漂移的在行内注明。"Lips Thickness" 在他机器上**故意留死**:
# 上/下唇厚各有独立杠杆,总厚度维度纯冗余,不值得占一维。
MORPH_ALIASES: Dict[str, List[str]] = {
    "Head Big": ["Head Size", "Head Large", "Head Big Small", "Head Scale Big",
                 # [近义] 语义从"整体大"漂到"头宽"——Head Scale 概念已占走
                 # 真正的整体缩放,宽度是此外唯一没被覆盖的颅型杠杆
                 "Head Width"],
    "Cranium Shape": ["Cranium Size",  # [真机]
                      "Skull Shape", "Head Shape", "Cranium"],
    "Face Long": ["Face Length",
                  "Face Height 2",  # [真机] 双向(-0.48..1),优先于单向的 Face Height
                  "Face Height",    # [真机] 0..1 只能拉长不能缩短
                  "Face Elongate", "Face Tall", "Face Long Short"],
    "Jaw Width": ["Jaws Width", "Jaw Wide", "Jaw Width Wide",
                  "Jaw Corner Width"],  # [真机] 0..1,颌角间宽,最贴切的宽度杠杆
    "Chin Forward": ["Chin Front", "Chin Forward Back", "Chin Protrude",
                     "Chin Depth Forward",
                     "Chin Out"],  # [真机] 命中 "ChinOut"(归一化吃掉空格差异)
    "Cheekbones Size": ["Cheek Bones Size",  # [真机]
                        "Cheekbone Size", "CheekBones Size", "Cheeks Bone Size"],
    "Cheekbones Width": ["Cheek Bones Width",  # [真机]
                         "Cheekbone Width", "CheekBones Width"],
    "Eyes Width Spacing": ["Eyes Spacing Width", "Eye Spacing", "Eyes Spacing",
                           "Eyes Distance", "Eye Distance", "Eyes Width Apart",
                           # [真机/近义] 眼鼻整体加宽,eye_gap 的最强可用杠杆
                           "Eyes Nose Width",
                           # [真机/近义] 只动内眼角,间距效果较弱
                           "Eyes Inner Corner Width"],
    "Eyes Slant": ["Eyes Angle",  # [真机]
                   "Eye Slant", "Eyes Slant Inner", "Eyes Tilt"],
    "Eyelids Height": ["Eyelid Height", "Eyes Lid Height", "Eyelids Top Height",
                       "Upper Eyelids Height",
                       # [真机/近义] ↑=眼皮更重=眼更闭,与本概念方向一致
                       "Eyelids Heavy",
                       "Eyelids Height Inner"],  # [真机] 只动内侧,兜底
    "Eye Fold Depth": ["Eye Fold",  # [真机] 0..1
                       "Eyes Fold Depth", "Eyelid Fold Depth", "Eye Folds",
                       "Eyelids Fold"],
    "Lips Thickness": ["Lip Thickness", "Lips Thick", "Lips Thin Thick",
                       "Lips Full", "Lips Fullness"],
    "Lips Width": ["Lip Width", "Lips Wide",
                   "Mouth Corner Width",  # [真机/近义] 嘴角间宽 ≈ 唇宽
                   "Lips Top Width"],     # [真机] 命中 "LIps Top Width"(大小写归一化)
    "Upper Lip Thickness": ["Lip Upper Thickness",
                            "Lip Upper Thick",  # [真机] 注意词序:归一化不吞词序
                            "Upper Lip Thick", "Lip Top Thickness",
                            "Lips Upper Thickness",
                            "Lips Top Full"],   # [真机] 兜底
    "Lower Lip Thickness": ["Lip Lower Thickness",
                            "Lips Bottom Full",  # [真机] -1..1 双向
                            "Lower Lip Thick", "Lip Bottom Thickness",
                            "Lips Lower Thickness",
                            "Lip Lower Size"],   # [真机] 兜底
    "Mouth Corners": ["Mouth Corner Height",  # [真机]
                      "Mouth Corners Up Down", "Lip Corners", "Mouth Corner Up-Down"],
}


@dataclass
class MorphResolution:
    """resolve_names 的产物:概念名 → 实际名的映射 + 解析不到的清单。"""

    mapping: Dict[str, str] = field(default_factory=dict)   # 概念名 → 实际名(全部已解析)
    renamed: Dict[str, str] = field(default_factory=dict)   # 仅"实际名 ≠ 概念名"的子集
    unresolved: List[str] = field(default_factory=list)     # 一个候选都没命中的概念名

    def actual(self, name: str) -> str:
        """概念名 → 实际名;解析不到时原样返回(调用方自行决定丢弃或硬试)。"""
        return self.mapping.get(name, name)

    def to_actual(self, names: List[str]) -> List[str]:
        """批量翻译,**丢弃**解析不到的名字(它们是死维度)。"""
        drop = set(self.unresolved)
        return [self.mapping.get(n, n) for n in names if n not in drop]


def resolve_names(wanted: List[str], available: List[str]) -> MorphResolution:
    """把概念名单解析到目标 VaM 的实际 morph 名单上。

    匹配顺序:规范名归一化等值 → 各候选别名归一化等值。
    同一个实际 morph 只能被一个槽位占用(先到先得),防止两个概念
    抢同一根滑块导致优化互相打架。
    """
    by_norm: Dict[str, str] = {}
    for a in available:
        by_norm.setdefault(norm_name(a), a)  # 重名首见为准,确定性

    res = MorphResolution()
    taken: set = set()
    for name in wanted:
        actual = None
        for cand in [name] + MORPH_ALIASES.get(name, []):
            hit = by_norm.get(norm_name(cand))
            if hit is not None and hit not in taken:
                actual = hit
                break
        if actual is None:
            res.unresolved.append(name)
            continue
        taken.add(actual)
        res.mapping[name] = actual
        if actual != name:
            res.renamed[name] = actual
    return res


# 概念被其他槽位组合覆盖时,解析不到不算真缺失(不进 missing 提示)
COVERED_BY: Dict[str, Tuple[str, ...]] = {
    "Lips Thickness": ("Upper Lip Thickness", "Lower Lip Thickness"),
}
