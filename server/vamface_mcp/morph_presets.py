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
