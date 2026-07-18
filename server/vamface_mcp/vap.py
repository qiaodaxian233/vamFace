"""Read/write VaM appearance presets (.vap) — the offline layer.

A .vap appearance preset is plain JSON. The part we care about for face
work is the "geometry" storable, which carries a list of morph entries:

    {
      "setUnlistedParamsToDefault": "true",
      "storables": [
        {
          "id": "geometry",
          "morphs": [
            {"name": "Nose Width", "value": "0.35"},
            ...
          ]
        },
        ...
      ]
    }

Notes:
- VaM serializes numbers as strings; we accept both on read and write
  strings for maximum compatibility.
- read_vap() is deliberately tolerant: it scans all storables and picks
  any "morphs" arrays it finds under a "geometry" id (some presets carry
  extra fields such as "uid" per morph — preserved on request).
- write_vap() by default produces a minimal, morphs-only preset. That is
  enough for face geometry; skin/hair/clothing presets are out of scope
  for v0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]


def read_vap(path: PathLike) -> Dict[str, float]:
    """Extract {morph display name: value} from an appearance preset."""
    raw = Path(path).read_text(encoding="utf-8")
    doc = json.loads(raw)
    values: Dict[str, float] = {}

    storables = doc.get("storables") or []
    for storable in storables:
        if storable.get("id") != "geometry":
            continue
        for entry in storable.get("morphs") or []:
            name = entry.get("name")
            if not name:
                continue
            try:
                values[name] = float(entry.get("value", 0))
            except (TypeError, ValueError):
                continue
    return values


def write_vap(path: PathLike, morphs: Dict[str, float],
              set_unlisted_to_default: bool = True) -> str:
    """Write a minimal morphs-only appearance preset. Returns the path."""
    storable: Dict[str, Any] = {
        "id": "geometry",
        "morphs": [
            {"name": name, "value": f"{float(value):.6g}"}
            for name, value in sorted(morphs.items())
        ],
    }
    doc = {
        "setUnlistedParamsToDefault": "true" if set_unlisted_to_default else "false",
        "storables": [storable],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=3, ensure_ascii=False), encoding="utf-8")
    return str(out)


def merge_morphs(base: Dict[str, float], override: Dict[str, float]) -> Dict[str, float]:
    """Overlay `override` on top of `base` (e.g. fit result over a base look)."""
    merged = dict(base)
    merged.update(override)
    return merged
