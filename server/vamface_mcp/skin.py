"""Level 0 skin support: sample skin tone from a photo, apply to VaM.

Scope (deliberately modest — this is the cheap tier):
  1. sample_skin_tone(photo) — estimate the subject's skin color.
  2. rgb_to_vam_hsv(rgb)     — convert to the HSV dict the bridge's
                               color params expect.
  3. apply_skin_color(...)   — push it onto a chosen color param.

Texture projection / generation (Level 1/2) is out of scope here; see
docs/roadmap.md.

Skin detection strategy (no OpenCV dependency):
  - If insightface is installed, use its detector to get the face box.
  - Otherwise fall back to the center crop (photos of characters are
    usually face-centered).
  - Within the box, mask pixels by the classic YCrCb skin range
    (Cr 133..173, Cb 77..127) and take the median color. If the mask is
    too small (stylized/anime skin tones often fall outside the classic
    range), degrade to the plain median of the box — degraded, not failed,
    and the result reports which path was used.
"""

from __future__ import annotations

import colorsys
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .bridge_client import BridgeClient


# ---------------------------------------------------------------------------
# Color math (pure numpy, no cv2)
# ---------------------------------------------------------------------------

def _rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> HxWx3 float YCrCb (ITU-R BT.601, JPEG offsets)."""
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return np.stack([y, cr, cb], axis=-1)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = _rgb_to_ycrcb(rgb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def _detect_face_box(rgb: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """Return ((x0,y0,x1,y1), method). Falls back to center crop."""
    try:
        from insightface.app import FaceAnalysis  # lazy, optional

        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(rgb[:, :, ::-1])
        if faces:
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)
            x0, y0, x1, y1 = [int(v) for v in faces[0].bbox]
            h, w = rgb.shape[:2]
            return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), "insightface"
    except Exception:
        pass
    h, w = rgb.shape[:2]
    return (w // 4, h // 4, w * 3 // 4, h * 3 // 4), "center_crop"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_skin_tone(image_path: str) -> Dict[str, Any]:
    """Estimate skin tone. Returns rgb/hex/hsv plus how it was derived."""
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    box, box_method = _detect_face_box(rgb)
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = rgb

    mask = _skin_mask(crop)
    if mask.sum() >= max(50, mask.size * 0.02):
        pixels = crop[mask]
        mask_method = "ycrcb_mask"
    else:
        pixels = crop.reshape(-1, 3)
        mask_method = "box_median_fallback"  # stylized tones may miss the mask

    med = np.median(pixels, axis=0)
    r, g, b = [int(round(float(v))) for v in med]
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

    return {
        "rgb": [r, g, b],
        "hex": f"#{r:02x}{g:02x}{b:02x}",
        "hsv": {"h": round(h, 4), "s": round(s, 4), "v": round(v, 4)},
        "face_box": list(box),
        "detector": box_method,
        "mask": mask_method,
        "pixels_used": int(pixels.shape[0]),
    }


def rgb_to_vam_hsv(rgb) -> Dict[str, float]:
    """[r,g,b] 0-255 -> {"h","s","v"} floats 0-1 (VaM HSVColor convention).

    TODO(verify): confirm VaM's HSVColor H is 0-1 (not 0-360) on live VaM;
    if it turns out to be 0-360, fix HERE (single conversion point).
    """
    r, g, b = [max(0, min(255, int(c))) / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return {"h": round(h, 4), "s": round(s, 4), "v": round(v, 4)}


def apply_skin_color(bridge: BridgeClient, atom: str, storable: str,
                     param: str, rgb) -> Dict[str, Any]:
    """Push an RGB tone onto one color param (discovered via find_color_params)."""
    hsv = rgb_to_vam_hsv(rgb)
    result = bridge.set_param(atom, storable, param, hsv, type="color")
    return {"applied": {"storable": storable, "param": param}, "hsv": hsv, **result}


def suggest_and_apply(bridge: BridgeClient, atom: str, image_path: str,
                      storable_filter: str = "skin") -> Dict[str, Any]:
    """One-shot Level 0: sample tone, find color params, apply to all hits.

    Conservative default: only storables whose id contains `storable_filter`.
    Returns what was applied and what was skipped so the caller can undo
    selectively (get_param before set could be added for full undo later).
    """
    tone = sample_skin_tone(image_path)
    hits = bridge.find_color_params(atom, storable_filter)
    applied = []
    failed = []
    for hit in hits:
        try:
            apply_skin_color(bridge, atom, hit["storable"], hit["param"], tone["rgb"])
            applied.append(hit)
        except Exception as e:
            failed.append({**hit, "error": str(e)})
    return {"tone": tone, "applied": applied, "failed": failed}
