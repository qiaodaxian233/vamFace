"""VamFace MCP server.

Exposes VaM face automation as MCP tools over stdio, so an MCP-capable
client (Claude Desktop, etc.) can:

  - inspect the running scene (list_atoms, list_morphs)
  - read/write morphs live (get_morphs, set_morphs, reset_morphs)
  - take a screenshot of the current face
  - save/load appearance presets (.vap) offline
  - run an automated fit of the face toward a target photo

Layering:
  online tools  -> BridgeClient -> VamFaceBridge plugin inside VaM
  offline tools -> vap.py (pure file I/O, VaM need not be running)

Env vars:
  VAMFACE_HOST (default 127.0.0.1)
  VAMFACE_PORT (default 8787)
  VAMFACE_OUTPUT_DIR (default ./out) — where screenshots/presets are written

Run:  python -m vamface_mcp.server
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .bridge_client import BridgeClient, BridgeError
from .vap import read_vap, write_vap, merge_morphs
from .morph_presets import default_face_morph_names, default_bounds
from .fitting import FitConfig, fit_face

HOST = os.environ.get("VAMFACE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VAMFACE_PORT", "8787"))
OUTPUT_DIR = Path(os.environ.get("VAMFACE_OUTPUT_DIR", "./out")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("vamface")
_bridge = BridgeClient(HOST, PORT)


def _err(e: Exception) -> Dict[str, Any]:
    return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

@mcp.tool()
def vam_ping() -> Dict[str, Any]:
    """Check that VaM and the VamFaceBridge plugin are reachable.

    Includes a protocol-version handshake: a `compat_warning` field appears
    when the plugin predates v0.5 or the protocol numbers mismatch.
    """
    try:
        return {"ok": True, **_bridge.handshake()}
    except BridgeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Scene / morph inspection (online)
# ---------------------------------------------------------------------------

@mcp.tool()
def vam_list_atoms() -> Dict[str, Any]:
    """List all atoms in the current scene (uid + type)."""
    try:
        return {"ok": True, **_bridge.list_atoms()}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_list_morphs(atom: str = "Person", filter: str = "",
                    region: str = "", limit: int = 200) -> Dict[str, Any]:
    """List morphs on a Person atom, optionally filtered by name/region."""
    try:
        return {"ok": True, **_bridge.list_morphs(atom, filter, region, limit)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_get_morphs(atom: str = "Person", changed_only: bool = True) -> Dict[str, Any]:
    """Read current morph values (changed-from-default by default)."""
    try:
        return {"ok": True, "values": _bridge.get_morphs(atom, changed_only)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_set_morphs(values: Dict[str, float], atom: str = "Person",
                   clamp: bool = True) -> Dict[str, Any]:
    """Set morph values on a Person atom. Unknown names are reported, not fatal."""
    try:
        return {"ok": True, **_bridge.set_morphs(atom, values, clamp)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_reset_morphs(atom: str = "Person") -> Dict[str, Any]:
    """Reset all changed morphs on a Person atom back to default."""
    try:
        return {"ok": True, **_bridge.reset_morphs(atom)}
    except BridgeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Screenshot (online)
# ---------------------------------------------------------------------------

@mcp.tool()
def vam_screenshot(atom: str = "Person", focus_head: bool = True,
                   max_width: int = 768) -> Dict[str, Any]:
    """Capture the VaM viewport. Saves a PNG and returns its path.

    If focus_head is set, the camera is pointed at the head first so the
    face is framed consistently.
    """
    try:
        if focus_head:
            try:
                _bridge.focus_head(atom)
                time.sleep(0.2)
            except BridgeError:
                pass  # focusing is best-effort, don't block the capture
        shot = _bridge.screenshot(max_width=max_width)
        png = base64.b64decode(shot["png_base64"])
        path = OUTPUT_DIR / f"shot_{int(time.time()*1000)}.png"
        path.write_bytes(png)
        return {"ok": True, "path": str(path),
                "width": shot.get("width"), "height": shot.get("height")}
    except BridgeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Appearance presets (offline .vap)
# ---------------------------------------------------------------------------

@mcp.tool()
def vap_read(path: str) -> Dict[str, Any]:
    """Read morph values from a .vap appearance preset file (offline)."""
    try:
        return {"ok": True, "values": read_vap(path)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def vap_write(morphs: Dict[str, float], path: Optional[str] = None) -> Dict[str, Any]:
    """Write a morphs-only .vap appearance preset (offline)."""
    try:
        out = path or str(OUTPUT_DIR / f"face_{int(time.time())}.vap")
        written = write_vap(out, morphs)
        return {"ok": True, "path": written, "count": len(morphs)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def vam_apply_vap(path: str, atom: str = "Person") -> Dict[str, Any]:
    """Load morphs from a .vap file and push them live into VaM."""
    try:
        morphs = read_vap(path)
        result = _bridge.set_morphs(atom, morphs, clamp=True)
        return {"ok": True, "count": len(morphs), **result}
    except (BridgeError, Exception) as e:
        return _err(e)


# ---------------------------------------------------------------------------
# The automation: fit face to a photo
# ---------------------------------------------------------------------------

@mcp.tool()
def vam_fit_face(target_image: str, atom: str = "Person",
                 optimizer: str = "cma", max_iters: int = 60,
                 groups: Optional[List[str]] = None,
                 save_vap: bool = True, style: str = "auto",
                 anime_onnx: Optional[str] = None,
                 use_prior: bool = True, neutralize: bool = True,
                 coarse_to_fine: bool = False,
                 use_basis: bool = False) -> Dict[str, Any]:
    """Automatically fit the VaM face to a target photo.

    Runs a black-box optimization loop: set morphs -> screenshot ->
    identity-similarity score -> repeat, keeping the best. Requires VaM
    running with a Person atom and the target photo readable on disk.

    Args:
      target_image: path to the target face photo.
      optimizer: "cma" (needs `cma` pkg) or "greedy" (dependency-free).
      max_iters: evaluation budget (each iter is one screenshot round-trip).
      groups: subset of morph regions to optimize
              (skull/jaw/cheeks/eyes/nose/mouth/ears); default = all.
      save_vap: also write the result as a .vap preset.
      style: scorer style — "auto" (detect from target), "real" (ArcFace),
             "anime" (animeface landmark geometry; use for stylized faces
             where ArcFace embeddings are meaningless), "pixel" (coarse,
             for mock-server testing).
      anime_onnx: optional path to a user-supplied anime face-recognition
             ONNX model, blended into the anime scorer.
      use_prior: seed the optimizer from geometry feature deltas (one probe
             evaluation; only when the scorer stack has a geometry part).
      neutralize: zero expression-type morphs before fitting so the loop
             can't cheat identity similarity with a smile.
      coarse_to_fine: two-stage fit — contour groups (skull/jaw/cheeks)
             first, frozen, then features (eyes/nose/mouth/ears).
      use_basis: scan installed full-head character morphs first and adopt
             the closest one as the starting point (v0.6; one evaluation
             per candidate, drawn from the same budget).

    NOTE: if no identity scorer is installed (insightface), the score is a
    placeholder 0.0 and `warning` will say so — the loop still runs but the
    result is not meaningful. Install insightface for real fitting.
    """
    try:
        from .morph_presets import FACE_MORPH_GROUPS
        if groups:
            names: List[str] = []
            for g in groups:
                names.extend(FACE_MORPH_GROUPS.get(g, []))
        else:
            names = default_face_morph_names()

        cfg = FitConfig(
            atom=atom,
            morph_names=names,
            bounds={n: default_bounds(n) for n in names},
            max_iters=max_iters,
        )
        result = fit_face(_bridge, target_image, cfg, optimizer=optimizer,
                          style=style, anime_onnx=anime_onnx,
                          use_prior=use_prior, neutralize=neutralize,
                          coarse_to_fine=coarse_to_fine, use_basis=use_basis)

        out: Dict[str, Any] = {
            "ok": True,
            "best_score": result.best_score,
            "style": result.style,
            "scorer": result.scorer_name,
            "morph_count": len(result.best_morphs),
            "iterations": len(result.history),
            "stages": result.stage_count,
        }
        if result.hints:
            out["hints"] = result.hints
        if result.basis:
            out["basis"] = result.basis
        if result.basis_missing:
            out["basis_missing"] = result.basis_missing
        if result.renamed:
            out["renamed"] = result.renamed
        if result.missing:
            out["missing"] = result.missing
        if result.neutralized:
            out["neutralized_expressions"] = result.neutralized
        if result.prior_seed:
            out["prior_seed"] = result.prior_seed
        if result.warning:
            out["warning"] = result.warning
        if save_vap:
            vap_path = OUTPUT_DIR / f"fit_{int(time.time())}.vap"
            write_vap(vap_path, result.best_morphs)
            out["vap_path"] = str(vap_path)
        return out
    except (BridgeError, Exception) as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Generic storable/param access (runtime discovery, e.g. skin color params)
# ---------------------------------------------------------------------------

@mcp.tool()
def vam_list_storables(atom: str = "Person") -> Dict[str, Any]:
    """List storable ids on an atom (entry point for parameter discovery)."""
    try:
        return {"ok": True, "storables": _bridge.list_storables(atom)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_list_params(storable: str, atom: str = "Person") -> Dict[str, Any]:
    """List a storable's params grouped by type (floats/bools/colors/choosers/strings)."""
    try:
        return {"ok": True, **_bridge.list_params(atom, storable)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_get_param(storable: str, param: str, atom: str = "Person") -> Dict[str, Any]:
    """Read one param's value (type auto-detected)."""
    try:
        return {"ok": True, **_bridge.get_param(atom, storable, param)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_set_param(storable: str, param: str, value: Any, type: str = "",
                  atom: str = "Person") -> Dict[str, Any]:
    """Write one param. `value`: number/bool/string, or {"h","s","v"} for colors."""
    try:
        return {"ok": True, **_bridge.set_param(atom, storable, param, value, type)}
    except BridgeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Skin: Level 0 (tone sampling + character selection + color application)
# ---------------------------------------------------------------------------

@mcp.tool()
def skin_sample_tone(image: str) -> Dict[str, Any]:
    """Sample the skin tone of a photo (offline). Returns rgb/hex/hsv."""
    try:
        from .skin import sample_skin_tone
        return {"ok": True, **sample_skin_tone(image)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def vam_list_characters(atom: str = "Person") -> Dict[str, Any]:
    """List installed character skins on a Person atom."""
    try:
        return {"ok": True, "characters": _bridge.list_characters(atom)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_set_character(name: str, atom: str = "Person") -> Dict[str, Any]:
    """Switch the Person's character skin by display name."""
    try:
        return {"ok": True, **_bridge.set_character(atom, name)}
    except BridgeError as e:
        return _err(e)


@mcp.tool()
def vam_apply_skin_tone(image: str, atom: str = "Person",
                        storable_filter: str = "skin") -> Dict[str, Any]:
    """Level 0 one-shot: sample the photo's skin tone and write it to every
    color param on storables matching the filter. Returns applied/failed
    per-param so partial success is visible (nothing hard-fails)."""
    try:
        from .skin import suggest_and_apply
        return {"ok": True, **suggest_and_apply(_bridge, atom, image, storable_filter)}
    except (BridgeError, Exception) as e:
        return _err(e)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
