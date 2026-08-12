# vamFace

An **MCP server + VaM plugin** that automates face creation in
**Virt-A-Mate 1.22.0.3**: point it at a face photo and it drives the
in-game morph sliders until the rendered face matches.

> ⚠️ Status: **v0.5**. The offline layer (`.vap` read/write) is tested; the
> whole pipeline (bridge client ↔ protocol ↔ fitting ↔ export) is
> exercised end-to-end against the bundled **mock VaM** (`vamface-mock`,
> 59 tests, CI on 3.10/3.12). What still needs a live VaM 1.22 pass is only
> the plugin-side API names marked `TODO(verify)` — see `docs/protocol.md`.

> 🤖 本项目由仓库主人与 AI 结对持续开发。跨对话接续协议与项目状态见
> **[《对话记忆.md》](对话记忆.md)** —— 新对话把仓库地址发给 AI 即可无缝续接。

## How it works

```
┌──────────────────────────── MCP client (Claude Desktop, …) ───────────────┐
│                                                                            │
│   vam_fit_face(photo)  vam_set_morphs  vam_screenshot  vap_read/write …    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │ MCP (stdio)
┌───────────────────────────────────▼────────────────────────────────────────┐
│ vamface_mcp  (Python, this repo /server)                                    │
│                                                                             │
│  offline: vap.py        .vap preset read/write  (no VaM needed)             │
│  online : bridge_client ─── TCP JSON ───► VaM plugin                        │
│  fit    : fitting.py    set_morphs → screenshot → identity score → repeat   │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │ TCP 127.0.0.1:8787
┌───────────────────────────────────▼────────────────────────────────────────┐
│ VamFaceBridge.cs  (VaM session plugin, this repo /plugin)                    │
│  reads/writes morphs, focuses head, captures screenshots                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

The key idea: **VaM appearance is JSON**. Morph values are a flat
`{name: value}` map, so a lot of work (building/reading/merging presets)
happens fully offline. Only the fitting loop needs a live VaM.

### Photo → face: the fitting loop

Black-box optimization with VaM in the loop:

1. propose a morph vector (over a curated ~44-morph subset, not all 1000+)
2. `set_morphs` → `screenshot`
3. score = **ArcFace identity cosine similarity** between the target photo
   and the rendered face (via `insightface`)
4. repeat with CMA-ES (or a dependency-free greedy fallback), keep the best
5. save the winner as a `.vap` preset

A future **route A** (single-image 3D face reconstruction → least-squares
solve for morph coefficients) can produce a warm-start seed to cut the
iteration count; the loop above then only refines. See `docs/roadmap.md`.

## Install

### 1. VaM plugin
Copy `plugin/VamFaceBridge.cs` to
`(VaM)/Custom/Scripts/VamFace/VamFaceBridge.cs`, then in VaM:
**Session Plugins → Add Plugin →** select the file. It opens
`127.0.0.1:8787`.

### 2. MCP server
```bash
cd server
pip install -e .            # base (offline + live morph control)
pip install -e ".[fit]"     # + insightface/onnxruntime/cma for real fitting
```

Register with your MCP client (example for Claude Desktop config):
```json
{
  "mcpServers": {
    "vamface": { "command": "vamface-mcp" }
  }
}
```

## Scorer styles (anime vs real)

ArcFace embeddings are trained on real human photos — on anime/stylized
faces they are basically noise, and the loop would optimize toward nothing.
The scorer is therefore **switchable** (`--style` on the CLI, a radio in the
GUI, `style=` on the MCP tool):

| style   | what it uses                                                     | use when |
|---------|------------------------------------------------------------------|----------|
| `real`  | ArcFace identity (0.75) + landmark geometry (0.25)               | photoreal targets |
| `anime` | animeface landmark geometry (+ optional user ONNX embedding via `--anime-onnx`) | anime / stylized targets (`pip install -e ".[anime]"`) |
| `pixel` | downsampled grayscale RMSE similarity                            | mock testing, render-vs-render |
| `auto`  | probes the target with both detectors, picks accordingly         | default |

Every style degrades gracefully with a warning when its dependency is
missing — the loop never crashes on a missing model.

The geometry scorer also emits **directional hints** after the fit
("target eye spacing larger → Eyes Width Spacing ↑"), printed by the CLI
and shown in the GUI — the black-box score says *how unlike*, the feature
deltas say *what to change*.

## Fitting quality (v0.4)

Four things run before/around the optimizer, all on by default:

- **Crop before scoring** — the whole-image scorers (pixel, user ONNX) are
  cropped to the subject box so background/composition can't pollute the
  score. ArcFace/geometry detect internally and are left alone.
- **Expression neutralization** — expression-type morphs (smile/blink/…)
  are zeroed before fitting so the loop can't cheat identity similarity
  with a grin (`--no-neutralize` to skip).
- **Geometry-prior seeding** — when the scorer stack has a landmark
  geometry part, one probe evaluation turns feature deltas into an initial
  morph seed and a coordinate order. Measured on the mock: evaluations to
  reach score 0.95 dropped **35 → 14 (-60%)** — on a live VaM each saved
  evaluation is a full screenshot round-trip (`--no-prior` to skip).
- **Coarse-to-fine** — `--coarse-to-fine` fits contour groups
  (skull/jaw/cheeks) first, freezes them, then features
  (eyes/nose/mouth/ears); budget split by dimensionality.

## v0.5 quality-of-life

- **Protocol handshake** — `ping` now reports a protocol version; server,
  CLI and GUI warn (never block) when plugin and server drift apart.
- **Screenshot cache** — exact revisits of a morph vector reuse the score
  instead of a full set-morphs/screenshot round-trip (on by default,
  `--no-cache` to disable; hit count reported in results).
- **Manual tune tab** (GUI) — sliders for the 44 curated morphs, written
  into VaM live on release; load current values, zero all, refresh
  screenshot, import/export `.vap`. For hand-finishing the last few percent
  after a fit.
- **Updater** — a "检查更新" button compares the installed version against
  GitHub main; the 连接调试 tab can write the bundled plugin `.cs` straight
  into your VaM folder (old file backed up, path remembered in
  `~/.vamface/config.json`). Reload the session plugin in VaM afterwards.

## Mock VaM (test without VaM)

`vamface-mock` is a fake VaM: same TCP protocol, but rendering a smooth
parametric cartoon face driven by the same 44 curated morph names. It
unblocks all development on machines without VaM:

```bash
vamface-mock --seed 42          # prints a hidden target morph vector,
                                # saves mock_target_42.png
vamface-fit mock_target_42.png --style pixel --optimizer cma --iters 120
```

The fit should recover the hidden morphs — this is the standing regression
that exercises bridge client, protocol, screenshot decode, scorers,
optimizers and `.vap` export in one go (`server/tests/`).

## GUI

A local web console (Gradio) — no MCP client needed:

```bash
pip install -e ".[gui]"
vamface-gui        # opens http://127.0.0.1:7860
```

Tabs: **照片拟合** (drop a photo, watch target-vs-render + score curve live,
export `.vap`), **皮肤 L0** (sample the photo's skin tone, switch character
skins, write the tone into runtime-discovered color params), **连接调试**
(ping / atoms / storable inspector).

## Skin (Level 0)

Geometry fitting and skin are separate problems. v0.2 ships the cheap tier:
sample the dominant skin tone from the photo (YCrCb mask inside the face
box, graceful fallback for stylized tones), pick the closest installed
character skin, and write the tone into skin color params. Param names are
**discovered at runtime** via the new generic storable API — nothing is
hard-coded. Texture projection/generation is roadmap v0.4.

## Using without MCP

The MCP server is only one frontend. The same fitting loop is available as
a plain CLI (talks to the same VaM plugin over TCP):

```bash
vamface-fit photo.png --optimizer cma --iters 80
vamface-fit photo.png --groups skull jaw --optimizer greedy   # coarse pass
```

A fully offline path (no VaM running at all: 3D reconstruction →
least-squares morph solve → `.vap`) is planned as v0.3, see `docs/roadmap.md`.

## MCP tools

| tool               | needs VaM | what it does                                  |
|--------------------|:---------:|-----------------------------------------------|
| `vam_ping`         | ✓         | check plugin connectivity                     |
| `vam_list_atoms`   | ✓         | list scene atoms                              |
| `vam_list_morphs`  | ✓         | list/filter morphs on a Person                |
| `vam_get_morphs`   | ✓         | read current morph values                     |
| `vam_set_morphs`   | ✓         | write morph values                           |
| `vam_reset_morphs` | ✓         | reset morphs to default                       |
| `vam_screenshot`   | ✓         | capture the viewport (focuses head first)     |
| `vam_apply_vap`    | ✓         | push a `.vap`'s morphs into VaM live          |
| `vap_read`         | ✗         | read morphs from a `.vap`                     |
| `vap_write`        | ✗         | write a morphs-only `.vap`                    |
| `vam_fit_face`     | ✓         | **auto-fit the face to a target photo**       |

## Repo layout

```
plugin/VamFaceBridge.cs        VaM session plugin (C#)
server/
  vamface_mcp/
    server.py                  MCP tool definitions
    bridge_client.py           TCP client for the plugin
    vap.py                     .vap read/write (offline, tested)
    fitting.py                 scorers + optimizers + fit loop
    morph_presets.py           curated face-morph subset
  pyproject.toml
docs/
  protocol.md                  TCP JSON protocol + TODO(verify) list
  roadmap.md                   what's next
对话记忆.md                     collaboration notes for the next AI instance
```

## Scope note

This project targets **virtual/original character images**, not the
recreation of real identifiable people. Building likenesses of real people
without consent raises portrait-right and deepfake concerns; that use is
out of scope.

## License

TBD by the repo owner.
