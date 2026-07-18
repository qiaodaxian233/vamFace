# vamFace

An **MCP server + VaM plugin** that automates face creation in
**Virt-A-Mate 1.22.0.3**: point it at a face photo and it drives the
in-game morph sliders until the rendered face matches.

> ⚠️ Status: **initial draft (v0.1)**. The offline layer (`.vap` read/write)
> is tested and working. The online layer (VaM plugin + fitting loop) is
> written but its VaM API calls are marked `TODO(verify)` and need one
> validation pass against a live VaM 1.22 — see `docs/protocol.md`.

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
