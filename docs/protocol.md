# VamFace Bridge Protocol (v0.1)

Transport: **TCP**, `127.0.0.1:8787` (configurable in the plugin UI).
Framing: **one JSON object per line**, UTF-8, `\n`-terminated.

The plugin (`VamFaceBridge.cs`) is the server; the MCP process is the client.

## Request

```json
{"id": "42", "cmd": "set_morphs", "args": {"atom": "Person", "values": {"Nose Width": 0.35}}}
```

- `id`   — arbitrary string, echoed back so the client can match responses.
- `cmd`  — command name (see below).
- `args` — command-specific object (may be omitted).

## Response

Success:
```json
{"id": "42", "ok": true, "data": {"applied": 1, "missing": []}}
```

Failure:
```json
{"id": "42", "ok": false, "error": "atom not found: Person"}
```

Responses on one connection are returned in request order, but clients
should still match on `id`. Errors are **always** returned as `ok:false`
with a message — the plugin never hard-fails silently and never blocks on
a guess (a missing morph is reported in `data.missing`, not an error).

## Commands

| cmd            | args                                                        | data                                              |
|----------------|-------------------------------------------------------------|---------------------------------------------------|
| `ping`         | —                                                           | `{version, app}`                                  |
| `list_atoms`   | —                                                           | `{atoms: [{uid, type}]}`                          |
| `list_morphs`  | `{atom, filter?, region?, limit?}`                          | `{count, total, morphs: [{name, uid, region, value, min, max}]}` |
| `get_morphs`   | `{atom, changed_only?}`                                     | `{values: {name: value}}`                         |
| `set_morphs`   | `{atom, values: {name: value}, clamp?}`                     | `{applied, missing: [name]}`                      |
| `reset_morphs` | `{atom}`                                                    | `{reset}`                                          |
| `load_scene`   | `{path}`                                                    | `{loading}`                                        |
| `focus_head`   | `{atom}`                                                    | `{focused}`                                        |
| `screenshot`   | `{max_width?}`                                              | `{width, height, png_base64}`                     |
| `list_storables` | `{atom}`                                                  | `{storables: [id]}`                               |
| `list_params`  | `{atom, storable}`                                          | `{floats, bools, colors, choosers, strings}`      |
| `get_param`    | `{atom, storable, param}`                                   | `{type, value, ...}` (colors: `{h,s,v}`)          |
| `set_param`    | `{atom, storable, param, value, type?}`                     | `{type, value?}`                                  |
| `call_action`  | `{atom, storable, action}`                                  | `{called}`                                        |
| `list_characters` | `{atom}`                                                 | `{characters: [name]}`                            |
| `set_character`| `{atom, name}`                                              | `{selected}`                                      |

## Reference implementation: mock server

`server/vamface_mcp/mock_vam.py` (`vamface-mock`) implements this entire
command table against a parametric renderer, and `server/tests/` pins the
behavior (missing-morph reporting, clamping, error shapes, screenshot
downscale). If the live plugin and the mock ever disagree, the protocol
doc + tests are the arbiter.

## Notes / open questions for the live-VaM validation pass

Several VaM API calls in the plugin are marked `TODO(verify)` because they
follow community-plugin convention but have not yet been run against
1.22.0.3. The ones to confirm first:

1. `DAZCharacterSelector.morphsControlUI` and
   `GenerateDAZMorphsControlUI.GetMorphs()` / `GetMorphByDisplayName()` /
   `GetMorphByUid()`.
2. `DAZMorph.startValue` as the default-value field for `reset` / `changed_only`.
3. `SuperController.FocusOnController(FreeControllerV3)` signature.
4. `SuperController.Load(string)` overload for loading a scene by path.
5. Whether `TcpListener` is permitted under the current plugin sandbox
   (community plugins like FacialMotionCapture use raw sockets, so this is
   expected to work; `HttpListener` is the one likely to be blocked).
6. Generic param accessors: `GetFloatParamNames` / `GetColorJSONParam` /
   `GetStringChooserJSONParam` etc., and `HSVColor` field names (`H/S/V`)
   plus its range convention (assumed 0-1; conversion isolated in
   `skin.rgb_to_vam_hsv`, fix there if it's 0-360).
7. `DAZCharacterSelector.characters` / `SelectCharacterByName` for skin
   switching.

Screenshots use `ReadPixels` on the end-of-frame backbuffer; in VR the
resolution can be large, hence the `max_width` downscale option.
