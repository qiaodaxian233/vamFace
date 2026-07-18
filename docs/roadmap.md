# Roadmap

## v0.1 (this draft)
- [x] VaM bridge plugin skeleton (TCP JSON server, morph r/w, screenshot)
- [x] MCP server with online + offline tools
- [x] `.vap` read/write (offline, tested)
- [x] Fitting loop: ArcFace scorer + CMA-ES / greedy, with safe degrade
- [x] Curated ~44-morph face subset
- [ ] **Validation pass on live VaM 1.22** — resolve every `TODO(verify)`
      in `plugin/VamFaceBridge.cs` (see `docs/protocol.md` for the list)

## v0.2 — make the loop actually good
- [ ] Face alignment/crop before scoring (both target and render), so the
      identity score isn't polluted by pose/background. Use the detector's
      landmarks to crop a canonical face box.
- [ ] Neutralize pose/expression on the VaM side before each capture
      (zero jaw rotation, eyes forward, neutral expression) for a fair compare.
- [ ] Coarse-to-fine: optimize `skull`+`jaw` groups first, freeze, then
      `eyes`/`nose`/`mouth`. Fewer dims per stage = faster convergence.
- [ ] Multi-view scoring (front + 3/4) to avoid overfitting to one angle.
- [ ] Cache screenshots keyed by morph vector hash to skip re-renders.

## v0.3 — route A warm start (big speedup)
- [ ] Single-image 3D face reconstruction (DECA / MICA / FLAME-based) →
      target mesh.
- [ ] Precompute Genesis 2 base mesh + per-morph vertex deltas once.
- [ ] Least-squares solve: find morph coefficients whose combined deltas
      best match the reconstructed target geometry → **seed** for the loop.
- [ ] With a good seed, the CMA-ES refine step should need far fewer iters.

## v0.4 — beyond geometry
- [ ] Skin tone / texture: sample dominant skin color from the photo, map
      to the closest installed skin, or drive a texture-tint slider.
- [ ] Optional VLM-in-the-loop scorer (screenshot + target → vision model
      returns a similarity/critique) as an alternative to ArcFace, useful
      for stylized/anime faces where face-recognition embeddings are weak.

## Known risks / things to watch
- Plugin sandbox: `TcpListener` expected OK (community precedent), but if
  blocked, fall back to a file-drop protocol (plugin polls a request file,
  writes a response file). Keep the client abstraction so this is swappable.
- Morph-name drift across installs: never hard-code trust in the curated
  names — always reconcile via `list_morphs`, treat misses as data.
- Screenshot size in VR: keep the `max_width` downscale; full-res encode
  per iteration would dominate the loop time.
