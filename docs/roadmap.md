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
- [x] *(shipped in v0.4)* Face crop before scoring for the whole-image
      scorers (pixel / user ONNX). Note: ArcFace & landmark geometry
      detect+align internally, so the v0.2 wording overstated the debt —
      the truly polluted paths were pixel/onnx, and those are now cropped
      (`CroppedScorer`, `bbox_from_background`, `box_from_animeface`).
- [x] *(shipped in v0.4)* Neutralize expression morphs before fitting
      (`neutralize_expression`, pattern-based, identity morphs untouched).
      Eye-gaze/head-pose reset is a storable param → live-validation list.
- [x] *(shipped in v0.4)* Coarse-to-fine staged fitting (contour groups →
      freeze → feature groups), budget split by dimensionality.
- [ ] Multi-view scoring (front + 3/4) to avoid overfitting to one angle.
- [ ] Cache screenshots keyed by morph vector hash to skip re-renders.

## v0.4 (shipped) — fitting-quality four-pack + CI
- [x] Geometry-prior seeding: one probe evaluation → feature deltas →
      optimizer seed (CMA x0 / greedy start + coordinate order), gains
      calibrated against measured mock feature-morph slopes.
      **Measured on the mock: evals-to-0.95 dropped 35 → 14 (-60%).**
- [x] The three v0.2 debts above (crop / neutralize / coarse-to-fine).
- [x] GitHub Actions CI: full suite (43 tests) on Python 3.10 & 3.12,
      every push — the mock made tests dependency-light enough for CI.

## v0.3 (shipped) — scorer stack + mock VaM
- [x] Switchable scorer styles: real / anime / pixel / auto, all with
      graceful degrade + warnings (`scorers.py`).
- [x] Anime path: animeface landmark geometry scorer; optional user-supplied
      anime-recognition ONNX embedding (`--anime-onnx`).
- [x] Directional hints from feature deltas ("Eyes Width Spacing ↑"),
      surfaced in CLI / GUI / MCP result.
- [x] Mock VaM server (`vamface-mock`): full protocol, parametric face
      renderer, hidden-target mode; 27-test suite incl. hidden-target
      end-to-end fit. Live-VaM validation demoted from blocker to final
      reconciliation.
- [x] Bugfix caught by the mock: `cma_optimize` numpy-truthiness crash on
      result assembly (`best_vec or x0`).

## v0.4 — route A warm start (big speedup)
- [ ] Single-image 3D face reconstruction (DECA / MICA / FLAME-based) →
      target mesh.
- [ ] Precompute Genesis 2 base mesh + per-morph vertex deltas once.
- [ ] Least-squares solve: find morph coefficients whose combined deltas
      best match the reconstructed target geometry → **seed** for the loop.
- [ ] With a good seed, the CMA-ES refine step should need far fewer iters.

## v0.5 — beyond geometry
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
