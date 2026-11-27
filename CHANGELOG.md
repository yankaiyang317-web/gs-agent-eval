# Unified hotel scenes (2026-10-18)

Default download now uses one scene archive containing both the original high-quality hotel and the degraded hotel, each with PLY, collision GLB and config. Scoring formulas and scene bytes unchanged.

# Hotel runtime supplement (2026-10-18)

Added the original high-quality hotel collision GLB, portable high-quality configs and default download registration. Evaluator version/formulas and frozen benchmark data are unchanged. Existing v0.1.1 release assets remain intact.

# Changes

## 0.1.1 — 2026-10-18

- Package the single hotel baseline as checksummed GitHub Release assets.
- Portable navigation manifest; navigation/reference data directories are separated.
- Add safe downloader, dataset card, original-Agent integration instructions and CPU CI.
- Provide tested dependency versions and a public CUDA Docker recipe; existing machine25 runtime remains supported.
- Keep the v1 scoring formulas, frozen states and semantic benchmark fingerprint unchanged; only asset locators are relocated.
- Preserve0.1.0 source/state release provenance. New packaging release does not retroactively change historical scores.
# Overlapping-view guju and laojie baselines (2026-11-27)

Default guju and laojie downloads now pair each unchanged high-quality PLY with a separately named, 15,000-step degraded PLY trained from more overlapping 960×720 synthetic views. New benchmark IDs bind separately recalibrated A/B states to the new PLY hashes. Guju includes a documented controlled local facade gap. Optional splattransform collision meshes are archived; default runtime configs use the stable reference collision meshes. The synthetic reconstruction image/depth files remain on machine25 and are not public in this release. Historical v2 scene assets, benchmark profiles, trajectories and results remain available for reproducibility and are not comparable to the new scene states.
