# Hotel scenes: high quality and low quality

This single-scene benchmark pairs a degraded hotel Gaussian scene for Agent navigation with a high-quality Gaussian reference for offline evaluation.

## What is included

- Navigation: `jiudian_degraded.ply`, its generated capsule-collision GLB, and a portable scene manifest. Start `[0,0,-0.15]`, camera quaternion wxyz `[0,1,0,0]`, world up `+Y`, capsule radius `.25`, body height `.9`, collision step `.02`.
- Offline reference: the owner-provided hotel Gaussian PLY. It is pseudo-GT, not measured physical ground truth.
- High-quality hotel runtime: the original hotel collision GLB and portable scene config (`scenes/jiudian.json` / `reference/scene.json`). The PLY is reused from the existing scene archive. Original start `[0,0,0]`, camera and collision parameters are preserved. This is for separate high-quality hotel runs, not the degraded evaluation protocol.
- Frozen state: 24,087 reference voxels and defect weights; reliability from all128 original training RGB-D views with depth-derived normals. The32 original held-out views are excluded from calibration.
- Optional reconstruction inputs: all160 synthetic RGB PNGs at1920×1440 and saved RGB-D initialization observations at480×360, camera poses/intrinsics, split/training metadata, and documented defects.

Both hotel scenes are distributed together in one scene archive and downloaded by default, with separate configs. The high-quality `jiudian` is the original Agent default scene; low-quality `jiudian_degraded` is the frozen evaluation scene.

Download scripts verify release archive SHA256 and each extracted file. The frozen scoring protocol is `benchmarks/jiudian_degraded_full_inputs_v1.json`.

## Provenance

The original high-quality hotel PLY was supplied for this project. Original photographs/COLMAP were unavailable. Twenty collision-connected camera locations were selected; eight headings per location produced160 synthetic views. A new Gaussian model was initialized from the synthetic RGB-D, then trained on128 RGB views for2400 steps at960×720. Fixed topology/SH0 limit reconstruction quality. Controlled surface deletions and detached fragments were subsequently applied and recorded in`defects.json`.

The navigation collision mesh was generated from the degraded PLY with splat-transform3.0.0, smooth extraction, voxel`.04`, opacity threshold`.8`. It has known additional blocking relative to the original collision proxy; do not silently replace it in v1 experiments. Original source asset hashes remain unchanged in the release.

## Intended use and separation

Agent navigation receives only`datasets/jiudian/navigation/`. Keep`reference/`, reconstruction inputs, reliability and scoring reports outside the acting Agent's filesystem/tools. The reference and scoring state belong to a separate offline process.

Use identical starts, FOV, collision settings and81 recorded-frame budgets when comparing Agent variants. Three consecutive clips are segments of one run, not three independent repeats. Camera conventions are OpenCV C2W; world Y is height.

## Limitations

The scene is a deliberately degraded synthetic reconstruction, not naturally incomplete real capture. The reference domain is partial and not demonstrated to have converged to all physical surfaces. B is a frozen scalar3DGS view-utility adaptation, not a full reproduction of2D GauSS-MI. It does not remove between-frame redundancy and cannot fully score absent Gaussians. Depth-derived normal differences can amplify depth noise.

GT-assisted RGB-D supplementation has a small single-scene pilot; RGB-only or generative video enhancement is not validated by that result. Public availability does not by itself select a new software/data license; no additional third-party license is asserted here.
