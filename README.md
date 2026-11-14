[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24231012.svg)](https://doi.org/10.5281/zenodo.24231012)
# Phase 3 reproducibility package: DVHN400, depth-conditioned evaluation, and field-direction analysis

This repository contains the DVHN400 selection manifest, a self-contained fixed-seed sampling script, the complete 400-image DVHN400 depth cache, the frozen Stage-A detector weight, the DDM `best.pt` and `last.pt` checkpoints, and the Phase 3 DDM/3D evaluation code snapshot used for the accompanying manuscript. It is prepared for a code-only GitHub repository and a full Zenodo archival ZIP.

## Release status

**Verified in this snapshot**

- `data/dvhn400/manifest.jsonl` has 400 unique RGB–IR pairs and 400 unique source indices.
- The source protocol has 8,980 test pairs and a 3,260-image night candidate pool.
- Recomputing `numpy.random.default_rng(20261002).permutation(pool)[:400]` from the original brightness cache reproduces every source index in the manifest, in the same order.
- Every rounded brightness value in the manifest agrees with the brightness cache.
- All 400 RGB and 400 IR files in the inventoried local copy exist and are 640 × 512 pixels.
- `depth_cache/dvhn400/rgb/` contains exactly 400 uint16 PNGs, all 640 × 512, with filenames exactly matching the manifest and nonconstant depth values.
- `weights/frp_stageA_v1_best.pt` is the supplied frozen Stage-A detector weight; its size, SHA-256, and PyTorch ZIP-container CRC have been checked.
- `weights/ddm_v3_train_20261001_best.pt` and `weights/ddm_v3_train_20261001_last.pt` are distinct supplied DDM checkpoints. Both passed PyTorch ZIP-container CRC and restricted checkpoint-structure checks without executing project code.
- The package contains 57 Python files from the Phase 3 evaluation snapshot, including the veto-only DDM hook, mainline and attribution drivers, direction evaluation, OOD evaluation, cache construction, and analysis utilities.

**Not included / not yet reproducible from this archive alone**

- Original public datasets and the complete Phase 2/ultralytics environment remain external inputs.
- Exact server package versions and a hardware-identifying environment lockfile were not available in the inventoried files.
- Sampling, manifest, image, depth-cache, checkpoint-container, and file-integrity checks have been run locally. GPU model loading, depth reconstruction, and detection evaluation have **not** been rerun for this repository snapshot.

## Repository layout

```text
configs/dvhn400.env                  environment template; edit a private copy
data/dvhn400/manifest.jsonl          400-pair evaluation manifest
data/dvhn400/sampling_meta.json      selection protocol and summary
scripts/dvhn400/sample_dvhn400.py    source-to-subset reproduction
scripts/dvhn400/validate_dvhn400.py  manifest/image/seed validator
scripts/phase3/                     Phase 3 evaluation and module snapshot
depth_cache/dvhn400/rgb/             400 verified uint16 depth PNGs
depth_cache/dvhn400/depth_cache_meta.json
weights/frp_stageA_v1_best.pt        frozen Stage-A detector weight
weights/ddm_v3_train_20261001_best.pt
weights/ddm_v3_train_20261001_last.pt
validation/                         machine-readable local validation reports
checksums.sha256                    SHA-256 hashes of released files
```

The original `rgb/` and `ir/` images are not redistributed here; they can be regenerated from the public DroneVehicle test split. Large depth caches and model weights are excluded from code-only GitHub staging by `.gitignore`, but are included in the full Zenodo candidate ZIP. Virtual environments and private `.env` files are excluded.

## 1. Reproduce DVHN400

Download DroneVehicle from its official repository: <https://github.com/VisDrone/DroneVehicle>. Use the test split containing `test/testimg/` and `test/testimgr/`.

Install the two sampling dependencies:

```bash
python -m pip install -r requirements-sampling.txt
```

Run the sampler with a new output path:

```bash
python scripts/dvhn400/sample_dvhn400.py \
  --source-root /path/to/DroneVehicle/test/test \
  --brightness-cache /path/to/test_brightness.npy \
  --out /path/to/DVHN400
```

If the brightness cache does not exist, the script computes and writes it. The cache is optional on later runs; deleting it does not change the algorithm.

The frozen protocol is:

1. Lexicographically sort the 8,980 RGB test filenames.
2. Crop both modalities with `(100, 100, 740, 612)`, producing 640 × 512 images.
3. Convert each cropped RGB image to 8-bit grayscale with Pillow and compute its mean.
4. Keep means below 50.0, producing 3,260 candidates.
5. Apply NumPy `default_rng(20261002).permutation(pool)[:400]` and sort the selected indices.
6. Save cropped JPEGs at quality 95 and write the JSONL manifest.

DVHN400 is a negative-only hard-negative cohort. It evaluates smoke/fire false-positive suppression and the person exemption; it does not measure fire recall.

## 2. Validate the released manifest

Manifest and metadata only:

```bash
python scripts/dvhn400/validate_dvhn400.py \
  --dataset-root data/dvhn400 --manifest-only
```

Full validation with reconstructed images and the original brightness cache:

```bash
python scripts/dvhn400/validate_dvhn400.py \
  --dataset-root /path/to/DVHN400 \
  --brightness-cache /path/to/test_brightness.npy \
  --output validation/dvhn400_full_validation.json
```

A passing full check verifies row count, uniqueness, image presence, dimensions, all 400 selected source indices, and rounded brightness values.

## 3. Use or regenerate the depth cache

The full archive supplies all 400 verified uint16 PNGs under `depth_cache/dvhn400/rgb/` and the corresponding metadata. Copy this `depth_cache/` directory into the reconstructed DVHN400 root so that each `rgb/<name>.jpg` has `depth_cache/rgb/<name>.png`.

To regenerate the cache instead, use frozen Depth Anything V2 Large weights. Set `PHY_DEPTH_REPO` and `PHY_DEPTH_WEIGHTS` in a private copy of `configs/dvhn400.env`; do not put local paths or credentials under version control.

```bash
source /path/to/private/dvhn400.env
python "$PKG/tools/build_depth_cache.py" \
  --dataset-root "$DV" --images "$DV/rgb" --device cuda:0 --batch 8
```

Evaluation divides stored integers by 65,535; normalized value 1 means far. Missing cache files are fatal by design. The evaluator does not silently substitute an all-one depth map.

## 4. Evaluation environment and commands

This is an overlay on the frozen project, not a standalone replacement for ultralytics. The archive supplies the Stage-A detector weight, DDM checkpoints, and DVHN400 depth cache. The remaining external inputs are:

- the compatible ultralytics/Phase 2 source tree;
- `FieldGuide_Phase2_FRP_20260916` and `FieldGuide_Phase2_FieldUse_20260916`;
- RGBT-3M validation images, labels, and depth cache;
- the reconstructed DVHN400 RGB/IR images, or another selected hard-negative dataset and its depth cache.

Set `W` to `weights/frp_stageA_v1_best.pt` and `CKPT` to `weights/ddm_v3_train_20261001_last.pt`, as in the supplied `configs/dvhn400.env` template. The earlier backbone archive at <https://doi.org/10.5281/zenodo.22646595> is a separate reference; it does not replace the supplied Stage-A weight or DDM checkpoint.

On a disposable copy of the compatible source tree, install the five Phase 3 module files:

```bash
python "$PKG/install_phase3.py" --ultralytics-root "$ROOT/ultralytics"
```

Keep a backup of the target tree. Then run the DVHN400 chain at the main operating point:

```bash
PHY_DDM=1 PHY_DDM_VETO_T=0.001 python -u "$PKG/eval_ddm_val.py" \
  --weights "$W" --ddm-ckpt "$CKPT" \
  --rgbt-root "$RGBT" --rgbt-cache "$RGBT/depth_cache" \
  --m3fd-root "$DV" --hardneg-manifest "$DV/manifest.jsonl" \
  --phase2-package "$P2" --device cuda:0 --tau 0.045 \
  --depth-ablation full --require-veto-only-hook \
  --out "$RUNS/dvhn400_theta_0.001"
```

Use `PHY_DDM_VETO_T=0` for the no-veto comparison and `0.005` for the parallel operating point, each with a new output directory. Outputs include `metrics.json`, `detections3d.jsonl`, and `hardneg_per_image.jsonl`.

`run_qfusion_removed_mainline.py` preserves the original preregistered three-chain driver and acceptance alarms. It intentionally requires an equivalence report and `cuda:2`; use `eval_ddm_val.py` directly for a different GPU. `run_depth_attribution.py` implements the full/shuffled/constant/RGB-only depth-attribution comparison and requires the corresponding manifests and fixed shuffle table. These scripts do not manufacture missing inputs.

## Source datasets

- RGBT-3M: Zhang, Rui, and Song, *Remote Sensing* 17(15), 2593 (2025), DOI `10.3390/rs17152593`.
- M3FD: Liu et al., CVPR 2022, “Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection.”
- IGNITE: <https://github.com/Ding-Research-Lab/IGNITE>.
- FLAME 2: DOI `10.21227/swyw-6j78`.
- DroneVehicle: <https://github.com/VisDrone/DroneVehicle>.

Users must comply with each source dataset's terms. No license is asserted here for third-party datasets or weights.

## Verification scope

`validation/dvhn400_full_validation.json` records the actual local DVHN400 selection and image check. `validation/artifact_validation.json` records the 400-PNG cache check and model-weight hashes. `validation/ddm_checkpoint_validation.json` records the DDM container CRC and restricted checkpoint-structure checks. `checksums.sha256` supports file-integrity verification after download. Python syntax compilation, cache checks, checkpoint-container checks, and repository scans are release checks only; they are not evidence that GPU inference has been rerun or that numerical paper results have been regenerated.

Before public release, complete `UPLOAD_CHECKLIST.md` and choose an appropriate license for the authors' original code. The remaining reproducibility limitations are the external datasets, compatible source environment, and clean-checkout GPU evaluation.
