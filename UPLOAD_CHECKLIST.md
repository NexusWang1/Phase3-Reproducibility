# GitHub / Zenodo upload checklist

## Completed in this snapshot

- [x] Located the real DVHN400 `manifest.jsonl` and `sampling_meta.json`.
- [x] Confirmed 400 manifest rows, 400 unique RGB paths, and 400 unique source indices.
- [x] Confirmed 8,980 source pairs and a 3,260-image candidate pool from the original brightness cache.
- [x] Reproduced all 400 selected indices with seed 20261002.
- [x] Checked all 800 cropped images in the local source copy; every pair exists and is 640 × 512.
- [x] Recovered the exact sampling operations and made source/output paths configurable.
- [x] Collected the current veto-only Phase 3 evaluation snapshot rather than the older `review_round2` evaluator.
- [x] Excluded virtual environments, bytecode, raw public datasets, credentials, and tool/runtime snapshots.
- [x] Replaced the functional server-specific depth-repository fallback with a package-relative path; depth weights remain configurable by environment variables.
- [x] Added all 400 DVHN400 uint16 depth-cache PNGs and `depth_cache_meta.json`.
- [x] Verified exact manifest/cache filename matching, 640 × 512 dimensions, uint16 type, and nonconstant depth for every PNG.
- [x] Added the frozen Stage-A detector `best.pt`; verified its size, SHA-256, and PyTorch ZIP-container CRC.
- [x] Added the distinct DDM `best.pt` and `last.pt` checkpoints; verified their sizes, SHA-256 hashes, ZIP-container CRCs, and expected checkpoint keys without executing project code.
- [x] Pointed the evaluation template to the supplied DDM `last.pt`; retained `best.pt` separately for checkpoint-trajectory diagnostics.
- [x] Normalized server-absolute visual-QA paths in the cache metadata to dataset-relative paths without changing depth PNG bytes.

## Blocking gaps before calling the archive fully reproducible

- [ ] Export the actual compatible Python/CUDA dependency environment and record the source-tree revision.
- [ ] Rerun at least the DVHN400 main operating point from a clean checkout and compare `metrics.json` with the reported result.
- [ ] Review and select a license for the authors' original code. No license file has been invented in this package.

## Release hygiene

- [ ] Replace every `/path/to` value in a **private copy** of `configs/dvhn400.env`; do not commit that private file.
- [ ] Check that no `.env`, credentials, server home paths, private manifests, or unrelated experiment outputs are staged.
- [ ] Verify `checksums.sha256` after copying to the upload machine.
- [ ] Test the uploaded ZIP with a ZIP integrity checker and compare its file list with the repository.
- [ ] Create the GitHub repository and push only after the authors approve public visibility.
- [ ] Create a GitHub release/tag and link that release to Zenodo.
- [ ] Reserve/review the Zenodo DOI and add the real repository URL and DOI to the manuscript only after they exist.
- [ ] Keep the raw public datasets out of the archive unless their terms and the release plan explicitly require redistribution.

## Current upload state

No GitHub repository or Zenodo record has been created by this packaging step. No new DOI has been obtained. The generated ZIP is a local candidate package, not evidence of public release.
