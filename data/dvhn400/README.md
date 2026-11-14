# DVHN400 data files

`manifest.jsonl` is the evaluation input. Each row contains:

- `rgb`: cropped RGB path relative to the DVHN400 root;
- `ir_gray`: cropped infrared path relative to the DVHN400 root;
- `src_index`: zero-based index in the lexicographically sorted 8,980-image DroneVehicle RGB test list;
- `rgb_mean_gray`: cropped 8-bit grayscale mean rounded to two decimals.

`sampling_meta.json` records the frozen crop, night rule, candidate count, sample count, seed, and selected-brightness percentiles.

The images are not bundled. Reconstruct them with `scripts/dvhn400/sample_dvhn400.py` from the official DroneVehicle test split. The manifest alone is sufficient to identify the selected source pairs; it is not sufficient to run detection or depth evaluation without the reconstructed images and depth cache.
