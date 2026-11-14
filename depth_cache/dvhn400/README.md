# DVHN400 depth cache

This directory now contains the server-generated DVHN400 depth cache:

- `rgb/` contains exactly 400 uint16 PNGs;
- `depth_cache_meta.json` records `depth-anything/Depth-Anything-V2-Large`, 400 created images, and the storage convention;
- every PNG filename matches one RGB entry in `data/dvhn400/manifest.jsonl`;
- every PNG is 640 × 512 pixels, has uint16 depth values, and contains nonconstant depth;
- the complete cache is 151,014,392 bytes before ZIP compression.

Evaluation reads `uint16 / 65535`; normalized value 1 means far. The cache was generated from the cropped RGB images with frozen Depth Anything V2 Large weights. The depth PNG bytes are unchanged from the supplied server copy. In `depth_cache_meta.json`, only the six visual-QA path strings were converted from server-absolute paths to DVHN400-relative paths.

The original cache layout on the evaluation server was:

```text
DVHN400/
  rgb/*.jpg
  ir/*.jpg
  manifest.jsonl
  depth_cache/
    rgb/*.png
    depth_cache_meta.json
```

To regenerate rather than reuse the cache, run:

```bash
python scripts/phase3/tools/build_depth_cache.py \
  --dataset-root /path/to/DVHN400 \
  --images /path/to/DVHN400/rgb \
  --device cuda:0 --batch 8
```

`validation/artifact_validation.json` records the actual cache checks. These checks establish file integrity, pairing, dimensions, and depth format; they are not a rerun of GPU depth inference.

Large cache files are excluded from code-only GitHub staging by `.gitignore`. The full Zenodo candidate ZIP includes all 400 PNGs and their SHA-256 entries.
