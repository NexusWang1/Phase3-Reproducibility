"""Reproduce DVHN400 from the public DroneVehicle test split.

Protocol preserved from the experiment:
- sort testimg/*.jpg lexicographically;
- crop (100, 100, 740, 612), producing 640 x 512 images;
- convert the cropped RGB image to 8-bit grayscale with Pillow and compute its mean;
- retain means < 50.0, yielding 3,260 candidates from 8,980 pairs;
- select numpy.random.default_rng(20261002).permutation(pool)[:400];
- sort the selected source indices before writing paired JPEGs and JSONL.

The source dataset is not bundled. --source-root must contain testimg/ and
testimgr/ from DroneVehicle/test/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CROP = (100, 100, 740, 612)
NIGHT_THRESHOLD = 50.0
SEED = 20261002
SAMPLE_COUNT = 400
EXPECTED_SOURCE_PAIRS = 8980
EXPECTED_NIGHT_POOL = 3260


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brightness(source_images: list[Path], cache: Path | None) -> np.ndarray:
    if cache is not None and cache.is_file():
        means = np.load(cache)
    else:
        means = np.empty(len(source_images), dtype=np.float32)
        for i, path in enumerate(source_images):
            with Image.open(path) as image:
                cropped = image.convert("L").crop(CROP)
                means[i] = np.asarray(cropped, dtype=np.float32).mean()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, means)
    if len(means) != len(source_images):
        raise RuntimeError(
            f"Brightness cache has {len(means)} entries for {len(source_images)} images"
        )
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="DroneVehicle/test/test containing testimg and testimgr")
    parser.add_argument("--out", type=Path, required=True,
                        help="New output directory; existing directories are refused")
    parser.add_argument("--brightness-cache", type=Path,
                        help="Optional .npy cache, loaded if present or created otherwise")
    args = parser.parse_args()

    source_images = sorted((args.source_root / "testimg").glob("*.jpg"))
    if len(source_images) != EXPECTED_SOURCE_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_PAIRS} RGB test images, found {len(source_images)}"
        )
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")

    means = brightness(source_images, args.brightness_cache)
    pool = [i for i, value in enumerate(means) if value < NIGHT_THRESHOLD]
    if len(pool) != EXPECTED_NIGHT_POOL:
        raise RuntimeError(
            f"Night-pool drift: expected {EXPECTED_NIGHT_POOL}, got {len(pool)}"
        )
    rng = np.random.default_rng(SEED)
    selected = sorted(rng.permutation(pool)[:SAMPLE_COUNT].tolist())

    (args.out / "rgb").mkdir(parents=True)
    (args.out / "ir").mkdir()
    rows = []
    for index in selected:
        name = source_images[index].name
        ir_source = args.source_root / "testimgr" / name
        if not ir_source.is_file():
            raise FileNotFoundError(ir_source)
        with Image.open(source_images[index]) as rgb_source, Image.open(ir_source) as ir_image:
            if rgb_source.size != (840, 712) or ir_image.size != (840, 712):
                raise RuntimeError(f"Unexpected source size for {name}")
            rgb_source.crop(CROP).save(args.out / "rgb" / name, quality=95)
            ir_image.crop(CROP).save(args.out / "ir" / name, quality=95)
        rows.append({
            "rgb": f"rgb/{name}",
            "ir_gray": f"ir/{name}",
            "src_index": int(index),
            "rgb_mean_gray": round(float(means[index]), 2),
        })

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    meta = {
        "source": "DroneVehicle test split (8980 pairs)",
        "crop": "100px border -> 640x512",
        "night_rule": f"cropped RGB grayscale mean < {NIGHT_THRESHOLD}",
        "night_pool": len(pool),
        "sampled": len(rows),
        "seed": SEED,
        "brightness_percentiles_selected": np.percentile(
            [row["rgb_mean_gray"] for row in rows], [0, 25, 50, 75, 100]
        ).round(1).tolist(),
    }
    metadata = args.out / "sampling_meta.json"
    metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_pairs": len(source_images),
        "night_pool": len(pool),
        "sampled": len(rows),
        "seed": SEED,
        "manifest_sha256": sha256(manifest),
        "sampling_meta_sha256": sha256(metadata),
    }, indent=2))


if __name__ == "__main__":
    main()
