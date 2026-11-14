"""Validate a DVHN400 manifest, optional paired images, and fixed-seed selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20261002


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"Path escapes dataset root: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--brightness-cache", type=Path,
                        help="Optional 8,980-entry test_brightness.npy for exact selection validation")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Do not require the rgb/ and ir/ image files")
    parser.add_argument("--output", type=Path, help="Optional JSON validation report")
    args = parser.parse_args()

    root = args.dataset_root
    manifest = root / "manifest.jsonl"
    metadata_path = root / "sampling_meta.json"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(rows) == 400, f"Expected 400 rows, got {len(rows)}"
    assert len({row["rgb"] for row in rows}) == 400, "Duplicate RGB entries"
    assert len({row["src_index"] for row in rows}) == 400, "Duplicate source indices"
    assert metadata["seed"] == SEED
    assert metadata["night_pool"] == 3260
    assert metadata["sampled"] == 400

    selection_check = "not run; brightness cache not supplied"
    if args.brightness_cache is not None:
        means = np.load(args.brightness_cache)
        assert len(means) == 8980, f"Expected 8,980 brightness entries, got {len(means)}"
        pool = [i for i, value in enumerate(means) if value < 50.0]
        assert len(pool) == 3260, f"Expected 3,260 candidates, got {len(pool)}"
        expected = sorted(np.random.default_rng(SEED).permutation(pool)[:400].tolist())
        assert [row["src_index"] for row in rows] == expected, "Fixed-seed selection mismatch"
        for row in rows:
            assert row["rgb_mean_gray"] == round(float(means[row["src_index"]]), 2)
        selection_check = "passed: all 400 source indices and rounded brightness values match"

    images_check = "skipped by --manifest-only"
    if not args.manifest_only:
        for row in rows:
            rgb = contained(root, row["rgb"])
            infrared = contained(root, row["ir_gray"])
            with Image.open(rgb) as image:
                assert image.size == (640, 512), f"RGB size mismatch: {rgb}"
            with Image.open(infrared) as image:
                assert image.size == (640, 512), f"IR size mismatch: {infrared}"
        images_check = "passed: 400 RGB and 400 IR files exist and are 640 x 512"

    report = {
        "status": "passed",
        "manifest_rows": len(rows),
        "unique_rgb_entries": len({row["rgb"] for row in rows}),
        "unique_source_indices": len({row["src_index"] for row in rows}),
        "fixed_seed": SEED,
        "selection_check": selection_check,
        "images_check": images_check,
        "manifest_sha256": sha256(manifest),
        "sampling_meta_sha256": sha256(metadata_path),
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
