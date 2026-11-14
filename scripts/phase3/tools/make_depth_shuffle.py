"""Preregistered per-cohort derangement for the depth-attribution ablation."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derangement(keys: list[str], rng: random.Random) -> dict[str, str]:
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError("shuffle cohort needs >=2 unique RGB images")
    order = sorted(keys)
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgbt-root", type=Path, required=True)
    ap.add_argument("--m3fd-root", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--test-manifest", type=Path, required=True)
    ap.add_argument("--phase2-package", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.seed != 20261003:
        raise ValueError("Preregistered shuffle seed is 20261003")
    sys.path.insert(0, str(a.phase2_package))
    sys.path.insert(0, str(a.phase2_package.parent / "FieldGuide_Phase2_FieldUse_20260916"))
    from field_use_common import paired_paths

    rgbt = [rgb.resolve().relative_to(a.rgbt_root.resolve()).as_posix()
            for rgb, _, _ in paired_paths(a.rgbt_root, "val")]
    val, test = rows(a.val_manifest), rows(a.test_manifest)
    if len(rgbt) != 905 or len(val) != 402 or len(test) != 593:
        raise RuntimeError(f"Expected RGBT val=905, M3FD val=402, test=593; got {len(rgbt)}/{len(val)}/{len(test)}")
    if any(r.get("split") != "val" or r.get("scene_block") != 1 for r in val):
        raise RuntimeError("M3FD val must be scene block 1")
    if any(r.get("split") != "test" or r.get("scene_block") != 0 or r.get("smoke_fire_negative") not in (1, "1", True)
           for r in test):
        raise RuntimeError("M3FD test must be negative scene block 0")
    if {r["rgb"] for r in val} & {r["rgb"] for r in test}:
        raise RuntimeError("M3FD val/test RGB overlap")
    cohorts = {"rgbt_val": rgbt, "m3fd_test": [r["rgb"] for r in test]}
    roots = {"rgbt_val": (a.rgbt_root, a.rgbt_root / "depth_cache"),
             "m3fd_test": (a.m3fd_root, a.m3fd_root / "depth_cache")}
    for cohort, keys in cohorts.items():
        root, cache = roots[cohort]
        for key in keys:
            rgb = (root / key).resolve()
            if not rgb.is_relative_to(root.resolve()):
                raise RuntimeError(f"Path escapes dataset root: {key}")
            depth = (cache / key).with_suffix(".png")
            if not rgb.is_file() or not depth.is_file():
                raise FileNotFoundError((rgb, depth))
    rng = random.Random(a.seed)
    mappings = {name: derangement(keys, rng) for name, keys in cohorts.items()}
    report = {"seed": a.seed, "method": "random cyclic derangement within each frozen cohort",
              "counts": {name: len(keys) for name, keys in cohorts.items()},
              "val_manifest_sha256": hashlib.sha256(a.val_manifest.read_bytes()).hexdigest(),
              "test_manifest_sha256": hashlib.sha256(a.test_manifest.read_bytes()).hexdigest(),
              "cohorts": mappings}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"seed": a.seed, "counts": report["counts"], "out": str(a.out)}))


if __name__ == "__main__":
    main()
