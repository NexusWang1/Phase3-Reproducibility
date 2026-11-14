"""Recompute Path 2 from veto-only YHT fields using the original lean functions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from flame2_component_lean import component_leans, resultant, THR, MIN_AREA


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fields-dir", type=Path, required=True)
    p.add_argument("--deflection-npy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ign-frame", type=int, default=13771)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    meta_path = a.fields_dir / "meta.json"
    field_root = a.fields_dir / "fields"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = np.load(a.deflection_npy, allow_pickle=False)
    frames = rows[:, 0].astype(int)
    if rows.ndim != 2 or rows.shape[1] != 4 or len(frames) != 1051 or frames[0] != 13651 or frames[-1] != 14701:
        raise RuntimeError("Unexpected veto-only deflection frame range/shape")
    if len(set(frames)) != len(frames) or any(int(frame) != float(raw) for frame, raw in zip(frames, rows[:, 0])):
        raise RuntimeError("Invalid or duplicate deflection frame IDs")
    meta_frames = {int(row["frame_id"]): row["file"] for row in meta["frames"]}
    if set(meta_frames) != set(frames):
        raise RuntimeError("Field metadata and deflection frames differ")

    pre_vec, post_vec, pre_weights, post_weights = [], [], [], []
    pre_count = post_count = 0
    per_frame_r_pre, per_frame_r_post = [], []
    raw_components = []
    valid_components = []
    input_md5 = {str(meta_path): md5(meta_path), str(a.deflection_npy): md5(a.deflection_npy)}
    for frame in frames:
        field_path = field_root / meta_frames[int(frame)]
        if not field_path.is_file():
            raise FileNotFoundError(field_path)
        input_md5[str(field_path)] = md5(field_path)
        s = np.nan_to_num(np.load(field_path, allow_pickle=False).astype(np.float32).squeeze())
        if s.shape != (160, 160):
            raise RuntimeError(f"Unexpected field shape: {field_path}: {s.shape}")
        _, n_raw = label(s > THR, structure=np.ones((3, 3), np.uint8))
        raw_components.append(int(n_raw))
        vecs, powers = component_leans(s)
        valid_components.append(len(vecs))
        if len(vecs) < 3:
            continue  # Original pooling excludes frames with <3 eligible leans.
        _, frame_r = resultant(vecs)
        if frame >= a.ign_frame:
            post_vec.append(vecs)
            post_weights.append(powers)
            post_count += len(vecs)
            per_frame_r_post.append(frame_r)
        else:
            pre_vec.append(vecs)
            pre_weights.append(powers)
            pre_count += len(vecs)
            per_frame_r_pre.append(frame_r)
    mean_raw = float(np.mean(raw_components))
    if not 205 <= mean_raw <= 227:
        raise RuntimeError(f"Acceptance gate failed: raw components/frame={mean_raw:.3f}, required 216±11; no output written")
    if not pre_vec or not post_vec:
        raise RuntimeError("No valid pre/post component pool")
    _, pooled_r = resultant(np.concatenate(post_vec))
    _, area_weighted_r = resultant(np.concatenate(post_vec), np.concatenate(post_weights))
    _, pre_r = resultant(np.concatenate(pre_vec))
    summary = {
        "n_components_post": post_count,
        "pooled_R": pooled_r,
        "area_weighted_R": area_weighted_r,
        "per_frame_median_R": float(np.median(per_frame_r_post)),
        "n_components_pre": pre_count,
        "pre_R": pre_r,
        "mean_components_per_frame": mean_raw,
        "params": {"ign_frame": a.ign_frame, "threshold": THR, "min_area": MIN_AREA,
                   "connectivity": "8-neighbor", "core": "pixels >= per-component 90th percentile intensity, unweighted mean xy",
                   "lean": "intensity-weighted centroid minus core; keep norm > 0.15 px",
                   "pool": "frames with >=3 eligible leans; each component gets one vote",
                   "area_weighted_R_label": "legacy key; original code weights by sum of field intensity (power), not pixel area",
                   "frame_count": len(frames), "raw_component_count": int(sum(raw_components)),
                   "mean_valid_leans_per_frame": float(np.mean(valid_components))},
        "old_vs_new": [
            {"metric": "n_components_post", "old": 20188, "new": post_count},
            {"metric": "pooled_R", "old": 0.12886251624148126, "new": pooled_r},
            {"metric": "area_weighted_R", "old": 0.15679610313505796, "new": area_weighted_r},
            {"metric": "per_frame_median_R", "old": 0.198815022025576, "new": float(np.median(per_frame_r_post))},
            {"metric": "n_components_pre", "old": 1002, "new": pre_count},
            {"metric": "pre_R", "old": 0.22311384161998002, "new": pre_r},
        ],
        "input_md5": input_md5,
        "source_md5": {str(Path(__file__).with_name("flame2_component_lean.py")): md5(Path(__file__).with_name("flame2_component_lean.py"))},
    }
    a.out.mkdir(parents=True)
    output = a.out / "path2_component_lean_qrm_20261003.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "mean_components_per_frame": mean_raw,
                      "n_components_post": post_count, "pooled_R": pooled_r}, ensure_ascii=False))


if __name__ == "__main__":
    main()
