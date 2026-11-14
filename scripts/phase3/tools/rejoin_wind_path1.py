"""Rejoin *stored* plume drifts with veto-only 3-D deflection by frame ID.

The final v3 array has dt=15/30/60 only. Optional five-column arrays
(`frame,A_dx,A_dy,B_dx,B_dy`) supply previously stored dt=1/5 drifts.
No drift vectors are estimated or regenerated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed(array: np.ndarray, cols: int, source: Path) -> dict[int, np.ndarray]:
    if array.ndim != 2 or array.shape[1] != cols:
        raise ValueError(f"{source}: expected N×{cols}, got {array.shape}")
    frames = array[:, 0].astype(np.int64)
    if len(set(frames)) != len(frames) or not np.array_equal(array[:, 0], frames):
        raise ValueError(f"{source}: duplicate/noninteger frame IDs")
    return dict(zip(map(int, frames), array))


def angle(vector: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vector[:, 0], -vector[:, 1]))


def segment(frame_ids: np.ndarray, drift: np.ndarray, direction: np.ndarray, ignition: int, post: bool) -> dict:
    norm_drift = np.linalg.norm(drift, axis=1)
    norm_direction = np.linalg.norm(direction, axis=1)
    valid = np.isfinite(drift).all(axis=1) & np.isfinite(direction).all(axis=1)
    valid &= (norm_drift > 1e-9) & (norm_direction > 1e-9)
    valid &= (frame_ids >= ignition) if post else (frame_ids < ignition)
    d, v = drift[valid], direction[valid]
    if len(d) < 3:
        return {"n_pairs": int(len(d)), "R": None, "median_angle_diff_deg": None,
                "deflection_R": None}
    unit_d = d / np.linalg.norm(d, axis=1, keepdims=True)
    unit_v = v / np.linalg.norm(v, axis=1, keepdims=True)
    delta = (angle(d) - angle(v) + 180) % 360 - 180
    return {"n_pairs": int(len(d)), "R": float(np.linalg.norm(unit_d.mean(axis=0))),
            "median_angle_diff_deg": float(np.median(np.abs(delta))),
            "deflection_R": float(np.linalg.norm(unit_v.mean(axis=0)))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drift-v3", type=Path, required=True)
    p.add_argument("--deflection-npy", type=Path, required=True)
    p.add_argument("--dt1-pairs", type=Path, help="Stored N×5 [frame,A_dx,A_dy,B_dx,B_dy]; no recomputation")
    p.add_argument("--dt5-pairs", type=Path, help="Stored N×5 [frame,A_dx,A_dy,B_dx,B_dy]; no recomputation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ign-frame", type=int, default=13771)
    p.add_argument("--allow-partial", action="store_true", help="Explicitly publish only available dt; marked incomplete")
    a = p.parse_args()
    if a.out.exists() and not a.out.is_dir():
        raise FileExistsError(a.out)
    output = a.out / "path1_drift_rejoin_qrm_20261003.json"
    if output.exists():
        raise FileExistsError(output)
    v3 = keyed(np.load(a.drift_v3, allow_pickle=False), 18, a.drift_v3)
    new = keyed(np.load(a.deflection_npy, allow_pickle=False), 4, a.deflection_npy)
    common = sorted(set(v3) & set(new))
    if len(common) < 800:
        raise RuntimeError(f"Frame-ID intersection {len(common)} < 800; refusing output")
    inputs = {str(a.drift_v3): md5(a.drift_v3), str(a.deflection_npy): md5(a.deflection_npy)}
    source = {15: v3, 30: v3, 60: v3}
    layouts = {15: {"A": (1, 2), "B": (3, 4)},
               30: {"A": (5, 6), "B": (7, 8)},
               60: {"A": (9, 10), "B": (11, 12)}}
    for dt, path in ((1, a.dt1_pairs), (5, a.dt5_pairs)):
        if path is not None:
            source[dt] = keyed(np.load(path, allow_pickle=False), 5, path)
            layouts[dt] = {"A": (1, 2), "B": (3, 4)}
            inputs[str(path)] = md5(path)
    missing = [dt for dt in (1, 5) if dt not in source]
    if missing and not a.allow_partial:
        raise RuntimeError(f"Final v3 array stores dt=15/30/60 only; stored A+B arrays for dt={missing} required. "
                           "No output written. Use --allow-partial only with explicit approval.")

    results = []
    for dt in sorted(source):
        joined = sorted(set(source[dt]) & set(new))
        if len(joined) < 800:
            raise RuntimeError(f"dt={dt} frame-ID intersection {len(joined)} < 800; refusing output")
        ids = np.asarray(joined)
        direction = np.asarray([[new[frame][1], new[frame][3]] for frame in joined], dtype=float)
        for estimator, (col_x, col_y) in layouts[dt].items():
            drift = np.asarray([[source[dt][frame][col_x], source[dt][frame][col_y]] for frame in joined], dtype=float)
            post = segment(ids, drift, direction, a.ign_frame, True)
            pre = segment(ids, drift, direction, a.ign_frame, False)
            results.append({"estimator": "mask-difference centroid" if estimator == "A" else "phase correlation",
                            "dt": dt, "intersected_frames": len(joined),
                            "R": post["R"], "median_angle_diff_deg": post["median_angle_diff_deg"],
                            "post": post, "pre_ignition_control": pre})
    report = {"status": "partial_missing_dt1_dt5" if missing else "complete",
              "v3_deflection_frame_intersection": len(common), "ign_frame": a.ign_frame,
              "results": results,
              "missing_dt": missing,
              "old_vs_new_reference": {"old_post_drift_R_upper": 0.055, "old_median_angle_diff_approx_deg": 90,
                                       "old_pre_ignition_platform_drift_R_approx": 0.75,
                                       "note": "Use exact archived v3 summary for exact old per-dt comparisons; do not promote approximate references to exact measurements."},
              "input_md5": inputs,
              "join": "integer frame-ID intersection; no positional array alignment; new 3-D angle atan2(dX,-dZ) in image convention",
              "drift_policy": "stored vectors reused byte-for-byte; no plume/drift recomputation"}
    a.out.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "intersected_frames": len(common)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
