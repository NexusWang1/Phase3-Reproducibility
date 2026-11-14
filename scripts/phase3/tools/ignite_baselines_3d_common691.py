"""Paired IGNITE 3-D/2-D baselines on the same 691 v4 frame pairs.

Reuses the original 2-D baseline operators and v4 thermal/depth pipeline.
The v4 3-D arm is X-Z ground-plane deflection; 2-D baselines are embedded
in full X-Y-Z as (dx, dy, 0). No detector or GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ignite_direction_baselines import binary_dilation, hier_boot, positive, unit
from tools.ignite_direct_v4 import (CX, CY, deflection_vec_2d, deflection_vec_3d,
                                    field_from_tiff, load_depth, load_mask, load_tiff,
                                    backproject_field)


EXPECTED_COUNTS = {"0001": 177, "0002": 171, "0003": 129, "0004": 214}
EXPECTED_3D_POS = 389
EXPECTED_2D_POS = 413


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchor(v4_dir: Path) -> tuple[dict, dict]:
    summary_path = v4_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary) != set(EXPECTED_COUNTS):
        raise RuntimeError(f"Unexpected v4 sequences: {sorted(summary)}")
    common = {}
    counts = {}
    positives3 = positives2 = 0
    digests = {str(summary_path): md5(summary_path)}
    for sid in sorted(summary):
        path = v4_dir / f"{sid}_direct_v4.npy"
        array = np.load(path, allow_pickle=False)
        columns = summary[sid]["columns"]
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise RuntimeError(f"v4 shape/column mismatch: {sid}")
        i3, i2 = columns.index("raw_static_3d_med"), columns.index("raw_static_2d_med")
        indices = np.flatnonzero(np.isfinite(array[:, i3]) & np.isfinite(array[:, i2]))
        counts[sid] = int(len(indices))
        common[sid] = (array, indices, i3, i2)
        positives3 += int((array[indices, i3] > 0).sum())
        positives2 += int((array[indices, i2] > 0).sum())
        digests[str(path)] = md5(path)
    if counts != EXPECTED_COUNTS or positives3 != EXPECTED_3D_POS or positives2 != EXPECTED_2D_POS:
        raise RuntimeError(f"Anchor failed: counts={counts}, 3d={positives3}, 2d={positives2}; refusing baselines")
    return common, {"common_pairs": sum(counts.values()), "per_seq_pairs": counts,
                    "3d_pos": positives3 / sum(counts.values()),
                    "2d_pos_on_common": positives2 / sum(counts.values()),
                    "3d_positive_count": positives3, "2d_positive_count": positives2,
                    "input_md5": digests}


def _mean_ci(rows: list[dict], key: str, reps: int, seed: int) -> dict:
    subset = [row for row in rows if row.get(key) is not None]
    if not subset:
        raise RuntimeError(f"No paired rows for {key}")
    return {"n": len(subset), "rate": float(np.mean([row[key] for row in subset])),
            "ci": hier_boot(subset, key, reps, seed)}


def _paired(rows: list[dict], truth: str, baseline: str, reps: int, seed: int) -> dict:
    subset = [row for row in rows if row.get(baseline) is not None and row.get(truth) is not None]
    if not subset:
        raise RuntimeError(f"No paired rows: {truth} vs {baseline}")
    paired_rows = [{"sequence": row["sequence"], "block": row["block"],
                    "baseline": row[baseline], "difference": row[truth] - row[baseline]}
                   for row in subset]
    return {"n": len(subset), "rate": float(np.mean([r["baseline"] for r in paired_rows])),
            "paired_deflection_rate": float(np.mean([r[truth] for r in subset])),
            "paired_diff_pp": 100 * float(np.mean([r["difference"] for r in paired_rows])),
            "ci_pp": [100 * value for value in hier_boot(paired_rows, "difference", reps, seed)]}


def _points_xyz(xs: np.ndarray, ys: np.ndarray, depth: np.ndarray, scale: float) -> np.ndarray:
    z = depth[ys, xs] * scale
    return np.column_stack(((xs - CX) * z / 640., (ys - CY) * z / 640., z))


def _row_to_frame(array: np.ndarray, items: list[dict], sid: str, delta: int) -> dict[int, int]:
    times = np.asarray([float(item["video_time_s"]) for item in items], dtype=float)
    mapping = {}
    for row_index, time_s in enumerate(array[:, 0]):
        k = int(np.argmin(np.abs(times - time_s)))
        if abs(times[k] - time_s) > 1e-7 or k >= len(items) - delta or k in mapping:
            raise RuntimeError(f"v4 time/frame pairing ambiguous: {sid}, row={row_index}, t={time_s}")
        mapping[k] = row_index
    return mapping


def _merge_diagnostic(report: dict, path: Path) -> dict:
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "diagnostic_2d_only_missing_depth" or diagnostic.get("baselines_vs_3d_truth") is not None:
        raise RuntimeError("Diagnostic JSON is not the approved 2-D-only artifact")
    for key in ("common_pairs", "per_seq_pairs", "anchor_check", "arms", "baselines_vs_2d_truth"):
        if report[key] != diagnostic.get(key):
            raise RuntimeError(f"Server run differs from approved local diagnostic in {key}; refusing merged output")
        report[key] = diagnostic[key]
    report["merge_provenance"] = {"diagnostic_json": str(path), "diagnostic_md5": md5(path),
                                  "rule": "2-D arms and baselines checked equal and carried unchanged; only 3-D baselines added"}
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ignite-root", type=Path, required=True)
    p.add_argument("--v4-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="JSON file; refuses overwrite")
    p.add_argument("--delta", type=int, default=10)
    p.add_argument("--t80", type=float, default=8800.)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--scale", type=float, default=50.)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--2d-only", dest="two_d_only", action="store_true", help="Diagnostic when 3-D depth cache is unavailable; not final 3-D delivery")
    p.add_argument("--diagnostic-json", type=Path,
                   help="Existing 2-D common-691 diagnostic; require exact numerical agreement before completing 3-D")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if not a.two_d_only and a.diagnostic_json is None:
        raise ValueError("Full 3-D delivery requires --diagnostic-json for exact 2-D preservation check")
    if (a.delta, a.t80, a.min_area, a.scale) != (10, 8800., 50, 50.):
        raise ValueError("This anchored comparison requires original v4 defaults")
    common, anchor = _anchor(a.v4_dir)
    print(f"[IGNITE-ANCHOR] {anchor['per_seq_pairs']} 3d={anchor['3d_positive_count']} 2d={anchor['2d_positive_count']}", flush=True)

    records = [json.loads(line) for line in (a.ignite_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in records:
        groups[row["sequence_id"]].append(row)
    if set(groups) != set(EXPECTED_COUNTS):
        raise RuntimeError("IGNITE metadata sequence set differs from v4")
    for rows in groups.values():
        rows.sort(key=lambda row: row["video_time_s"])
    # A current/future depth pair is indispensable for the X-Y-Z truth. Fail before work.
    common_row_by_k = {}
    for sid, (array, indices, _, _) in common.items():
        mapping = _row_to_frame(array, groups[sid], sid, a.delta)
        common_row_by_k[sid] = {k: mapping[k] for k in mapping if mapping[k] in set(indices)}
        if len(common_row_by_k[sid]) != len(indices):
            raise RuntimeError(f"Common pair time mapping incomplete: {sid}")
        if a.two_d_only:
            continue
        for k in common_row_by_k[sid]:
            for j in (k, k + a.delta):
                row = groups[sid][j]
                path = a.ignite_root / "rgb_only" / "depth_cache" / sid / (row["aligned_id"] + ".png")
                if not path.is_file():
                    raise FileNotFoundError(f"3-D truth requires existing depth cache: {path}")

    rng = np.random.default_rng(a.seed)
    common_rows = []
    calib_rows = []
    for sid in sorted(groups):
        items = groups[sid]
        array, indices, col3, col2 = common[sid]
        common_set = set(common_row_by_k[sid])

        @lru_cache(maxsize=2 * a.delta + 4)
        def thermal(index: int) -> np.ndarray:
            return load_tiff(a.ignite_root / items[index]["thermal_tiff_file_name"])

        @lru_cache(maxsize=2 * a.delta + 4)
        def mask(index: int) -> np.ndarray:
            return load_mask(a.ignite_root / items[index]["mask_80c_file_name"])

        @lru_cache(maxsize=2 * a.delta + 4)
        def depth(index: int) -> np.ndarray:
            item = items[index]
            return load_depth(a.ignite_root / "rgb_only" / "depth_cache" / sid / (item["aligned_id"] + ".png"))

        for k in range(len(items) - a.delta):
            is_common = k in common_set
            if not is_common and k < a.delta:
                continue
            current = mask(k)
            if current.sum() < a.min_area:
                if is_common:
                    raise RuntimeError(f"v4 common pair failed current-mask gate: {sid}:{k}")
                continue
            grown = binary_dilation(current, iterations=3)
            newly = mask(k + a.delta) & ~grown
            if newly.sum() < a.min_area:
                if is_common:
                    raise RuntimeError(f"v4 common pair failed new-fire gate: {sid}:{k}")
                continue
            cy, cx = np.where(current)
            ny, nx = np.where(newly)
            points2 = np.column_stack((nx - cx.mean(), ny - cy.mean()))
            field = field_from_tiff(thermal(k), a.t80)
            prediction2 = deflection_vec_2d(field)
            if unit(prediction2) is None:
                if is_common:
                    raise RuntimeError(f"v4 common pair failed 2-D predictor: {sid}:{k}")
                continue
            time_s = float(items[k]["video_time_s"])
            block = int(time_s // 60)
            record = {"sequence": sid, "frame_index": k, "block": block,
                      "pred_positive": positive(points2, prediction2)}
            if k >= a.delta:
                calib_rows.append(record)
            if not is_common:
                continue
            row_index = common_row_by_k[sid][k]
            if not np.isclose(array[row_index, 0], time_s, rtol=0, atol=1e-7):
                raise RuntimeError(f"v4 time/frame pairing mismatch: {sid}:{k}")
            if record["pred_positive"] != int(array[row_index, col2] > 0):
                raise RuntimeError(f"2-D projection differs from v4 anchor: {sid}:{k}")
            points3 = None
            if not a.two_d_only:
                points3 = _points_xyz(nx, ny, depth(k + a.delta), a.scale)
                center3 = _points_xyz(cx, cy, depth(k), a.scale).mean(axis=0)
                points3 -= center3
                prediction_xz = deflection_vec_3d(backproject_field(depth(k), field,
                                          fx=640., fy=640., cx=CX, cy=CY,
                                          scale=a.scale, threshold=.05, topk=20000))
                if prediction_xz is None:
                    raise RuntimeError(f"v4 common pair missing 3-D predictor: {sid}:{k}")
                prediction3 = np.array([prediction_xz[0], 0., prediction_xz[1]])
                if positive(points3, prediction3) != int(array[row_index, col3] > 0):
                    raise RuntimeError(f"X-Y-Z reconstruction differs from v4 3-D anchor: {sid}:{k}")

            # Reuse the original baseline formulas and exclusions verbatim.
            hot_index = np.argmax(np.where(current, thermal(k), -np.inf))
            hy, hx = np.unravel_index(hot_index, current.shape)
            hot = np.array([hx - cx.mean(), hy - cy.mean()], dtype=float)
            gy, gx = np.gradient(thermal(k))
            weights = np.clip(thermal(k) - a.t80, 0, None) * current
            grad = np.array([(gx * weights).sum(), (gy * weights).sum()], dtype=float)
            prev = None
            if k >= a.delta:
                prior = mask(k - a.delta)
                py, px = np.where(prior)
                prev_new = current & ~binary_dilation(prior, iterations=3)
                vy, vx = np.where(prev_new)
                if len(px) and len(vx) >= a.min_area:
                    prev = np.array([vx.mean() - px.mean(), vy.mean() - py.mean()])
            theta = rng.uniform(0, 2 * np.pi, size=100)
            random2 = []
            random3 = []
            for angle in theta:
                direction = np.array([np.cos(angle), np.sin(angle)])
                random2.append(positive(points2, direction))
                if points3 is not None:
                    random3.append(positive(points3, np.r_[direction, 0.]))
            result = {"sequence": sid, "frame_index": k, "block": block,
                      "deflection_3d": int(array[row_index, col3] > 0),
                      "deflection_2d": int(array[row_index, col2] > 0),
                      "random_2d": float(np.mean(random2))}
            if points3 is not None:
                result["random_3d"] = float(np.mean(random3))
            for name, direction in (("hottest", hot), ("gradient", grad), ("prev_displacement", prev)):
                result[name + "_2d"] = positive(points2, direction)
                if points3 is not None:
                    result[name + "_3d"] = positive(points3, np.r_[direction, 0.]) if direction is not None else None
            common_rows.append(result)
        print(f"[IGNITE-COMMON] {sid}: {sum(r['sequence'] == sid for r in common_rows)}", flush=True)

    if len(common_rows) != 691 or len(calib_rows) != 921:
        raise RuntimeError(f"Pair counts changed: common={len(common_rows)}, 2D921={len(calib_rows)}")
    if abs(np.mean([r["pred_positive"] for r in calib_rows]) - 581 / 921) > 1e-12:
        raise RuntimeError("2D@921 positive-rate calibration failed")
    calibration = hier_boot(calib_rows, "pred_positive", a.bootstrap, a.seed)
    if max(abs(calibration[0] - .568), abs(calibration[1] - .693)) > .005:
        raise RuntimeError(f"2D@921 bootstrap calibration failed: {calibration}")
    anchor["boot_calib_2d921_ci"] = calibration
    arms = {"3d_deflection": _mean_ci(common_rows, "deflection_3d", a.bootstrap, a.seed + 1),
            "2d_deflection": _mean_ci(common_rows, "deflection_2d", a.bootstrap, a.seed + 2)}
    baselines = {}
    truths = ("2d",) if a.two_d_only else ("3d", "2d")
    for truth in truths:
        base = {}
        for idx, name in enumerate(("random", "hottest", "gradient", "prev_displacement")):
            base[name] = _paired(common_rows, "deflection_" + truth, name + "_" + truth,
                                 a.bootstrap, a.seed + 10 * (truth == "2d") + idx + 3)
        baselines[truth] = base
    report = {"common_pairs": 691, "per_seq_pairs": anchor["per_seq_pairs"],
              "anchor_check": {key: anchor[key] for key in ("3d_pos", "2d_pos_on_common", "boot_calib_2d921_ci")},
              "arms": arms, "baselines_vs_3d_truth": baselines.get("3d"),
              "baselines_vs_2d_truth": baselines["2d"],
              "status": "diagnostic_2d_only_missing_depth" if a.two_d_only else "complete",
              "protocol": {"pair_key": "sequence ID + sorted metadata index k; v4 time checked exactly",
                           "truth_3d": "full X-Y-Z new-fire displacement relative to current fire centroid; v4 3D arm uses (dX,0,dZ)",
                           "baseline_embedding": "2D image-plane (dx,dy) embedded (dx,dy,0)",
                           "positive": "median signed projection > 0", "random_directions_per_pair": 100,
                           "bootstrap": "sequence then absolute-time 60 s block, paired, with replacement",
                           "bootstrap_reps": a.bootstrap, "seed": a.seed,
                           "delta": a.delta, "t80": a.t80, "min_area": a.min_area, "scale": a.scale},
              "input_md5": anchor["input_md5"] | {str(a.ignite_root / "metadata.jsonl"): md5(a.ignite_root / "metadata.jsonl")}}
    if a.diagnostic_json is not None:
        if a.two_d_only:
            raise ValueError("--diagnostic-json is for the full 3-D run, not --2d-only")
        report = _merge_diagnostic(report, a.diagnostic_json)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("[IGNITE-COMMON] " + json.dumps({"out": str(a.out), "common_pairs": 691,
                                            "3d_rate": arms["3d_deflection"]["rate"],
                                            "2d_rate": arms["2d_deflection"]["rate"]}), flush=True)


if __name__ == "__main__":
    main()
