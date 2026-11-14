"""Audit YHT field identity; recompute 3-D deflection only when fields differ.

Run on the server after run_field_sequence.py. Existing results are never replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _fig_provenance(fig_dir: Path) -> dict:
    if not fig_dir.is_dir():
        raise FileNotFoundError(fig_dir)
    files = sorted(p for p in fig_dir.glob("*") if p.is_file() and p.name.startswith(("fig3d_", "fig3view_")) and p.suffix.lower() in {".png", ".pdf"})
    records = [{"file": p.name, "mtime_local": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
               for p in files]
    target = [p for p in files if "013787" in p.name]
    # The task's date rule applies to actual server originals, not transferred copies.
    if len(target) != 4:
        verdict = "unknown_incomplete_frame_013787"
    elif all(datetime.fromtimestamp(p.stat().st_mtime).date().isoformat() == "2026-09-23" for p in target):
        verdict = "pre_hook_frp_no_redraw_needed"
    elif all(datetime.fromtimestamp(p.stat().st_mtime).date().isoformat() > "2026-09-27" for p in target):
        verdict = "q_era_redraw_required_if_fields_differ"
    else:
        verdict = "mixed_or_ambiguous_redraw_recommended_if_fields_differ"
    return {"directory": str(fig_dir), "files": records, "frame_013787_verdict": verdict}


def _compare(new_dir: Path, old_dir: Path) -> dict:
    new_meta = json.loads((new_dir / "meta.json").read_text(encoding="utf-8"))
    old_meta = json.loads((old_dir / "meta.json").read_text(encoding="utf-8"))
    keys = ("start", "end", "stride", "requested_frames")
    if any(new_meta.get(k) != old_meta.get(k) for k in keys):
        raise RuntimeError("YHT window or stride differs from baseline")
    expected = {str(row["file"]) for row in old_meta["frames"]}
    new_files = {p.name for p in (new_dir / "fields").glob("*.npy")}
    old_files = {p.name for p in (old_dir / "fields").glob("*.npy")}
    missing_new = sorted(expected - new_files)
    missing_old = sorted(expected - old_files)
    extra_new = sorted(new_files - expected)
    rows = []
    mismatch = 0
    global_max = 0.0
    nonfinite = []
    shape_mismatch = []
    dtype_mismatch = []
    for name in sorted(expected & new_files & old_files):
        a = np.load(new_dir / "fields" / name, allow_pickle=False)
        b = np.load(old_dir / "fields" / name, allow_pickle=False)
        if a.shape != b.shape:
            shape_mismatch.append(name)
            continue
        if a.dtype != b.dtype:
            dtype_mismatch.append(name)
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            nonfinite.append(name)
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        maximum = float(delta.max(initial=0))
        equal = bool(a.tobytes(order="C") == b.tobytes(order="C"))
        mismatch += not equal
        global_max = max(global_max, maximum)
        rows.append({"frame": name, "bit_equal": equal, "max_abs_diff": maximum})
    complete = not (missing_new or missing_old or extra_new or nonfinite or shape_mismatch or dtype_mismatch) and len(rows) == len(expected)
    return {"baseline_dir": str(old_dir), "new_dir": str(new_dir), "expected_frames": len(expected),
            "compared_frames": len(rows), "mismatched_frames": mismatch, "global_max_abs_diff": global_max,
            "missing_new": missing_new, "missing_baseline": missing_old, "extra_new": extra_new,
            "nonfinite": nonfinite, "shape_mismatch": shape_mismatch, "dtype_mismatch": dtype_mismatch,
            "status": "bit_equal" if complete and mismatch == 0 else "different" if complete else "incomplete",
            "per_frame": rows}


def _recompute(new_dir: Path, rgb_root: Path, depth_cache: Path, ignition: int) -> dict:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    package = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package))
    from tools.ddm_geometry import depth_letterboxed, letterbox_transform
    from ultralytics.models.yolo.detect.field3d import backproject_field

    meta = json.loads((new_dir / "meta.json").read_text(encoding="utf-8"))
    image_files = {int(p.stem.split("_")[-1]): p for p in rgb_root.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and p.stem.split("_")[-1].isdigit()}
    out = new_dir / "deflection_series.npy"
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for item in meta["frames"]:
        frame = int(item["frame_id"])
        rgb = image_files.get(frame)
        if rgb is None:
            raise FileNotFoundError(f"RGB frame {frame} missing")
        depth_path = (depth_cache / rgb.relative_to(rgb_root)).with_suffix(".png")
        with Image.open(rgb) as im:
            transform = letterbox_transform(*im.size)
        depth = depth_letterboxed(depth_path, transform, "cpu")[0, 0].numpy()
        field = np.load(new_dir / "fields" / item["file"], allow_pickle=False)
        s = F.interpolate(torch.from_numpy(field)[None, None], (640, 640), mode="bilinear", align_corners=False)[0, 0].numpy()
        points = backproject_field(depth, s, threshold=.05, topk=10000, scale=50)
        if len(points) < 4:
            vec = np.zeros(3, dtype=np.float64)
        else:
            strength = points[:, 3]
            center = np.average(points[:, :3], axis=0, weights=strength)
            core = points[strength >= np.percentile(strength, 90), :3].mean(axis=0)
            vec = center - core
        rows.append((float(frame), *map(float, vec)))
    series = np.asarray(rows, dtype=np.float64)
    if len(series) != len(meta["frames"]):
        raise RuntimeError("Deflection row count does not match field frames")
    np.save(out, series)
    post = series[:, 0] >= ignition
    if not post.any():
        raise RuntimeError("No post-ignition frames")
    angles = np.degrees(np.arctan2(series[post, 1], series[post, 3]))
    start = int(np.argmax(post)) // 200 * 200
    segments = []
    for lo in range(start, len(series), 200):
        seg = series[lo:min(lo + 200, len(series))]
        norm = np.hypot(seg[:, 1], seg[:, 3])
        valid = norm > 1e-6
        if valid.sum() < 5:
            continue
        ux, uz = seg[valid, 1] / norm[valid], seg[valid, 3] / norm[valid]
        segments.append({"first_frame": int(seg[0, 0]), "last_frame": int(seg[-1, 0]),
                         "R": float(np.hypot(ux.mean(), uz.mean())),
                         "angle_deg": float(np.degrees(np.arctan2(ux.mean(), uz.mean())))})
    frame13787 = series[series[:, 0] == 13787]
    return {"deflection_series": str(out), "math": "paper_fig_3d._vector_3d: bilinear 640, threshold .05, topk 10000, SCALE 50, weighted center minus top-10% strength core; extended identically to Y",
            "post_ignition_median_angle_deg": float(np.median(angles)),
            "post_ignition_mean_dxyz": series[post, 1:4].mean(axis=0).tolist(),
            "segments_200_row_resultant": segments,
            "frame_013787_angle_deg": float(np.degrees(np.arctan2(frame13787[0, 1], frame13787[0, 3]))) if len(frame13787) else None,
            "c_dir": "requires detection-aware demo rerun; cannot derive from fields alone"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True, help="Original server paper_fig_v4, not transferred copies")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--ignition", type=int, default=13771)
    parser.add_argument("--recompute-only", action="store_true", help="Resume numerical recomputation after an existing diff report")
    args = parser.parse_args()
    report_path = args.new_dir / "field_diff_report.json"
    if args.recompute_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        comparison, provenance = report["comparison"], report["figure_provenance"]
    else:
        if report_path.exists():
            raise FileExistsError(report_path)
        comparison = _compare(args.new_dir, args.baseline_dir)
        provenance = _fig_provenance(args.fig_dir)
        report = {"comparison": comparison, "figure_provenance": provenance}
        _write_new(report_path, report)
    if comparison["status"] == "different":
        summary_path = args.new_dir / "deflection_summary.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        _write_new(summary_path, _recompute(args.new_dir, args.rgb_root, args.depth_cache, args.ignition))
    print(json.dumps({"status": comparison["status"], "mismatched_frames": comparison["mismatched_frames"],
                      "global_max_abs_diff": comparison["global_max_abs_diff"],
                      "figure_verdict": provenance["frame_013787_verdict"]}, ensure_ascii=False))
    if comparison["status"] == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
