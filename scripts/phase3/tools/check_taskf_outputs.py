"""Read-only Task-F regression and output-accounting check after server runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-gate", type=Path, required=True)
    p.add_argument("--eval-a", type=Path, required=True)
    p.add_argument("--eval-b", type=Path, required=True)
    p.add_argument("--demo", type=Path, required=True)
    p.add_argument("--bitcheck-log", type=Path, required=True)
    a = p.parse_args()
    bitcheck = a.bitcheck_log.read_text(encoding="utf-8")
    if "ddm_off_bitcheck=PASS" not in bitcheck or "real_model_bitcheck=PASS" not in bitcheck:
        raise RuntimeError("Both FRP-off bitchecks are required")
    old = {group["group"]: group["metrics"] for group in read(a.old_gate)["groups"]}
    groups = {}
    for name, folder, fp_reference in (("A", a.eval_a, 155), ("B", a.eval_b, 71)):
        metrics = read(folder / "metrics.json")
        records = [json.loads(line) for line in (folder / "detections3d.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != metrics["n_detections3d"]:
            raise RuntimeError(f"{name}: JSONL rows != n_detections3d")
        if any(row["dataset"] not in {"rgbt_val", "m3fd_hardneg"} or row["scale_note"] != "nominal, PHY_F3D_SCALE=50, not metric" for row in records):
            raise RuntimeError(f"{name}: dataset or nominal-scale label invalid")
        delta = abs(metrics["mAP50"] - old[name]["mAP50"])
        fp_delta = abs(metrics["hardneg_fp"] - fp_reference)
        if delta > .001 or fp_delta > 1:
            raise RuntimeError(f"{name}: regression failed, delta_mAP50={delta}, delta_FP={fp_delta}")
        groups[name] = {"map50": metrics["mAP50"], "delta_map50": delta,
                        "hardneg_fp": metrics["hardneg_fp"], "delta_hardneg_fp": fp_delta,
                        "n_detections3d": len(records),
                        "z_median_rgbt": metrics["z_median_rgbt"],
                        "z_median_hardneg": metrics["z_median_hardneg"]}
    demo = read(a.demo / "demo_meta.json")
    frames = demo["frames"]
    if not frames or len(frames) != demo["frames_saved"]:
        raise RuntimeError("Demo frame count invalid")
    if any("boxes3d" not in frame or "c_dir" not in frame for frame in frames):
        raise RuntimeError("Demo frame missing boxes3d/c_dir")
    if any(len(frame["boxes3d"]) != frame["kept_count"] for frame in frames):
        raise RuntimeError("Demo boxes3d count differs from kept detections")
    if any(not 0 <= frame["c_dir"] <= 1 for frame in frames):
        raise RuntimeError("Demo c_dir outside [0,1]")
    comparison = demo["c_dir_comparison"]
    if comparison["pre_fire_n"] == 0 or comparison["after_ignition_n"] == 0:
        raise RuntimeError("Demo lacks one side of the ignition comparison")
    output = {"bitcheck": "PASS", "groups": groups, "demo_frames": len(frames),
              "c_dir_comparison": comparison}
    print("[TASKF-CHECK] " + json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
