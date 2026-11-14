"""Bit-gated q-fusion-free mainline retest on cuda:2, frozen models only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

THETAS = ("0.001", "0.005")
EXPECTED = {"m3fd": {"0.001": 54, "0.005": 50},
            "dvhn400": {"0.001": 388, "0.005": 327}}


def records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preflight(a):
    if a.device != "cuda:2":
        raise RuntimeError("Only cuda:2 is authorized for this task")
    if a.out.exists():
        raise FileExistsError(f"Refusing overwrite: {a.out}")
    for path in (a.weights, a.ddm_ckpt, a.rgbt_root, a.m3fd_root, a.m3fd_manifest,
                 a.dvhn_root, a.dvhn_manifest, a.phase2_package, a.equivalence):
        if not path.exists():
            raise FileNotFoundError(path)
    eq = json.loads(a.equivalence.read_text(encoding="utf-8"))
    if eq.get("samples", 0) < 20 or eq.get("all_pass") is not True or not all(eq.get("checks", {}).values()):
        raise RuntimeError("Q-fusion removal equivalence gate has not passed")
    installed_hook = a.phase2_package.parent / "ultralytics" / "nn" / "modules" / "ddm_fusion_hook.py"
    if (eq.get("weights") != str(a.weights) or eq.get("ddm_ckpt") != str(a.ddm_ckpt) or
            eq.get("rgbt_root") != str(a.rgbt_root) or not installed_hook.is_file() or
            eq.get("new_hook_md5") != hashlib.md5(installed_hook.read_bytes()).hexdigest()):
        raise RuntimeError("Equivalence evidence belongs to different weights/data or an older installed hook")
    manifests = ((a.m3fd_root, a.m3fd_manifest, 593, True),
                 (a.dvhn_root, a.dvhn_manifest, 400, False))
    for root, path, expected, scene_check in manifests:
        rows = records(path)
        if len(rows) != expected or len({r["rgb"] for r in rows}) != expected:
            raise RuntimeError(f"{path}: expected {expected} unique rows")
        if scene_check and any(r.get("split") != "test" or r.get("scene_block") != 0 or
                               r.get("smoke_fire_negative") not in (1, "1", True) for r in rows):
            raise RuntimeError("M3FD manifest must be negative test block 0")
        for row in rows:
            rgb = (root / row["rgb"]).resolve()
            ir = (root / row["ir_gray"]).resolve()
            cache = (root / "depth_cache" / row["rgb"]).with_suffix(".png").resolve()
            if any(not p.is_relative_to(root.resolve()) or not p.is_file() for p in (rgb, ir, cache)):
                raise FileNotFoundError((rgb, ir, cache))


def gpu_check(device):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    free, _ = torch.cuda.mem_get_info(device)
    if free < 8 * 1024**3:
        raise RuntimeError(f"cuda:2 free memory {free / 1024**3:.2f} GiB < 8 GiB safety floor")


def launch(cmd, env, notes):
    with notes.open("a", encoding="utf-8") as f:
        f.write("\n```bash\n" + " ".join(shlex.quote(str(x)) for x in cmd) + "\n```\n")
    print("[QFUSION-MAINLINE] " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), env=env, check=True)


def split_eval(source: Path, target: Path, chain: str, theta: str):
    metric = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    dataset = "rgbt_val" if chain == "rgbt_val" else "m3fd_hardneg"
    detections = [row for row in records(source / "detections3d.jsonl") if row["dataset"] == dataset]
    target.mkdir(parents=True)
    if chain == "rgbt_val":
        out = {"chain": chain, "theta": float(theta), "n_images": metric["n_rgbt_images"],
               "mAP50": metric["mAP50"], "AP50_95": metric["AP50_95"],
               "n_detections3d": len(detections)}
    else:
        per_image = records(source / "hardneg_per_image.jsonl")
        if sum(row["event_fp"] for row in per_image) != metric["hardneg_fp"]:
            raise RuntimeError(f"{chain}: per-image event FP does not sum to aggregate")
        out = {"chain": chain, "theta": float(theta), "n_images": metric["n_hardneg"],
               "hardneg_fp": metric["hardneg_fp"], "person_fp": metric["hardneg_fp_person"],
               "n_vetoed_hardneg": sum(row["n_vetoed"] for row in per_image),
               "n_detections3d": len(detections)}
        (target / "hardneg_per_image.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in per_image), encoding="utf-8")
    out["weights"] = metric["weights"]
    out["ddm_ckpt"] = metric["ddm_ckpt"]
    out["hook_fusion_mode"] = metric.get("hook_fusion_mode")
    out["veto_confidence_ceiling"] = metric["phy_ddm_veto_c"]
    out["source_combined_metrics"] = str(source / "metrics.json")
    out["metrics_path"] = str(target / "metrics.json")
    (target / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (target / "detections3d.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in detections), encoding="utf-8")
    return out


def alarm(chain: str, theta: str, row: dict):
    expected = EXPECTED[chain][theta]
    problems = []
    if abs(row["hardneg_fp"] - expected) > 5:
        problems.append(f"{chain} theta={theta} event FP {row['hardneg_fp']} outside {expected}±5")
    if chain == "m3fd" and row["person_fp"] != 363:
        problems.append(f"M3FD person FP {row['person_fp']} != 363")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--ddm-ckpt", type=Path, required=True)
    ap.add_argument("--rgbt-root", type=Path, required=True)
    ap.add_argument("--m3fd-root", type=Path, required=True)
    ap.add_argument("--m3fd-manifest", type=Path, required=True)
    ap.add_argument("--dvhn-root", type=Path, required=True)
    ap.add_argument("--dvhn-manifest", type=Path, required=True)
    ap.add_argument("--phase2-package", type=Path, required=True)
    ap.add_argument("--equivalence", type=Path, required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    preflight(a)
    gpu_check(a.device)
    pkg = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(PHY_DDM="1", PHY_DDM_CUE_MODE="depth", PHY_DDM_W1="0.5", PHY_DDM_W2="0.3",
               PHY_DDM_W3="0.2", PHY_F3D_SCALE="50", PHY_FG_MODE="fixed", PHY_FG_ALPHA="0.5",
               PHY_FRP="1", PHY_STAGE="3", PHY_FG="1", PHY_FIELD_IMPL="heatfield",
               PHY_FRP_TAU="0.045", PHY_FRP_CC_IMPL="batched", PHY_FRP_TOPK="0",
               PHY_DDM_VETO_C="0.25", PHY_DDM_FIELD_VETO_T="0")
    env["PYTHONPATH"] = os.pathsep.join((str(a.phase2_package.parent), str(pkg),
                                          str(a.phase2_package), env.get("PYTHONPATH", "")))
    a.out.mkdir(parents=True)
    notes = a.out / "NOTES.md"
    template = (pkg / "NOTES_q融合摘除与主结果重测_20261003.md").read_text(encoding="utf-8")
    notes.write_text(template + "\n## 本次服务器实际执行\n\n等价性闸：" + str(a.equivalence) +
                     "\n\n环境：\n```text\n" + "\n".join(f"{k}={v}" for k, v in sorted(env.items())
                         if k.startswith(("PHY_", "PYTHONPATH"))) + "\n```\n\n执行命令：\n", encoding="utf-8")
    summary = {"chain": "qfusion_removed_veto_only", "device": a.device,
               "equivalence": str(a.equivalence), "rows": [], "rgbt_repeatability": []}
    for dataset_root, manifest, hard_chain in ((a.m3fd_root, a.m3fd_manifest, "m3fd"),
                                                (a.dvhn_root, a.dvhn_manifest, "dvhn400")):
        for theta in THETAS:
            gpu_check(a.device)
            source = a.out / "combined" / f"{hard_chain}_theta_{theta}"
            call_env = env.copy()
            call_env["PHY_DDM_VETO_T"] = theta
            cmd = [sys.executable, "-u", str(pkg / "eval_ddm_val.py"),
                   "--weights", a.weights, "--ddm-ckpt", a.ddm_ckpt,
                   "--rgbt-root", a.rgbt_root, "--rgbt-cache", a.rgbt_root / "depth_cache",
                   "--m3fd-root", dataset_root, "--hardneg-manifest", manifest,
                   "--phase2-package", a.phase2_package, "--device", a.device,
                   "--tau", "0.045", "--depth-ablation", "full",
                   "--require-veto-only-hook", "--out", source]
            with notes.open("a", encoding="utf-8") as f:
                f.write(f"\nPHY_DDM_VETO_T={theta}\n")
            launch(cmd, call_env, notes)
            hard = split_eval(source, a.out / hard_chain / f"theta_{theta}", hard_chain, theta)
            summary["rows"].append(hard)
            problems = alarm(hard_chain, theta, hard)
            if hard["n_images"] != (593 if hard_chain == "m3fd" else 400):
                problems.append(f"{hard_chain}: evaluation image count drifted to {hard['n_images']}")
            if problems:
                (a.out / "deviation_alarm.json").write_text(json.dumps({"problems": problems,
                    "last_metric": hard, "stopped_before_next_run": True}, indent=2), encoding="utf-8")
                raise RuntimeError("; ".join(problems) + "; stopping before next run")
            rgbt = split_eval(source, a.out / "rgbt_val" / f"{hard_chain}_theta_{theta}", "rgbt_val", theta)
            if rgbt["n_images"] != 905:
                raise RuntimeError(f"RGBT val count drifted to {rgbt['n_images']}")
            rgbt["paired_hardneg_chain"] = hard_chain
            (summary["rows"] if hard_chain == "m3fd" else summary["rgbt_repeatability"]).append(rgbt)
            (a.out / "summary.partial.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # The M3FD-paired RGBT rows are the main validation chain; DVHN-paired
    # rows are retained separately to expose any repeatability difference.
    for chain in ("m3fd", "rgbt_val", "dvhn400"):
        chain_root = a.out / chain
        rows = [row for row in summary["rows"] if row["chain"] == chain and
                (chain != "rgbt_val" or row["paired_hardneg_chain"] == "m3fd")]
        (chain_root / "metrics.json").write_text(json.dumps({"chain": chain, "theta_rows": rows}, indent=2), encoding="utf-8")
        all_boxes = []
        for row in rows:
            sub = f"theta_{row['theta']:g}" if chain != "rgbt_val" else f"m3fd_theta_{row['theta']:g}"
            for box in records(chain_root / sub / "detections3d.jsonl"):
                all_boxes.append(dict(box, theta=row["theta"]))
        (chain_root / "detections3d.jsonl").write_text(
            "".join(json.dumps(box, allow_nan=False) + "\n" for box in all_boxes), encoding="utf-8")
    dvhn_person = [row["person_fp"] for row in summary["rows"] if row["chain"] == "dvhn400"]
    if len(set(dvhn_person)) != 1:
        (a.out / "deviation_alarm.json").write_text(json.dumps({
            "problems": [f"DVHN person FP changed across theta: {dvhn_person}"],
            "stopped_before_acceptance": True}, indent=2), encoding="utf-8")
        raise RuntimeError("DVHN person exemption changed across theta")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[QFUSION-MAINLINE-SUMMARY] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
