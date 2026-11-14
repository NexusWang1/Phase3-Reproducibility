"""Frozen-weight four-arm depth attribution at val-selected theta=0.001."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

THETA = "0.001"
ARMS = {"shuffle": "shuffle", "const": "const", "rgb_only": "rgb_only"}


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preflight(a):
    if a.device != "cuda:2":
        raise RuntimeError("This task is registered for cuda:2 only")
    if a.out.exists():
        raise FileExistsError(f"Refusing overwrite: {a.out}")
    if a.full_reference.name != "m3fd_theta_0.001" or a.full_reference.parent.name != "combined":
        raise RuntimeError("A must directly reuse qfusion-removed mainline combined/m3fd_theta_0.001")
    for path in (a.weights, a.ddm_ckpt, a.val_manifest, a.test_manifest,
                 a.full_selection, a.full_reference / "metrics.json",
                 a.full_reference / "detections3d.jsonl", a.full_reference / "hardneg_per_image.jsonl",
                 a.rgbt_root, a.m3fd_root, a.phase2_package):
        if not path.exists():
            raise FileNotFoundError(path)
    val, test = read_rows(a.val_manifest), read_rows(a.test_manifest)
    errors = []
    if len(val) != 402 or len(test) != 593:
        errors.append(f"Manifest counts must be 402/593; got {len(val)}/{len(test)}")
    if any(r.get("split") != "val" or r.get("scene_block") != 1 for r in val):
        errors.append("Val manifest is not scene block 1")
    if any(r.get("split") != "test" or r.get("scene_block") != 0 or
           r.get("smoke_fire_negative") not in (1, "1", True) for r in test):
        errors.append("Test manifest is not negative scene block 0")
    selection = json.loads(a.full_selection.read_text(encoding="utf-8"))
    selected_t = float(selection["selected"]["theta"])
    if selected_t != float(THETA) or len(selection.get("grid", [])) != 8:
        errors.append("Existing val-only selection must be theta=0.001 from eight grid points")
    if selection.get("test_block_0_used") is not False:
        errors.append("Theta selection must not use test block 0")
    if selection.get("val_manifest_sha256") != hashlib.sha256(a.val_manifest.read_bytes()).hexdigest():
        errors.append("A selection val manifest differs from current block 1")
    full = json.loads((a.full_reference / "metrics.json").read_text(encoding="utf-8"))
    if (full.get("n_hardneg") != 593 or full.get("n_rgbt_images") != 905 or
            float(full.get("phy_ddm_veto_t", -1)) != float(THETA) or
            full.get("hook_fusion_mode") != "veto_only" or full.get("depth_ablation") != "full"):
        errors.append("A reference must be the new veto-only full chain, 593+905 images, theta=0.001")
    if Path(full.get("weights", "")) != a.weights or Path(full.get("ddm_ckpt", "")) != a.ddm_ckpt:
        errors.append("A reference frozen weights/checkpoint differ")
    if errors:
        raise RuntimeError("Pre-registration conflicts; no GPU work started: " + "; ".join(errors))
    return selection, full


def gpu_check(device: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    free, total = torch.cuda.mem_get_info(device)
    if free < 8 * 1024**3:
        raise RuntimeError(f"cuda:2 has only {free / 1024**3:.2f} GiB free (<8 GiB safety floor)")
    return {"free_gib": round(free / 1024**3, 3), "total_gib": round(total / 1024**3, 3)}


def run(cmd: list[str], env: dict[str, str], notes: Path):
    with notes.open("a", encoding="utf-8") as file:
        file.write("\n```bash\n" + " ".join(__import__("shlex").quote(x) for x in cmd) + "\n```\n")
    print("[DEPTH-ATTR]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--ddm-ckpt", type=Path, required=True)
    p.add_argument("--rgbt-root", type=Path, required=True)
    p.add_argument("--m3fd-root", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--full-selection", type=Path, required=True)
    p.add_argument("--full-reference", type=Path, required=True)
    p.add_argument("--phase2-package", type=Path, required=True)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    selection, full = preflight(a)  # No GPU work or output directory before this gate.
    gpu_check(a.device)
    pkg = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(PHY_DDM="1", PHY_DDM_W1="0.5", PHY_DDM_W2="0.3", PHY_DDM_W3="0.2",
               PHY_F3D_SCALE="50", PHY_FG_MODE="fixed", PHY_FG_ALPHA="0.5",
               PHY_FRP="1", PHY_STAGE="3", PHY_FG="1", PHY_FIELD_IMPL="heatfield",
               PHY_FRP_TAU="0.045", PHY_FRP_CC_IMPL="batched", PHY_FRP_TOPK="0",
               PHY_DDM_VETO_C="0.25", PHY_DDM_VETO_T=THETA,
               PHY_DDM_FIELD_VETO_T="0")
    env["PYTHONPATH"] = os.pathsep.join((str(a.phase2_package.parent), str(pkg),
                                           str(a.phase2_package), env.get("PYTHONPATH", "")))
    a.out.mkdir(parents=True)
    notes = a.out / "NOTES.md"
    template = (pkg / "NOTES_深度归因四臂_20261003.md").read_text(encoding="utf-8")
    notes.write_text(template + "\n## 本次服务器实际执行\n\n"
                     "预注册：只用 cuda:2；seed=20261003；冻结权重；四臂统一使用既有 "
                     "val block 1 选定的 θ=0.001；不重新扫描，不用 test 选型。\n\n"
                     "环境：\n```text\n" + "\n".join(f"{k}={env[k]}" for k in (
                         "PHY_DDM", "PHY_DDM_W1", "PHY_DDM_W2", "PHY_DDM_W3", "PHY_F3D_SCALE",
                         "PHY_FG_MODE", "PHY_FG_ALPHA", "PHY_FRP", "PHY_STAGE", "PHY_FG",
                         "PHY_FIELD_IMPL", "PHY_FRP_TAU", "PHY_FRP_CC_IMPL", "PHY_FRP_TOPK",
                         "PHY_DDM_VETO_C", "PHY_DDM_VETO_T",
                         "PHY_DDM_FIELD_VETO_T")) +
                     "\n```\n\n执行命令：\n", encoding="utf-8")
    (a.out / "selection_source.json").write_text(json.dumps({"path": str(a.full_selection),
        "val_manifest_sha256": selection["val_manifest_sha256"], "theta": float(THETA)}, indent=2), encoding="utf-8")
    full_dir = a.out / "full"
    full_dir.mkdir()
    for name in ("metrics.json", "detections3d.jsonl", "hardneg_per_image.jsonl"):
        shutil.copy2(a.full_reference / name, full_dir / name)
    (full_dir / "source.json").write_text(json.dumps({"from_qfusion_removed_mainline": str(a.full_reference),
        "not_rerun": True}, indent=2), encoding="utf-8")
    table = a.out / "depth_shuffle_seed20261003.json"
    run([sys.executable, str(pkg / "tools" / "make_depth_shuffle.py"),
         "--rgbt-root", str(a.rgbt_root), "--m3fd-root", str(a.m3fd_root),
         "--val-manifest", str(a.val_manifest), "--test-manifest", str(a.test_manifest),
         "--phase2-package", str(a.phase2_package), "--seed", "20261003", "--out", str(table)], env, notes)
    summary = {"protocol": "frozen four-arm depth attribution", "device": a.device,
               "test_count": 593, "rgbt_val_count": 905, "theta": float(THETA),
               "theta_selection_source": str(a.full_selection),
               "arms": {"full": {"hardneg_fp": full["hardneg_fp"], "mAP50": full["mAP50"],
                                  "theta": float(THETA), "n_vetoed": full["n_vetoed"],
                                  "fp_minus_full": 0, "map_minus_full": 0,
                                  "metrics": str(full_dir / "metrics.json")}}}
    for arm, mode in ARMS.items():
        arm_dir = a.out / arm
        gpu_check(a.device)
        cmd = [sys.executable, "-u", str(pkg / "eval_ddm_val.py"),
               "--weights", str(a.weights), "--ddm-ckpt", str(a.ddm_ckpt),
               "--rgbt-root", str(a.rgbt_root), "--rgbt-cache", str(a.rgbt_root / "depth_cache"),
               "--m3fd-root", str(a.m3fd_root), "--hardneg-manifest", str(a.test_manifest),
               "--phase2-package", str(a.phase2_package), "--device", a.device,
               "--tau", "0.045", "--depth-ablation", mode,
               "--require-veto-only-hook",
               "--m3fd-cohort", "m3fd_test", "--out", str(arm_dir)]
        if arm == "shuffle":
            cmd += ["--shuffle-table", str(table)]
        run(cmd, env, notes)
        metric = json.loads((arm_dir / "metrics.json").read_text(encoding="utf-8"))
        if metric["n_hardneg"] != 593 or metric["n_rgbt_images"] != 905 or float(metric["phy_ddm_veto_t"]) != float(THETA):
            raise RuntimeError(f"{arm}: test counts drifted")
        if metric.get("depth_ablation") != mode:
            raise RuntimeError(f"{arm}: wrong depth ablation mode")
        if metric.get("hook_fusion_mode") != "veto_only":
            raise RuntimeError(f"{arm}: q-fusion hook unexpectedly active")
        summary["arms"][arm] = {"hardneg_fp": metric["hardneg_fp"], "mAP50": metric["mAP50"],
            "theta": float(THETA), "n_vetoed": metric["n_vetoed"],
            "fp_minus_full": metric["hardneg_fp"] - full["hardneg_fp"],
            "map_minus_full": metric["mAP50"] - full["mAP50"],
            "metrics": str(arm_dir / "metrics.json")}
        (a.out / "summary.partial.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["directional_fp_increase"] = {arm: summary["arms"][arm]["hardneg_fp"] > summary["arms"]["full"]["hardneg_fp"]
                                          for arm in ("shuffle", "const")}
    summary["interpretation"] = ("B/C both increase FP; assess magnitude before claiming significant depth attribution"
                                 if all(summary["directional_fp_increase"].values())
                                 else "Negative or mixed: do not claim depth structure is proven")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DEPTH-ATTR-SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
