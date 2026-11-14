"""Detection-aware c_dir audit for the q-removal YHT chain; no rendering."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ("weights", "ddm-ckpt", "rgb-root", "ir-root", "depth-cache", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--phase2-package", type=Path, default=Path(__file__).resolve().parents[1].parent / "FieldGuide_Phase2_FRP_20260916")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start", type=int, default=13680)
    p.add_argument("--end", type=int, default=14500)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--ignition-frame", type=int, default=13771)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.stride < 1:
        raise ValueError("stride must be positive")
    os.environ["PHY_DDM"] = "1"
    os.environ.setdefault("PHY_DDM_VETO_T", ".001")
    os.environ.setdefault("PHY_DDM_VETO_C", ".25")
    os.environ.setdefault("PHY_DDM_FIELD_VETO_T", "0")
    os.environ.setdefault("PHY_FRP_TAU", ".045")
    sys.path[:0] = [str(a.phase2_package), str(a.phase2_package.parent / "FieldGuide_Phase2_FieldUse_20260916"), str(Path(__file__).resolve().parents[1])]
    from smoke_frp_v2_7 import make
    from field_use_common import prepare_six, field_and_detections
    from eval_ddm_val import _cache, _depth_letterboxed, _apply_box_veto
    from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
    from ultralytics.models.yolo.detect.field3d import backproject_field, deflection_confidence
    from tools.paper_fig_3d import _vector_3d
    from tools.ddm_dataset import images, frame_id
    from ultralytics.utils.nms import non_max_suppression

    rgb = {frame_id(x): x for x in images(a.rgb_root) if frame_id(x) is not None}
    ir = {frame_id(x): x for x in images(a.ir_root) if frame_id(x) is not None}
    active = [None]
    model, _, _ = make(a.weights, a.device, True)
    DDMFusionHook(depth_provider=lambda _rgb: active[0]).attach(model, a.ddm_ckpt)
    history = []
    frames = []
    missing = []
    for frame in sorted(k for k in rgb if a.start <= k <= a.end and (k - a.start) % a.stride == 0):
        if frame not in ir:
            missing.append({"frame": frame, "reason": "ir"})
            continue
        depth_path = _cache(a.depth_cache, a.rgb_root, rgb[frame])
        if not depth_path.is_file():
            missing.append({"frame": frame, "reason": "depth"})
            continue
        six, tr = prepare_six(rgb[frame], ir[frame], torch, a.device)
        active[0] = _depth_letterboxed(depth_path, tr, a.device)
        _, det = field_and_detections(model, torch, non_max_suppression, False, six, tr, .045, {0, 1, 2})
        pddm = model._last_frp_aux["p_ddm"][0, 0].detach().float().cpu().numpy()
        kept, _ = _apply_box_veto(det, pddm, tr, float(os.environ["PHY_DDM_VETO_T"]), float(os.environ["PHY_DDM_VETO_C"]), True)
        pur = getattr(model, "_last_t_pur", None)
        if pur is None:
            raise RuntimeError(f"purified field missing at {frame}")
        field = pur[0, 0].detach().float().cpu().numpy()
        depth = active[0][0, 0].detach().float().cpu().numpy()
        _, vec = _vector_3d(field, depth, backproject_field)
        score = deflection_confidence(field, vec, history, kept_classes={int(row[5]) for row in kept})
        history.append(vec)
        history = history[-50:]
        frames.append({"frame": frame, "c_dir": score, "kept_classes": sorted({int(row[5]) for row in kept})})
        if len(frames) % 20 == 0:
            print(f"[C-DIR] {len(frames)}", flush=True)
    pre = [row["c_dir"] for row in frames if row["frame"] < a.ignition_frame]
    post = [row["c_dir"] for row in frames if row["frame"] >= a.ignition_frame]
    summary = {"weights": str(a.weights), "ddm_ckpt": str(a.ddm_ckpt), "start": a.start, "end": a.end,
               "stride": a.stride, "ignition_frame": a.ignition_frame, "pre_fire_n": len(pre),
               "pre_fire_mean": float(np.mean(pre)) if pre else None, "after_ignition_n": len(post),
               "after_ignition_mean": float(np.mean(post)) if post else None,
               "missing": missing, "frames": frames,
               "formula": "field core strength × kept smoke/fire gate × angular consistency; identical to demo_phase3"}
    a.out.mkdir(parents=True)
    with (a.out / "cdir_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("[C-DIR] " + json.dumps({k: summary[k] for k in ("pre_fire_n", "pre_fire_mean", "after_ignition_n", "after_ignition_mean")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
