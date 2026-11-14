"""Real-frame bit-equality gate before q-fusion-free mainline evaluation."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def load_old_hook(path: Path):
    name = "ultralytics.nn.modules._ddm_fusion_hook_backup"
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DDMFusionHook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--ddm-ckpt", type=Path, required=True)
    ap.add_argument("--rgbt-root", type=Path, required=True)
    ap.add_argument("--phase2-package", type=Path, required=True)
    ap.add_argument("--old-hook", type=Path, required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.device != "cuda:2" or a.samples < 20:
        raise ValueError("Requires cuda:2 and >=20 real paired samples")
    for path in (a.weights, a.ddm_ckpt, a.rgbt_root, a.phase2_package, a.old_hook):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    free, _ = torch.cuda.mem_get_info(a.device)
    if free < 8 * 1024**3:
        raise RuntimeError(f"cuda:2 free memory {free / 1024**3:.2f} GiB < 8 GiB safety floor")
    os.environ.update(PHY_DDM_VETO_T="0", PHY_DDM_VETO_C="0.25", PHY_DDM="0",
                      PHY_DDM_CUE_MODE="depth", PHY_FRP="1", PHY_FRP_TAU="0.045",
                      PHY_FRP_CC_IMPL="batched", PHY_FRP_TOPK="0", PHY_STAGE="3",
                      PHY_FG="1", PHY_FG_MODE="fixed", PHY_FG_ALPHA="0.5",
                      PHY_FIELD_IMPL="heatfield", PHY_F3D_SCALE="50")
    sys.path[:0] = [str(a.phase2_package), str(a.phase2_package.parent / "FieldGuide_Phase2_FieldUse_20260916")]
    from smoke_frp_v2_7 import make
    from field_use_common import paired_paths, prepare_six, field_and_detections
    from eval_ddm_val import _cache, _depth_letterboxed
    from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
    from ultralytics.utils.nms import non_max_suppression

    pairs = list(paired_paths(a.rgbt_root, "val"))
    if len(pairs) != 905:
        raise RuntimeError(f"Expected 905 RGBT val pairs; got {len(pairs)}")
    selected = [pairs[i * len(pairs) // a.samples] for i in range(a.samples)]
    old_cls = load_old_hook(a.old_hook)
    active = [None]

    def snapshot(model, rgb, ir):
        six, tr = prepare_six(rgb, ir, torch, a.device)
        active[0] = _depth_letterboxed(_cache(a.rgbt_root / "depth_cache", a.rgbt_root, rgb), tr, a.device)
        _, det = field_and_detections(model, torch, non_max_suppression, False, six, tr, .045, {0, 1, 2})
        pur = model._last_t_pur.detach().cpu().clone()
        return pur, det, model._last_frp_aux

    old, _, _ = make(a.weights, a.device, True)
    old_cls(depth_provider=lambda _rgb: active[0]).attach(old, a.ddm_ckpt)
    matched_state = {key: value.detach().cpu().clone() for key, value in old.state_dict().items()}
    old_records = []
    for rgb, ir, _ in selected:
        os.environ["PHY_DDM"] = "0"
        pur, det, aux = snapshot(old, rgb, ir)
        if "p_ddm" in aux:
            raise RuntimeError("Old hook computed DDM with PHY_DDM=0")
        old_records.append((pur, det))
    del old
    gc.collect()
    torch.cuda.empty_cache()

    new, _, _ = make(a.weights, a.device, True)
    new.load_state_dict(matched_state, strict=True)
    DDMFusionHook(depth_provider=lambda _rgb: active[0]).attach(new, a.ddm_ckpt)
    rows = []
    for (rgb, ir, _), (old_pur, old_det) in zip(selected, old_records):
        os.environ["PHY_DDM"] = "0"
        off_pur, off_det, off_aux = snapshot(new, rgb, ir)
        off_no_ddm = "p_ddm" not in off_aux
        os.environ["PHY_DDM"] = "1"
        on_pur, on_det, on_aux = snapshot(new, rgb, ir)
        row = {"image": str(rgb),
               "old_new_off_pur_equal": bool(torch.equal(old_pur, off_pur)),
               "old_new_off_det_equal": bool(np.array_equal(old_det, off_det)),
               "new_on_off_pur_equal": bool(torch.equal(off_pur, on_pur)),
               "new_on_off_det_equal": bool(np.array_equal(off_det, on_det)),
               "off_skips_ddm": off_no_ddm,
               "on_has_p_ddm": "p_ddm" in on_aux,
               "on_has_no_q": "q" not in on_aux}
        rows.append(row)
        print("[QFUSION-EQ] " + json.dumps(row), flush=True)
    checks = [key for key in rows[0] if key != "image"]
    result = {"device": a.device, "samples": len(rows), "threshold": 0,
              "weights": str(a.weights), "ddm_ckpt": str(a.ddm_ckpt),
              "rgbt_root": str(a.rgbt_root), "old_hook": str(a.old_hook),
              "new_hook_md5": hashlib.md5(Path(sys.modules[DDMFusionHook.__module__].__file__).read_bytes()).hexdigest(),
              "checks": {key: all(row[key] for row in rows) for key in checks},
              "all_pass": all(all(row[key] for key in checks) for row in rows),
              "per_image": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[QFUSION-EQ-SUMMARY] " + json.dumps({k: result[k] for k in ("samples", "checks", "all_pass")}), flush=True)
    if not result["all_pass"]:
        raise RuntimeError("Q-fusion removal equivalence failed; do not start mainline retest")


if __name__ == "__main__":
    main()
