"""IGNITE: 3D radiometric-field deflection vs mask-derived spread ground truth.

Pure analysis script (numpy/PIL only). For each sequence:
1. Calibrate the raw uint16 radiometric value corresponding to 80 C by
   maximizing IoU((tiff>t), mask_80c) over a percentile grid.
2. Build the heat field s = clip((T-t80)/(p99.5(T)-t80),0,1) per frame.
3. Backproject with the offline depth cache (native 640x512, nominal scale)
   and compute the deflection vector (strength-weighted centroid minus
   top-10% core), identical definition to the YHT pipeline.
4. Ground-truth spread: backprojected mask_80c centroid displacement over
   --delta frames; compare 3D direction against the deflection vector.

Outputs per sequence: deflection_vs_spread.npy + summary.json (global).
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
from PIL import Image

_f3d = Path(__file__).resolve().parents[1] / 'ultralytics/models/yolo/detect/field3d.py'
_spec = importlib.util.spec_from_file_location('phase3_field3d', _f3d)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
backproject_field = _mod.backproject_field

W, H = 640, 512
CX, CY = W / 2, H / 2


def load_tiff(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.shape != (H, W):
        raise ValueError(f'tiff shape {a.shape} != {(H, W)}: {path}')
    return a.astype(np.float64)


def load_mask(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert('L'))
    if a.shape != (H, W):
        raise ValueError(f'mask shape {a.shape} != {(H, W)}: {path}')
    return a > 127


def load_depth(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.shape != (H, W):
        raise ValueError(f'depth shape {a.shape} != {(H, W)}: {path}')
    return a.astype(np.float64) / 65535.0


def calibrate_t80(samples, n_grid=41):
    """Threshold of raw tiff value best reproducing mask_80c (max mean IoU)."""
    pairs = []
    lo_s, hi_s = [], []
    for tiff, mask in samples:
        if mask.sum() < 50:
            continue
        lo_s.append(np.percentile(tiff, 90))
        hi_s.append(np.percentile(tiff, 99.95))
        pairs.append((tiff, mask))
    if not pairs:
        raise RuntimeError('calibration: no usable samples')
    lo, hi = float(np.median(lo_s)), float(np.median(hi_s))
    if hi <= lo:
        raise RuntimeError('calibration: degenerate grid')
    best_t, best_iou = None, -1.0
    for t in np.linspace(lo, hi, n_grid):
        ious = []
        for tiff, mask in pairs:
            pred = tiff > t
            union = np.logical_or(pred, mask).sum()
            ious.append(np.logical_and(pred, mask).sum() / union if union else 1.0)
        iou = float(np.mean(ious))
        if iou > best_iou:
            best_t, best_iou = float(t), iou
    return best_t, best_iou


def deflection_vec(points):
    if len(points) < 4:
        return None
    w = points[:, 3]
    center = np.average(points[:, [0, 2]], axis=0, weights=w)
    core = points[w >= np.percentile(w, 90)][:, [0, 2]].mean(0)
    return center - core


def mask_centroid_3d(mask, depth, scale=50.0):
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    z = depth[ys, xs] * scale
    x3 = (xs - CX) * z / 640.0
    z3 = z  # depth axis
    return np.array([x3.mean(), z3.mean()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--delta', type=int, default=10, help='frames between mask centroids for displacement')
    p.add_argument('--min-area', type=int, default=50, help='min mask pixels for fire-active frame')
    p.add_argument('--min-disp', type=float, default=0.3, help='min displacement (nominal m) for angle comparison')
    p.add_argument('--scale', type=float, default=50.0)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    rows = [json.loads(l) for l in (a.ig / 'metadata.jsonl').open(encoding='utf-8')]
    seqs = {}
    for r in rows:
        seqs.setdefault(r['sequence_id'], []).append(r)
    a.out.mkdir(parents=True)
    summary = {}
    for sid, rs in sorted(seqs.items()):
        rs.sort(key=lambda r: r['video_time_s'])
        # --- calibration sample: fire frames spread over the sequence
        fire = [r for r in rs if r['mask_positive_pixels'] >= 200]
        step = max(1, len(fire) // 60)
        cal = [(load_tiff(a.ig / r['thermal_tiff_file_name']),
                load_mask(a.ig / r['mask_80c_file_name'])) for r in fire[::step]]
        t80, iou = calibrate_t80(cal)
        # --- per-frame pass
        recs = []
        for r in rs:
            tiff = load_tiff(a.ig / r['thermal_tiff_file_name'])
            mask = load_mask(a.ig / r['mask_80c_file_name'])
            dpath = a.ig / 'rgb_only' / 'depth_cache' / sid / (r['aligned_id'] + '.png')
            depth = load_depth(dpath)
            hi = np.percentile(tiff, 99.5)
            s = np.clip((tiff - t80) / max(hi - t80, 1e-9), 0, 1).astype(np.float32)
            pts = backproject_field(depth, s, fx=640., fy=640., cx=CX, cy=CY,
                                    scale=a.scale, threshold=.05, topk=20000)
            vec = deflection_vec(pts)
            mc = mask_centroid_3d(mask, depth, a.scale)
            recs.append({'t': r['video_time_s'], 'vec': vec, 'mc': mc,
                         'area': int(mask.sum())})
        # --- displacement + angle comparison
        out_rows = []
        for k, rec in enumerate(recs):
            j = k + a.delta
            disp = ang = None
            if j < len(recs) and rec['mc'] is not None and recs[j]['mc'] is not None:
                disp = recs[j]['mc'] - rec['mc']
                if rec['vec'] is not None and np.linalg.norm(disp) >= a.min_disp \
                        and np.linalg.norm(rec['vec']) > 0:
                    cos = np.dot(rec['vec'], disp) / (np.linalg.norm(rec['vec']) * np.linalg.norm(disp))
                    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            out_rows.append([rec['t'],
                             *(rec['vec'].tolist() if rec['vec'] is not None else [np.nan, np.nan]),
                             *(rec['mc'].tolist() if rec['mc'] is not None else [np.nan, np.nan]),
                             rec['area'],
                             *(disp.tolist() if disp is not None else [np.nan, np.nan]),
                             ang if ang is not None else np.nan])
        arr = np.array(out_rows, dtype=np.float64)
        np.save(a.out / f'{sid}_deflection_vs_spread.npy', arr)
        cols = ['t', 'defl_dX', 'defl_dZ', 'mask_cX', 'mask_cZ', 'mask_area',
                'disp_dX', 'disp_dZ', 'angle_diff_deg']
        angs = arr[:, 8][np.isfinite(arr[:, 8])]
        active = arr[(arr[:, 5] >= a.min_area)]
        summary[sid] = {
            'columns': cols, 'frames': len(rs),
            'fire_frames': int((arr[:, 5] >= a.min_area).sum()),
            't80_raw': t80, 'calib_iou': round(iou, 4),
            'n_angle_compared': int(len(angs)),
            'median_angle_diff_deg': float(np.median(angs)) if len(angs) else None,
            'mean_angle_diff_deg': float(np.mean(angs)) if len(angs) else None,
            'angle_concentration_R': float(np.abs(np.exp(1j * np.radians(angs)).mean())) if len(angs) else None,
            'first_fire_s': float(active[0, 0]) if len(active) else None,
        }
        print(f'[IGNITE] {sid} ' + json.dumps({k: v for k, v in summary[sid].items() if k != "columns"}), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('[IGNITE] done ->', a.out)


if __name__ == '__main__':
    main()
