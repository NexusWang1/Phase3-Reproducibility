"""IGNITE v2: fire-front formulation.

Predictor and ground truth both target the ADVANCING FRONT instead of the
whole burn scar:
- field: positive temperature rise dT over --dtdiff frames (the advancing
  front is where temperature is rising), normalized by its own p99
- predictor A: 3D deflection vector on the dT field (same definition as YHT)
- predictor B: advance vector = centroid3d(dT-hot region) - centroid3d(mask_k)
- truth: advance vector = centroid3d(NEW mask pixels at k+--delta, after
  dilating mask_k away) - centroid3d(mask_k)

Angle between each predictor and truth is reported per frame pair, plus the
direction-stability R of each series. Pure numpy/PIL (+optional scipy).
"""
from __future__ import annotations
import argparse, importlib.util, json
from collections import deque
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

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None


def load_tiff(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.shape != (H, W):
        raise ValueError(f'tiff shape {a.shape}: {path}')
    return a.astype(np.float64)


def load_mask(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert('L'))
    if a.shape != (H, W):
        raise ValueError(f'mask shape {a.shape}: {path}')
    return a > 127


def load_depth(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.float64) / 65535.0


def centroid3d(mask, depth, scale):
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    z = depth[ys, xs] * scale
    return np.array([((xs - CX) * z / 640.0).mean(), z.mean()])


def deflection_vec(points):
    if len(points) < 4:
        return None
    w = points[:, 3]
    center = np.average(points[:, [0, 2]], axis=0, weights=w)
    core = points[w >= np.percentile(w, 90)][:, [0, 2]].mean(0)
    return center - core


def angle_between(u, v, min_norm=0.3):
    if u is None or v is None:
        return None
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < min_norm or nv < min_norm:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def series_r(vecs):
    u = [v / np.linalg.norm(v) for v in vecs if v is not None and np.linalg.norm(v) > 0.3]
    return float(np.linalg.norm(np.mean(u, axis=0))) if len(u) >= 5 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--t80', type=float, default=8800.0, help='raw tiff value of 80C from v1 calibration')
    p.add_argument('--delta', type=int, default=10, help='frames ahead for truth')
    p.add_argument('--dtdiff', type=int, default=5, help='frames back for dT field')
    p.add_argument('--min-area', type=int, default=50)
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
        hist = deque(maxlen=a.dtdiff + 1)  # (tiff, mask)
        recs = []
        for k, r in enumerate(rs):
            tiff = load_tiff(a.ig / r['thermal_tiff_file_name'])
            mask = load_mask(a.ig / r['mask_80c_file_name'])
            hist.append((tiff, mask))
            rec = {'t': r['video_time_s'], 'defl': None, 'padv': None, 'truth': None,
                   'area': int(mask.sum()), 'new_area': 0}
            if len(hist) == a.dtdiff + 1 and mask.sum() >= a.min_area:
                depth = load_depth(a.ig / 'rgb_only' / 'depth_cache' / sid / (r['aligned_id'] + '.png'))
                dT = tiff - hist[0][0]
                pos = dT[dT > 0]
                if pos.size >= 100:
                    s = np.clip(dT / np.percentile(pos, 99), 0, 1).astype(np.float32)
                    pts = backproject_field(depth, s, fx=640., fy=640., cx=CX, cy=CY,
                                            scale=a.scale, threshold=.3, topk=20000)
                    rec['defl'] = deflection_vec(pts)
                    c_new, c_old = centroid3d(s > 0.5, depth, a.scale), centroid3d(mask, depth, a.scale)
                    if c_new is not None and c_old is not None:
                        rec['padv'] = c_new - c_old
                rec['depth'] = depth  # kept only until truth is resolved below
            recs.append(rec)
        # truth pass: new-mask advance direction
        for k, rec in enumerate(recs):
            j = k + a.delta
            if j >= len(recs) or 'depth' not in rec:
                rec.pop('depth', None)
                continue
            mask_k = load_mask(a.ig / rs[k]['mask_80c_file_name'])
            mask_j = load_mask(a.ig / rs[j]['mask_80c_file_name'])
            grown = binary_dilation(mask_k, iterations=3) if binary_dilation else mask_k
            new = mask_j & ~grown
            rec['new_area'] = int(new.sum())
            if new.sum() >= a.min_area and mask_k.sum() >= a.min_area:
                c_new, c_old = centroid3d(new, rec['depth'], a.scale), centroid3d(mask_k, rec['depth'], a.scale)
                if c_new is not None and c_old is not None:
                    rec['truth'] = c_new - c_old
            del rec['depth']
        cols = ['t', 'defl_dX', 'defl_dZ', 'padv_dX', 'padv_dZ', 'truth_dX', 'truth_dZ',
                'ang_defl_deg', 'ang_padv_deg', 'mask_area', 'new_area']
        arr = []
        for rec in recs:
            a_defl = angle_between(rec['defl'], rec['truth'])
            a_padv = angle_between(rec['padv'], rec['truth'])
            arr.append([rec['t'],
                        *(rec['defl'] if rec['defl'] is not None else [np.nan] * 2),
                        *(rec['padv'] if rec['padv'] is not None else [np.nan] * 2),
                        *(rec['truth'] if rec['truth'] is not None else [np.nan] * 2),
                        a_defl if a_defl is not None else np.nan,
                        a_padv if a_padv is not None else np.nan,
                        rec['area'], rec['new_area']])
        arr = np.array(arr)
        np.save(a.out / f'{sid}_front_v2.npy', arr)
        ad, ap = arr[:, 7][np.isfinite(arr[:, 7])], arr[:, 8][np.isfinite(arr[:, 8])]
        summary[sid] = {
            'columns': cols, 'frames': len(rs),
            'R_defl': series_r([r['defl'] for r in recs]),
            'R_padv': series_r([r['padv'] for r in recs]),
            'R_truth': series_r([r['truth'] for r in recs]),
            'n_defl': int(len(ad)), 'median_ang_defl': float(np.median(ad)) if len(ad) else None,
            'n_padv': int(len(ap)), 'median_ang_padv': float(np.median(ap)) if len(ap) else None,
        }
        print(f'[IGNITE-V2] {sid} ' + json.dumps({k: v for k, v in summary[sid].items() if k != 'columns'}), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('[IGNITE-V2] done ->', a.out)


if __name__ == '__main__':
    main()
