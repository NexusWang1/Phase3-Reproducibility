"""IGNITE v3: ego-motion-stabilized front-direction test.

UAV platform motion dominates camera-frame direction signals (v1/v2 finding:
even the front-advance ground truth has R<0.12). Here we:
1. Estimate the global translation between frame k and k+--delta by phase
   correlation on the radiometric tiff (hot pixels replaced by the frame
   median so the non-rigid fire does not bias the rigid background shift).
2. In the STABILIZED frame: truth = centroid(new mask pixels) - centroid(old
   mask), predictor = deflection vector on the positive-dT field, both in
   image coordinates (x right, y down).
3. Report direction-stability R of truth/predictor, median angle difference,
   and the platform-shift magnitude as a motion diagnostic.

Pure numpy/PIL (+optional scipy for dilation).
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
from PIL import Image

W, H = 640, 512
CX, CY = W / 2, H / 2

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None

_HANN = None


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


def phase_shift(a: np.ndarray, b: np.ndarray):
    """Integer (dy, dx) such that b[y,x] ~= a[y+dy, x+dx]; align b to a with shift_image(b, dy, dx)."""
    global _HANN
    if _HANN is None:
        wy = np.hanning(H)[:, None]
        wx = np.hanning(W)[None, :]
        _HANN = wy * wx
    A = np.fft.fft2(a * _HANN)
    B = np.fft.fft2(b * _HANN)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.ifft2(R)
    py, px = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    dy = py - H if py > H // 2 else py
    dx = px - W if px > W // 2 else px
    return int(dy), int(dx)


def shift_image(img: np.ndarray, dy: int, dx: int, fill=0):
    out = np.full_like(img, fill)
    ys = slice(max(0, dy), H + min(0, dy))
    xs = slice(max(0, dx), W + min(0, dx))
    ysrc = slice(max(0, -dy), H - max(0, dy))
    xsrc = slice(max(0, -dx), W - max(0, dx))
    out[ys, xs] = img[ysrc, xsrc]
    return out


def centroid2d(mask):
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    return np.array([xs.mean(), ys.mean()])


def deflection2d(s: np.ndarray, threshold=0.3):
    ys, xs = np.where(s > threshold)
    if len(xs) < 4:
        return None
    w = s[ys, xs]
    center = np.average(np.c_[xs, ys], axis=0, weights=w)
    core = np.c_[xs, ys][w >= np.percentile(w, 90)].mean(0)
    return center - core


def angle_between(u, v, min_norm=3.0):
    if u is None or v is None:
        return None
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < min_norm or nv < min_norm:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def series_r(vecs, min_norm=3.0):
    u = [v / np.linalg.norm(v) for v in vecs if v is not None and np.linalg.norm(v) >= min_norm]
    return float(np.linalg.norm(np.mean(u, axis=0))) if len(u) >= 5 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--t80', type=float, default=8800.0)
    p.add_argument('--delta', type=int, default=10)
    p.add_argument('--min-area', type=int, default=50)
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
        tiffs, masks = [], []
        for r in rs:
            t = load_tiff(a.ig / r['thermal_tiff_file_name'])
            tiffs.append(t)
            masks.append(load_mask(a.ig / r['mask_80c_file_name']))
        recs = []
        for k in range(len(rs) - a.delta):
            j = k + a.delta
            if masks[k].sum() < a.min_area:
                continue
            med = np.median(tiffs[k])
            ta = np.where(tiffs[k] > a.t80, med, tiffs[k])
            tb = np.where(tiffs[j] > a.t80, np.median(tiffs[j]), tiffs[j])
            dy, dx = phase_shift(ta, tb)
            tj = shift_image(tiffs[j], dy, dx, fill=med)       # j warped into k frame
            mj = shift_image(masks[j].view(np.uint8), dy, dx) > 0
            grown = binary_dilation(masks[k], iterations=3) if binary_dilation else masks[k]
            new = mj & ~grown
            truth = None
            if new.sum() >= a.min_area:
                c_new, c_old = centroid2d(new), centroid2d(masks[k])
                if c_new is not None and c_old is not None:
                    truth = c_new - c_old
            dT = tj - tiffs[k]
            pos = dT[dT > 0]
            defl = None
            if pos.size >= 100:
                s = np.clip(dT / np.percentile(pos, 99), 0, 1)
                defl = deflection2d(s)
            recs.append({'t': rs[k]['video_time_s'], 'defl': defl, 'truth': truth,
                         'shift': (dy, dx), 'new_area': int(new.sum())})
        cols = ['t', 'defl_dx', 'defl_dy', 'truth_dx', 'truth_dy', 'ang_deg',
                'shift_dy', 'shift_dx', 'new_area']
        arr = []
        for rec in recs:
            ang = angle_between(rec['defl'], rec['truth'])
            arr.append([rec['t'],
                        *(rec['defl'] if rec['defl'] is not None else [np.nan] * 2),
                        *(rec['truth'] if rec['truth'] is not None else [np.nan] * 2),
                        ang if ang is not None else np.nan,
                        *rec['shift'], rec['new_area']])
        arr = np.array(arr)
        np.save(a.out / f'{sid}_stab_v3.npy', arr)
        angs = arr[:, 5][np.isfinite(arr[:, 5])]
        shifts = np.abs(arr[:, 6:8]).max(axis=1)
        summary[sid] = {
            'columns': cols, 'pairs': len(recs),
            'R_truth': series_r([r['truth'] for r in recs]),
            'R_defl': series_r([r['defl'] for r in recs]),
            'n_compared': int(len(angs)),
            'median_ang_deg': float(np.median(angs)) if len(angs) else None,
            'platform_shift_px_median': float(np.median(shifts)) if len(shifts) else None,
            'platform_shift_px_p90': float(np.percentile(shifts, 90)) if len(shifts) else None,
        }
        print(f'[IGNITE-V3] {sid} ' + json.dumps({k: v for k, v in summary[sid].items() if k != 'columns'}), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('[IGNITE-V3] done ->', a.out)


if __name__ == '__main__':
    main()
