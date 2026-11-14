"""IGNITE v4: direct spread-along-prediction test (scalar, per user spec).

Question: in consecutive frames, does the NEW fire area actually grow along
the predicted direction? For each frame k:
- predicted direction v_k = 3D deflection vector (X-Z ground plane) from the
  radiometric field (same definition as the YHT pipeline), plus a dT-field
  variant as second predictor
- new fire pixels at k+--delta = mask_{k+d} & ~dilate(mask_k)
- scalar evidence: projections of (new-pixel 3D positions - current fire
  centroid) onto v_hat; fire validated if the projection distribution is
  consistently positive

Also runs an image-space variant (2D deflection, pixel units) and an optional
phase-correlation translation-stabilized variant. Per sequence reports:
median signed projection, fraction of new pixels ahead (proj>0), and the
fraction of frame pairs with positive median. Pure numpy/PIL (+opt. scipy).
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

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None

_HANN = None


def load_tiff(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.float64)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L')) > 127


def load_depth(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.float64) / 65535.0


def phase_shift(a: np.ndarray, b: np.ndarray):
    global _HANN
    if _HANN is None:
        _HANN = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    A = np.fft.fft2(a * _HANN)
    B = np.fft.fft2(b * _HANN)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.ifft2(R)
    py, px = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    return (int(py - H if py > H // 2 else py), int(px - W if px > W // 2 else px))


def shift_image(img, dy, dx, fill=0):
    out = np.full_like(img, fill)
    ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
    out[ys, xs] = img[slice(max(0, -dy), H - max(0, dy)), slice(max(0, -dx), W - max(0, dx))]
    return out


def field_from_tiff(tiff, t80):
    hi = np.percentile(tiff, 99.5)
    return np.clip((tiff - t80) / max(hi - t80, 1e-9), 0, 1).astype(np.float32)


def deflection_vec_3d(points):
    if len(points) < 4:
        return None
    w = points[:, 3]
    center = np.average(points[:, [0, 2]], axis=0, weights=w)
    core = points[w >= np.percentile(w, 90)][:, [0, 2]].mean(0)
    v = center - core
    return v if np.linalg.norm(v) > 0 else None


def deflection_vec_2d(s, threshold=.3):
    ys, xs = np.where(s > threshold)
    if len(xs) < 4:
        return None
    w = s[ys, xs]
    center = np.average(np.c_[xs, ys], axis=0, weights=w)
    core = np.c_[xs, ys][w >= np.percentile(w, 90)].mean(0)
    v = center - core
    return v if np.linalg.norm(v) > 0 else None


def centroid3d(mask, depth, scale):
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    z = depth[ys, xs] * scale
    return np.array([((xs - CX) * z / 640.0).mean(), z.mean()]), (ys, xs)


def projection_stats(new_pts_rel, v):
    """new_pts_rel: (N,2) positions relative to current centroid; v: (2,) direction."""
    if new_pts_rel is None or v is None or len(new_pts_rel) < 10:
        return None, None
    n = np.linalg.norm(v)
    if n == 0:
        return None, None
    proj = new_pts_rel @ (v / n)
    return float(np.median(proj)), float((proj > 0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--t80', type=float, default=8800.0)
    p.add_argument('--delta', type=int, default=10)
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
        tiffs = [load_tiff(a.ig / r['thermal_tiff_file_name']) for r in rs]
        masks = [load_mask(a.ig / r['mask_80c_file_name']) for r in rs]
        depths = [load_depth(a.ig / 'rgb_only' / 'depth_cache' / sid / (r['aligned_id'] + '.png')) for r in rs]
        recs = []
        for k in range(len(rs) - a.delta):
            j = k + a.delta
            if masks[k].sum() < a.min_area:
                continue
            c_old = centroid3d(masks[k], depths[k], a.scale)
            if c_old[0] is None:
                continue
            # predictors at frame k
            s_static = field_from_tiff(tiffs[k], a.t80)
            v3_static = deflection_vec_3d(backproject_field(depths[k], s_static, fx=640., fy=640.,
                                                            cx=CX, cy=CY, scale=a.scale, threshold=.05, topk=20000))
            v2_static = deflection_vec_2d(s_static)
            v3_dt = v2_dt = None
            if k >= 5:
                dT = tiffs[k] - tiffs[k - 5]
                pos = dT[dT > 0]
                if pos.size >= 100:
                    s_dt = np.clip(dT / np.percentile(pos, 99), 0, 1).astype(np.float32)
                    v3_dt = deflection_vec_3d(backproject_field(depths[k], s_dt, fx=640., fy=640.,
                                                                cx=CX, cy=CY, scale=a.scale, threshold=.3, topk=20000))
                    v2_dt = deflection_vec_2d(s_dt)
            # truth: new fire pixels at k+delta, raw and translation-stabilized
            grown = binary_dilation(masks[k], iterations=3) if binary_dilation else masks[k]
            variants = {}
            for tag, stabilize in (('raw', False), ('stab', True)):
                if stabilize:
                    med = np.median(tiffs[k])
                    ta = np.where(tiffs[k] > a.t80, med, tiffs[k])
                    tb = np.where(tiffs[j] > a.t80, np.median(tiffs[j]), tiffs[j])
                    dy, dx = phase_shift(ta, tb)
                    mj = shift_image(masks[j].view(np.uint8), dy, dx) > 0
                    dj = shift_image(depths[j], dy, dx, fill=1.0)
                else:
                    mj, dj = masks[j], depths[j]
                new = mj & ~grown
                if new.sum() < a.min_area:
                    variants[tag] = None
                    continue
                ys, xs = np.where(new)
                z = dj[ys, xs] * a.scale
                pts3 = np.c_[(xs - CX) * z / 640.0, z] - c_old[0][None, :]
                pts2 = np.c_[xs, ys].astype(float) - np.array([(np.where(masks[k])[1]).mean(), (np.where(masks[k])[0]).mean()])[None, :]
                variants[tag] = (pts3, pts2, int(new.sum()))
            rec = {'t': rs[k]['video_time_s']}
            for tag in ('raw', 'stab'):
                pack = variants[tag]
                for pname, v3, v2 in (('static', v3_static, v2_static), ('dt', v3_dt, v2_dt)):
                    if pack is None:
                        rec[f'{tag}_{pname}_3d_med'] = rec[f'{tag}_{pname}_3d_frac'] = None
                        rec[f'{tag}_{pname}_2d_med'] = rec[f'{tag}_{pname}_2d_frac'] = None
                        continue
                    pts3, pts2, na = pack
                    m3, f3 = projection_stats(pts3, v3)
                    m2, f2 = projection_stats(pts2, v2)
                    rec[f'{tag}_{pname}_3d_med'], rec[f'{tag}_{pname}_3d_frac'] = m3, f3
                    rec[f'{tag}_{pname}_2d_med'], rec[f'{tag}_{pname}_2d_frac'] = m2, f2
                rec[f'{tag}_new_area'] = pack[2] if pack else 0
            recs.append(rec)
        keys = [k_ for k_ in recs[0] if k_.endswith(('_med', '_frac'))] if recs else []
        np.save(a.out / f'{sid}_direct_v4.npy',
                np.array([[r['t']] + [r[k_] if r[k_] is not None else np.nan for k_ in keys] for r in recs]))
        s = {'columns': ['t'] + keys, 'pairs': len(recs)}
        for k_ in keys:
            vals = np.array([r[k_] for r in recs if r[k_] is not None])
            if not len(vals):
                s[k_] = None
                continue
            if k_.endswith('_med'):
                s[k_] = {'n': int(len(vals)), 'median': float(np.median(vals)),
                         'mean': float(np.mean(vals)), 'frac_pairs_positive': float((vals > 0).mean())}
            else:
                s[k_] = {'n': int(len(vals)), 'mean': float(np.mean(vals))}
        summary[sid] = s
        brief = {k_: (round(v_['median'], 3), round(v_['frac_pairs_positive'], 3)) for k_, v_ in s.items()
                 if k_.endswith('_med') and v_}
        print(f'[IGNITE-V4] {sid} (median_proj, frac_pairs_pos): ' + json.dumps(brief), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('[IGNITE-V4] done ->', a.out)


if __name__ == '__main__':
    main()
