"""FLAME2 烟羽漂移 vs 热场偏转 v3：羽流区隔离 + 双估计器。

v2 教训：全图差分质心被火核燃烧脉动淹没（起火后 drift R≈0.04 三档 dt 全噪声；
起火前平台漂移 R 随 dt 升至 0.75，证明估计器对相干运动敏感）。
v3 对策：把 top-10% 强度核心（膨胀 3px）从测量区域剔除——燃烧脉动集中于核心，
风致平流发生在羽流外缘。两个独立估计器互为印证：
  A. 掩膜差分质心：G/L 只在羽流区（field>thr 且非核心）计算
  B. 掩膜相位相关：羽流区图像对做相位相关，主峰位移即平流（IGNITE v3 同款，
     合成数据回卷误差为 0 验证过的实现）
dt 取 {15, 30, 60}（0.5s/1s/2s）。

运行：
  PYTHONPATH="$ROOT:$PKG" python -u "$PKG/tools/flame2_plume_vs_deflection_v3.py" \
    --fields-dir "$ROOT/runs/phase3/yht_fields_baseline" \
    --deflection-npy "$ROOT/runs/phase3/deflection_series.npy" \
    --out "$ROOT/runs/phase3/flame2_plume_vs_deflection_v3"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None

IGN_DEFAULT = 13771
THR = 0.3
EPS_REL = 1e-4     # 羽流区能量下限（核心已抠，阈值降一档）
DTS = (15, 30, 60)
DT_FIG = 30


def wcentroid(mag):
    ys, xs = np.nonzero(mag > 0)
    if len(xs) < 4:
        return None
    w = mag[ys, xs]
    return np.array([np.average(xs, weights=w), np.average(ys, weights=w)])


def deflection_2d(s, thr=THR):
    m = s > thr
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    w = s[ys, xs]
    center = np.array([np.average(xs, weights=w), np.average(ys, weights=w)])
    core = np.c_[xs, ys][w >= np.percentile(w, 90)].mean(0)
    v = center - core
    return v if np.linalg.norm(v) > 0 else None


def plume_mask(s, thr=THR):
    """羽流区 = field>thr 且不在（膨胀的）top-10% 核心内。"""
    m = s > thr
    if m.sum() < 8:
        return None
    w = s[m]
    core = m & (s >= np.percentile(w, 90))
    if binary_dilation is not None:
        core = binary_dilation(core, iterations=3)
    pm = m & ~core
    return pm if pm.sum() >= 8 else None


def drift_diffcentroid(s0, s1, pm):
    """估计器 A：羽流区差分质心。"""
    G = np.clip(s1 - s0, 0, None) * pm
    L = np.clip(s0 - s1, 0, None) * pm
    ref = max(float((s0 * pm).sum()), 1e-9)
    if G.sum() < EPS_REL * ref or L.sum() < EPS_REL * ref:
        return None
    cg, cl = wcentroid(G), wcentroid(L)
    return None if cg is None or cl is None else cg - cl


_HANN_CACHE = {}


def drift_phasecorr(s0, s1, pm):
    """估计器 B：羽流区相位相关平移（Hann 窗）。"""
    p0 = (s0 * pm).astype(np.float64)
    p1 = (s1 * pm).astype(np.float64)
    if p0.sum() < 1e-9 or p1.sum() < 1e-9:
        return None
    H, W = p0.shape
    key = (H, W)
    if key not in _HANN_CACHE:
        _HANN_CACHE[key] = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    hw = _HANN_CACHE[key]
    A = np.fft.fft2(p0 * hw)
    B = np.fft.fft2(p1 * hw)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.ifft2(R)
    py, px = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    dy = py - H if py > H // 2 else py
    dx = px - W if px > W // 2 else px
    return np.array([dx, dy], dtype=float)


def median_smooth(v, k):
    h = k // 2
    out = v.copy()
    for i in range(len(v)):
        lo, hi = max(0, i - h), min(len(v), i + h + 1)
        out[i] = np.nanmedian(v[lo:hi], axis=0)
    return out


def circ_diff(a, b):
    return (a - b + 180) % 360 - 180


def resultant(vecs):
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    u = vecs[n.ravel() > 1e-9] / n[n.ravel() > 1e-9]
    if len(u) < 3:
        return None, 0.0
    mx, my = u[:, 0].mean(), u[:, 1].mean()
    return float(np.degrees(np.arctan2(mx, -my))), float(np.hypot(mx, my))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fields-dir', type=Path, required=True)
    p.add_argument('--deflection-npy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ign-frame', type=int, default=IGN_DEFAULT)
    p.add_argument('--smooth', type=int, default=5)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)

    d3 = np.load(a.deflection_npy)
    frames = d3[:, 0].astype(int)
    cand = sorted(a.fields_dir.glob('*.npy')) or sorted((a.fields_dir / 'fields').glob('*.npy'))
    by_id = {int(f.stem): f for f in cand}
    missing = [int(fr) for fr in frames if int(fr) not in by_id]
    assert not missing, f'缺 {len(missing)} 帧场文件，如 {missing[:3]}'
    files = [by_id[int(fr)] for fr in frames]
    print(f'[XC3] frames {frames[0]}..{frames[-1]}, n={len(frames)}', flush=True)

    def2d = np.full((len(files), 2), np.nan)
    fields, masks = [], []
    n_nomask = 0
    for i, f in enumerate(files):
        s = np.nan_to_num(np.load(f).astype(np.float32).squeeze())
        fields.append(s)
        masks.append(plume_mask(s))
        if masks[-1] is None:
            n_nomask += 1
        v = deflection_2d(s)
        if v is not None:
            def2d[i] = v
    print(f'[XC3] plume-mask missing: {n_nomask}/{len(files)}', flush=True)

    post = frames >= a.ign_frame
    ang = lambda v: np.degrees(np.arctan2(v[:, 0], -v[:, 1]))

    by_dt = {}
    store = {}
    for dt in DTS:
        est = {'A': np.full((len(files), 2), np.nan), 'B': np.full((len(files), 2), np.nan)}
        for i in range(len(files) - dt):
            pm = masks[i]
            if pm is None:
                continue
            va = drift_diffcentroid(fields[i], fields[i + dt], pm)
            vb = drift_phasecorr(fields[i], fields[i + dt], pm)
            if va is not None:
                est['A'][i] = va
            if vb is not None:
                est['B'][i] = vb
        est['A'] = median_smooth(est['A'], a.smooth)
        store[dt] = est
        dt_res = {}
        for name in ('A', 'B'):
            drift = est[name]
            valid = post & ~np.isnan(drift[:, 0]) & ~np.isnan(def2d[:, 0])
            valid_pre = (~post) & ~np.isnan(drift[:, 0]) & ~np.isnan(def2d[:, 0])
            diff = np.abs(circ_diff(ang(drift[valid]), ang(def2d[valid])))
            seg = {}
            for tag, mask in (('post', valid), ('pre', valid_pre)):
                if mask.sum() < 10:
                    continue
                a_d, R_d = resultant(drift[mask])
                a_e, R_e = resultant(def2d[mask])
                seg[tag] = {'n_pairs': int(mask.sum()),
                            'drift_resultant_angle_deg': a_d, 'drift_R': R_d,
                            'def2d_resultant_angle_deg': a_e, 'def2d_R': R_e}
            dt_res[name] = {
                'post_angle_diff_median_deg': float(np.median(diff)) if len(diff) else None,
                'post_frac_within_45deg': float((diff <= 45).mean()) if len(diff) else None,
                'segments': seg}
        by_dt[str(dt)] = {'seconds': round(dt / 30, 3), **dt_res}
        print(f'[XC3] dt={dt}:', json.dumps(by_dt[str(dt)])[:600], flush=True)

    summary = {'frames': [int(frames[0]), int(frames[-1])], 'ign_frame': a.ign_frame,
               'estimators': {'A': 'masked diff-centroid', 'B': 'masked phase-correlation'},
               'by_dt': by_dt,
               'metar_context': 'KGCN 2021-11-19 16:54-23:54 UTC: wind from 200-230 deg (SW), 8-12 kt, gust 16-18 kt'}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.save(a.out / 'plume_vs_deflection_series_v3.npy',
            np.c_[frames, store[DTS[0]]['A'], store[DTS[0]]['B'],
                  store[DTS[1]]['A'], store[DTS[1]]['B'],
                  store[DTS[2]]['A'], store[DTS[2]]['B'], def2d, d3[:, 1:]])

    # ---- 图：DT_FIG 档，A/B 两估计器对照 ----
    driftA, driftB = store[DT_FIG]['A'], store[DT_FIG]['B']
    validA = post & ~np.isnan(driftA[:, 0]) & ~np.isnan(def2d[:, 0])
    validB = post & ~np.isnan(driftB[:, 0]) & ~np.isnan(def2d[:, 0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'FLAME2 plume drift (core-masked) vs deflection · dt={DT_FIG} fr = {DT_FIG / 30:.1f}s\n'
                 'METAR: SW wind 210-230°, 8-12 kt during burn', fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.axvspan(frames[0], a.ign_frame, color='gray', alpha=.12)
    ax.scatter(frames[validA], ang(def2d[validA]), s=4, c='#7570b3', alpha=.5, linewidths=0, label='deflection 2D')
    ax.scatter(frames[validA], ang(driftA[validA]), s=4, c='#1b9e77', alpha=.5, linewidths=0, label='drift A: diff-centroid')
    ax.scatter(frames[validB], ang(driftB[validB]), s=4, c='#d95f02', alpha=.4, linewidths=0, label='drift B: phase-corr')
    ax.axvline(a.ign_frame, color='k', ls='--', lw=1)
    ax.set_ylim(-180, 180); ax.set_xlabel('frame'); ax.set_ylabel('image angle (deg, 0=up, 90=right)')
    ax.set_title('(a) Per-frame directions'); ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax = axes[0, 1]
    xs = np.arange(len(DTS)); w = .25
    for k, (name, c) in enumerate((('A', '#1b9e77'), ('B', '#d95f02'))):
        fr = [by_dt[str(d)][name]['post_frac_within_45deg'] or 0 for d in DTS]
        ax.bar(xs + (k - .5) * w, [f * 100 for f in fr], width=w, color=c,
               label=f'drift {name}: pairs within ±45° (%)')
    fr_e = []
    for d in DTS:
        valid = post & ~np.isnan(def2d[:, 0])
        # def2d 与自身 3D 投影的一致度作参照上限
        a3 = np.degrees(np.arctan2(d3[:, 1], -d3[:, 3]))
        dd = np.abs(circ_diff(ang(def2d[valid]), a3[valid]))
        fr_e.append((dd <= 45).mean() * 100)
    ax.bar(xs + w, fr_e, width=w, color='#7570b3', label='def2d vs def3D (reference)')
    ax.set_xticks(xs); ax.set_xticklabels([f'dt={d}\n({d / 30:.2f}s)' for d in DTS])
    ax.set_title('(b) Agreement vs time span'); ax.legend(frameon=False, fontsize=8)

    for k, (drift, valid, name, c) in enumerate(((driftA, validA, 'A: diff-centroid', '#1b9e77'),
                                                 (driftB, validB, 'B: phase-corr', '#d95f02'))):
        ax = axes[1, k]
        bins = np.linspace(-180, 180, 37)
        ax.hist(ang(def2d[valid]), bins=bins, alpha=.55, color='#7570b3', label='deflection 2D')
        ax.hist(ang(drift[valid]), bins=bins, alpha=.55, color=c, label=f'drift {name}')
        ax.set_xlabel('image angle (deg)'); ax.set_ylabel('frame pairs')
        ax.set_title(f'({"cd"[k]}) Histograms, post-ignition, drift {name}')
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(a.out / 'flame2_plume_vs_deflection_v3.png', bbox_inches='tight', dpi=150)
    fig.savefig(a.out / 'flame2_plume_vs_deflection_v3.pdf', bbox_inches='tight')
    print('[XC3] done ->', a.out, flush=True)


if __name__ == '__main__':
    main()
