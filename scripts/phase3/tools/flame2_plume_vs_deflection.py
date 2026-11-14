"""FLAME2 烟羽漂移 vs 热场偏转：相机系内物理一致性检验（多 dt 版）。

命题：热场定向偏转方向 = 火焰下风扩散方向。
无需相机航向角的验证设计（FLAME2 MP4 无 GPS 元数据）：
- 烟羽/热羽漂移用"差分质心"估计器：帧 i 与 i+dt 场差分，增强区 G=clip(S1-S0,0,.)
  出现在下风前缘，衰减区 L=clip(S0-S1,0,.) 出现在上风后缘，
  v_drift = wcentroid(G) - wcentroid(L) 即漂移方向（图像坐标）。
- YHT 为 30fps 视频，dt=1 时平流位移远小于像素、差分被火焰闪烁主导（v1 实测
  drift R=0.055 纯噪声），故扫描 dt∈{5,15,30}（0.17/0.5/1.0 秒）。
- 同帧图像系偏转 def2d = 场加权质心 - 前10%强度核心（与 IGNITE v4 同款定义）。
- 3D 偏转（论文 76° 主结果）从 deflection_series.npy 读入对照。
- 起火前段作对照（预期两者都散乱）。

输入：
  --fields-dir  runs/phase3/yht_fields_baseline  （1051 个净化场 npy，帧 13651-14701）
  --deflection-npy  deflection_series.npy（本地上传，列：frame,dX,dY,dZ nominal）
输出：summary.json / plume_vs_deflection_series.npy / fig PNG+PDF

运行：
  PYTHONPATH="$ROOT:$PKG" python -u "$PKG/tools/flame2_plume_vs_deflection.py" \
    --fields-dir "$ROOT/runs/phase3/yht_fields_baseline" \
    --deflection-npy "$ROOT/runs/phase3/deflection_series.npy" \
    --out "$ROOT/runs/phase3/flame2_plume_vs_deflection_v2"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IGN_DEFAULT = 13771
THR = 0.3          # def2d 场阈值（与 ignite v4 一致）
EPS_REL = 1e-3     # 差分能量下限（相对当帧场强总和）
DTS = (5, 15, 30)  # 30fps 视频：0.17s / 0.5s / 1.0s
DT_FIG = 15        # 图用哪个 dt


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


def drift_series(fields, dt):
    """drift[i] = 帧 i -> i+dt 的差分质心漂移向量（图像 px）。"""
    out = np.full((len(fields), 2), np.nan)
    for i in range(len(fields) - dt):
        s0, s1 = fields[i], fields[i + dt]
        G = np.clip(s1 - s0, 0, None)
        L = np.clip(s0 - s1, 0, None)
        ref = max(float(s0.sum()), 1e-9)
        if G.sum() < EPS_REL * ref or L.sum() < EPS_REL * ref:
            continue
        cg, cl = wcentroid(G), wcentroid(L)
        if cg is None or cl is None:
            continue
        out[i] = cg - cl
    return out


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

    d3 = np.load(a.deflection_npy)  # (N,4): frame, dX, dY, dZ
    frames = d3[:, 0].astype(int)
    cand = sorted(a.fields_dir.glob('*.npy')) or sorted((a.fields_dir / 'fields').glob('*.npy'))
    by_id = {int(f.stem): f for f in cand}
    missing = [int(fr) for fr in frames if int(fr) not in by_id]
    assert not missing, f'缺 {len(missing)} 帧场文件，如 {missing[:3]}'
    files = [by_id[int(fr)] for fr in frames]
    print(f'[XC] frames {frames[0]}..{frames[-1]}, n={len(frames)}', flush=True)

    def2d = np.full((len(files), 2), np.nan)
    fields = []
    for i, f in enumerate(files):
        s = np.nan_to_num(np.load(f).astype(np.float32).squeeze())
        fields.append(s)
        v = deflection_2d(s)
        if v is not None:
            def2d[i] = v

    post = frames >= a.ign_frame
    ang = lambda v: np.degrees(np.arctan2(v[:, 0], -v[:, 1]))  # 0=上,90=右

    by_dt = {}
    store = {}
    for dt in DTS:
        drift = median_smooth(drift_series(fields, dt), a.smooth)
        store[dt] = drift
        valid = post & ~np.isnan(drift[:, 0]) & ~np.isnan(def2d[:, 0])
        valid_pre = (~post) & ~np.isnan(drift[:, 0]) & ~np.isnan(def2d[:, 0])
        diff = np.abs(circ_diff(ang(drift[valid]), ang(def2d[valid])))
        seg = {}
        for tag, mask in (('post', valid), ('pre', valid_pre)):
            if mask.sum() < 10:
                continue
            a_drift, R_drift = resultant(drift[mask])
            a_def, R_def = resultant(def2d[mask])
            a_3d, R_3d = resultant(np.c_[d3[mask, 1], -d3[mask, 3]])
            seg[tag] = {'n_pairs': int(mask.sum()),
                        'drift_resultant_angle_deg': a_drift, 'drift_R': R_drift,
                        'def2d_resultant_angle_deg': a_def, 'def2d_R': R_def,
                        'def3d_resultant_angle_img_deg': a_3d, 'def3d_R': R_3d}
        by_dt[str(dt)] = {
            'seconds': round(dt / 30, 3),
            'post_angle_diff_median_deg': float(np.median(diff)) if len(diff) else None,
            'post_frac_within_45deg': float((diff <= 45).mean()) if len(diff) else None,
            'segments': seg,
        }
        print(f'[XC] dt={dt}:', json.dumps(by_dt[str(dt)])[:400], flush=True)

    summary = {
        'frames': [int(frames[0]), int(frames[-1])], 'ign_frame': a.ign_frame,
        'by_dt': by_dt,
        'metar_context': 'KGCN 2021-11-19 16:54-23:54 UTC: wind from 200-230 deg (SW), 8-12 kt, gust 16-18 kt',
    }
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.save(a.out / 'plume_vs_deflection_series.npy',
            np.c_[frames, store[DTS[0]], store[DTS[1]], store[DTS[2]], def2d, d3[:, 1:]])

    # ---- 图（全英文）----
    drift = store[DT_FIG]
    valid = post & ~np.isnan(drift[:, 0]) & ~np.isnan(def2d[:, 0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'FLAME2 plume drift vs thermal-field deflection (camera frame, dt={DT_FIG} fr = {DT_FIG / 30:.1f}s)\n'
                 'METAR: SW wind 210-230°, 8-12 kt during burn', fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.axvspan(frames[0], a.ign_frame, color='gray', alpha=.12)
    ax.scatter(frames[valid], ang(def2d[valid]), s=4, c='#7570b3', alpha=.5, linewidths=0, label='deflection 2D (field)')
    ax.scatter(frames[valid], ang(drift[valid]), s=4, c='#1b9e77', alpha=.5, linewidths=0, label='plume drift (diff-centroid)')
    ax.axvline(a.ign_frame, color='k', ls='--', lw=1)
    ax.set_ylim(-180, 180); ax.set_xlabel('frame'); ax.set_ylabel('image angle (deg, 0=up, 90=right)')
    ax.set_title('(a) Per-frame directions'); ax.legend(frameon=False, fontsize=9, loc='lower right')

    ax = axes[0, 1]
    xs = np.arange(len(DTS)); wdt = .35
    frac = [by_dt[str(d)]['post_frac_within_45deg'] for d in DTS]
    med = [by_dt[str(d)]['post_angle_diff_median_deg'] for d in DTS]
    ax.bar(xs - wdt / 2, [f * 100 if f is not None else 0 for f in frac], width=wdt,
           color='#1b9e77', label='pairs within ±45° (%)')
    ax.bar(xs + wdt / 2, [m if m is not None else 0 for m in med], width=wdt,
           color='#d95f02', label='median |angle diff| (deg)')
    ax.set_xticks(xs); ax.set_xticklabels([f'dt={d}\n({d / 30:.2f}s)' for d in DTS])
    ax.axhline(50, color='gray', ls=':', lw=1)
    ax.set_title('(b) Agreement vs time span'); ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    vd = drift[valid]; ve = def2d[valid]
    sc = max(np.nanpercentile(np.linalg.norm(ve, axis=1), 95), 1e-6)
    ax.hist2d(ve[:, 0] / sc, ve[:, 1] / sc, bins=40, range=[[-1.2, 1.2], [-1.2, 1.2]], cmap='inferno')
    ax.arrow(0, 0, * (ve.mean(0) / sc), color='#7570b3', width=.02, head_width=.07,
             length_includes_head=True, zorder=5, label='deflection 2D mean')
    ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
    ax.set_xlabel('x (right, normalized)'); ax.set_ylabel('y (down, normalized)')
    ax.set_title('(c) Post-ignition deflection endpoints'); ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    bins = np.linspace(-180, 180, 37)
    ax.hist(ang(def2d[valid]), bins=bins, alpha=.55, color='#7570b3', label='deflection 2D')
    ax.hist(ang(drift[valid]), bins=bins, alpha=.55, color='#1b9e77', label='plume drift')
    ax.set_xlabel('image angle (deg, 0=up, 90=right)'); ax.set_ylabel('frame pairs')
    ax.set_title(f'(d) Direction histograms, post-ignition (dt={DT_FIG})')
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(a.out / 'flame2_plume_vs_deflection.png', bbox_inches='tight', dpi=150)
    fig.savefig(a.out / 'flame2_plume_vs_deflection.pdf', bbox_inches='tight')
    print('[XC] done ->', a.out, flush=True)


if __name__ == '__main__':
    main()
