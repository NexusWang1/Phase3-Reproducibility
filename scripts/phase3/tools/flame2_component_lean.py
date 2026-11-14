"""FLAME2 逐组件倾斜共识：多火点方向一致 = 风向特征（免疫平台运动）。

设计逻辑（用户提出）：
- 每个连通火点的 lean = 组件内加权质心 - 组件内 top-10% 核心（形状内禀属性）。
  相机整体平移不会让火点"变歪" → 平台运动干扰路径被结构性堵死。
- 燃烧脉动是组件级局部噪声（方向随机）；风是全局相干场。
  ~236 组件/帧若一致倾向同一方向，该共同方向即下风方向。
- 同时是对全局偏转（76°/R=0.83）的分布式验证：排除"被单个大团块主导"。

输出：summary.json + component_lean_series.npy + 四联图 PNG/PDF。

运行：
  PYTHONPATH="$ROOT:$PKG" python -u "$PKG/tools/flame2_component_lean.py" \
    --fields-dir "$ROOT/runs/phase3/yht_fields_baseline" \
    --deflection-npy "$ROOT/runs/phase3/deflection_series.npy" \
    --out "$ROOT/runs/phase3/flame2_component_lean_v1"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import label
except ImportError:
    raise SystemExit('需要 scipy（服务器环境已有）')

IGN_DEFAULT = 13771
THR = 0.05       # 组件切分阈值（对齐 dir_eval 口径，该口径下平均 236 组件/帧）
MIN_AREA = 8     # 组件最小像素数（再小 lean 无意义）


def component_leans(s, thr=THR, min_area=MIN_AREA):
    """逐组件 lean 向量 + 组件功率。返回 (vecs (M,2), powers (M,))。"""
    marks, n = label(s > thr, structure=np.ones((3, 3), np.uint8))
    vecs, powers = [], []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(marks == k)
        if len(xs) < min_area:
            continue
        w = s[ys, xs]
        center = np.array([np.average(xs, weights=w), np.average(ys, weights=w)])
        core = np.c_[xs, ys][w >= np.percentile(w, 90)].mean(0)
        v = center - core
        nv = np.linalg.norm(v)
        if nv > 0.15:  # lean 至少 0.15px，排除近圆对称组件的方向噪声
            vecs.append(v / nv)
            powers.append(float(w.sum()))
    return np.array(vecs) if vecs else np.zeros((0, 2)), np.array(powers)


def resultant(vecs, weights=None):
    if len(vecs) < 3:
        return None, 0.0
    w = np.ones(len(vecs)) if weights is None else weights
    mx = np.average(vecs[:, 0], weights=w)
    my = np.average(vecs[:, 1], weights=w)
    return float(np.degrees(np.arctan2(mx, -my))), float(np.hypot(mx, my))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fields-dir', type=Path, required=True)
    p.add_argument('--deflection-npy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ign-frame', type=int, default=IGN_DEFAULT)
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
    print(f'[LEAN] frames {frames[0]}..{frames[-1]}, n={len(frames)}', flush=True)

    post = frames >= a.ign_frame
    per_frame_ang = np.full(len(files), np.nan)   # 每帧组件共识方向
    per_frame_R = np.full(len(files), np.nan)     # 每帧共识强度
    per_frame_n = np.zeros(len(files), int)       # 每帧有效组件数
    all_ang_post, all_ang_pre = [], []            # 逐组件角度池（加权合成用）
    all_w_post, all_w_pre = [], []

    for i, f in enumerate(files):
        s = np.nan_to_num(np.load(f).astype(np.float32).squeeze())
        vecs, powers = component_leans(s)
        per_frame_n[i] = len(vecs)
        if len(vecs) < 3:
            continue
        ang_f, R_f = resultant(vecs)  # 不加权：每组件一票，防止大团块主导
        per_frame_ang[i], per_frame_R[i] = ang_f, R_f
        (all_ang_post if post[i] else all_ang_pre).append(vecs)
        (all_w_post if post[i] else all_w_pre).append(powers)

    print(f'[LEAN] valid comps/frame: median={np.median(per_frame_n):.0f}, '
          f'p10={np.percentile(per_frame_n, 10):.0f}, p90={np.percentile(per_frame_n, 90):.0f}, '
          f'frames>=3: {(per_frame_n >= 3).sum()}/{len(files)}', flush=True)

    def pool(lst_v, lst_w):
        if not lst_v:
            return None, 0.0, 0, 0.0
        v = np.concatenate(lst_v)
        ang, R = resultant(v)  # 组件等权
        _, Rw = resultant(v, np.concatenate(lst_w))  # 功率加权参照
        return ang, R, len(v), Rw

    ang_post, R_post, n_post, Rw_post = pool(all_ang_post, all_w_post)
    ang_pre, R_pre, n_pre, Rw_pre = pool(all_ang_pre, all_w_pre)

    # 全局参照（官方序列起火后均值方向，图像约定）
    a3_post = float(np.degrees(np.arctan2(d3[post, 1].mean(), -d3[post, 3].mean())))
    summary = {
        'frames': [int(frames[0]), int(frames[-1])], 'ign_frame': a.ign_frame,
        'thr': THR, 'min_area': MIN_AREA,
        'post': {'n_components': n_post, 'consensus_angle_deg': ang_post,
                 'consensus_R_equal_weight': R_post, 'consensus_R_power_weighted': Rw_post,
                 'per_frame_R_median': float(np.nanmedian(per_frame_R[post])),
                 'per_frame_R_p25': float(np.nanpercentile(per_frame_R[post], 25)),
                 'per_frame_n_components_median': float(np.median(per_frame_n[post]))},
        'pre': {'n_components': n_pre, 'consensus_angle_deg': ang_pre,
                'consensus_R_equal_weight': R_pre,
                'per_frame_R_median': float(np.nanmedian(per_frame_R[~post]))},
        'global_deflection_reference_post': {'angle_deg': a3_post,
                                             'note': '官方 deflection_series 起火后均值方向（图像约定）'},
        'metar_context': 'KGCN 2021-11-19 16:54-23:54 UTC: wind from 200-230 deg (SW), 8-12 kt, gust 16-18 kt',
    }
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.save(a.out / 'component_lean_series.npy',
            np.c_[frames, per_frame_ang, per_frame_R, per_frame_n])
    print('[LEAN] summary:', json.dumps(summary, indent=2)[:900], flush=True)

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('FLAME2 per-component lean consensus (platform-motion immune) · '
                 'METAR: SW wind 210-230°, 8-12 kt', fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.axvspan(frames[0], a.ign_frame, color='gray', alpha=.12)
    ax.plot(frames, per_frame_R, lw=.8, c='#1b9e77')
    ax.axvline(a.ign_frame, color='k', ls='--', lw=1)
    ax.set_xlabel('frame'); ax.set_ylabel('per-frame consensus R')
    ax.set_title(f'(a) Consensus strength over time (median post={summary["post"]["per_frame_R_median"]:.2f})')

    ax = axes[0, 1]
    ax.axvspan(frames[0], a.ign_frame, color='gray', alpha=.12)
    ax.scatter(frames, per_frame_ang, s=5, c='#7570b3', alpha=.5, linewidths=0)
    ax.axvline(a.ign_frame, color='k', ls='--', lw=1)
    ax.set_ylim(-180, 180); ax.set_xlabel('frame'); ax.set_ylabel('consensus angle (deg)')
    ax.set_title(f'(b) Per-frame consensus direction (post={ang_post:.0f}°)' if ang_post is not None else '(b)')

    ax = axes[1, 0]
    vp = np.concatenate(all_ang_post) if all_ang_post else np.zeros((0, 2))
    vq = np.concatenate(all_ang_pre) if all_ang_pre else np.zeros((0, 2))
    bins = np.linspace(-180, 180, 37)
    angp = np.degrees(np.arctan2(vp[:, 0], -vp[:, 1]))
    ax.hist(angp, bins=bins, color='#1b9e77', alpha=.75)
    if ang_post is not None:
        ax.axvline(ang_post, color='red', lw=2, label=f'consensus = {ang_post:.0f}° (R={R_post:.2f})')
    ax.set_xlabel('component lean angle (deg, 0=up, 90=right)'); ax.set_ylabel('components')
    ax.set_title(f'(c) Post-ignition component lean distribution (n={n_post})')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    if len(vq):
        angq = np.degrees(np.arctan2(vq[:, 0], -vq[:, 1]))
        ax.hist(angq, bins=bins, color='gray', alpha=.75)
        if ang_pre is not None:
            ax.axvline(ang_pre, color='red', lw=2, label=f'consensus = {ang_pre:.0f}° (R={R_pre:.2f})')
    ax.set_xlabel('component lean angle (deg)'); ax.set_ylabel('components')
    ax.set_title(f'(d) Pre-ignition control (n={n_pre}) — expect scattered')
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(a.out / 'flame2_component_lean.png', bbox_inches='tight', dpi=150)
    fig.savefig(a.out / 'flame2_component_lean.pdf', bbox_inches='tight')
    print('[LEAN] done ->', a.out, flush=True)


if __name__ == '__main__':
    main()
