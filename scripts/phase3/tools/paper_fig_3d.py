"""Publication 3-D field figures for the Phase3 paper (analysis-only; no model changes).

For each requested YHT frame renders a 2x2 figure:
  (a) IR + purified field S' overlay with the *apparent* 2-D centroid->core arrow
  (b) 3-D back-projected point cloud (camera coordinates, meters), colored by strength
  (c) ground-plane (X-Z) projection with the 3-D deflection vector
  (d) depth-layered ground-plane view: per-layer deflection arrows (near/mid/far)
Optionally (--timeline) also renders the deflection angle / coherence R time series
with the ignition frame marked. Saves PNG (300 dpi) + PDF per figure.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'axes.linewidth': .8, 'mathtext.fontset': 'dejavuserif'})


def _frame_id(path):
    import re
    n = re.findall(r'\d+', Path(path).stem)
    return int(n[-1]) if n else None


def _images(root):
    return sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})


def _vector_3d(field, depth640, backproject):
    """Identical math to tools/demo_phase3.py (mask/list indexing fixed)."""
    s = F.interpolate(torch.from_numpy(field)[None, None], (640, 640), mode='bilinear', align_corners=False)[0, 0].numpy()
    pts = backproject(depth640, s, threshold=.05, topk=10000)
    if len(pts) < 4:
        return pts, np.zeros(2)
    w = pts[:, 3]
    center = np.average(pts[:, [0, 2]], axis=0, weights=w)
    core = pts[w >= np.percentile(w, 90)][:, [0, 2]].mean(0)
    return pts, center - core


def _vector_2d(field):
    ys, xs = np.where(field > .05)
    if len(xs) < 4:
        return np.zeros(2), np.zeros(2), np.zeros(2)
    w = field[ys, xs]
    center = np.array([np.average(xs, weights=w), np.average(ys, weights=w)])
    m = w >= np.percentile(w, 90)
    core = np.array([xs[m].mean(), ys[m].mean()])
    return center - core, center, core


def _angle_r(history, vec):
    units = np.asarray([v / np.linalg.norm(v) for v in history if np.linalg.norm(v)])
    R = float(np.linalg.norm(units.mean(0))) if len(units) else 0.
    ang = float(np.degrees(np.arctan2(vec[1], vec[0]))) if np.linalg.norm(vec) else 0.
    return ang, R


def _smooth(H):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(H, 1.2)
    except Exception:
        return H


def _stacked_layers(axb, pts, vec, ang, n_layers=12):
    """Flame as horizontally stacked X-Z slices (layered volumetric look)."""
    hgt = -pts[:, 1]
    w = pts[:, 3]
    hmin, hmax = np.percentile(hgt, [2, 98])
    xr = np.percentile(pts[:, 0], [1, 99])
    zr = np.percentile(pts[:, 2], [1, 99])
    gx = np.linspace(xr[0], xr[1], 61)
    gz = np.linspace(zr[0], zr[1], 61)
    cx = .5 * (gx[:-1] + gx[1:]); cz = .5 * (gz[:-1] + gz[1:])
    edges = np.linspace(hmin, hmax, n_layers + 1)
    layers = []; gmax = 0.
    for i in range(n_layers):
        m = (hgt >= edges[i]) & (hgt < edges[i + 1])
        if m.sum() < 3:
            continue
        H, _, _ = np.histogram2d(pts[m, 0], pts[m, 2], bins=[gx, gz], weights=w[m])
        H = _smooth(H)
        if H.max() > 0:
            layers.append((.5 * (edges[i] + edges[i + 1]), H.T)); gmax = max(gmax, float(H.max()))
    for hc, G in layers:
        Gm = np.ma.masked_less(G, gmax * .04)
        axb.contourf(cx, cz, Gm, zdir='z', offset=hc, cmap='inferno',
                     vmin=0, vmax=gmax, alpha=.92, levels=14)
    if np.linalg.norm(vec):
        bx0 = float(np.average(pts[:, 0], weights=w))
        bz0 = float(np.average(pts[:, 2], weights=w))
        hy = float(np.clip(-np.average(pts[:, 1], weights=w), hmin, hmax))
        axb.quiver(bx0, bz0, hmin, vec[0] * 3, vec[1] * 3, 0, color='cyan', linewidth=2.5,
                   arrow_length_ratio=.15, label=f'deflection {ang:.1f}°')
        axb.legend(loc='upper left', frameon=False)
    axb.set_xlim(xr); axb.set_ylim(zr); axb.set_zlim(hmin, hmax)
    from matplotlib.ticker import MaxNLocator
    axb.yaxis.set_major_locator(MaxNLocator(5))
    axb.zaxis.set_major_locator(MaxNLocator(6))
    axb.set_xlabel('X (m)'); axb.set_ylabel('Z — depth (m)'); axb.set_zlabel('Height (m)')
    axb.set_title('(b) Layered 3-D field model (stacked X-Z slices)')
    axb.view_init(elev=ELEV, azim=AZIM)
    try:
        axb.set_box_aspect((1.4, 1.4, max(.6, (hmax - hmin) / max(xr[1] - xr[0], 1e-6) * 2)))
    except Exception:
        pass


def render_threeview(path, frame_id, pts, vec):
    """Engineering three-view: front (X×H), side (Z×H), top (X×Z) density maps."""
    hgt = -pts[:, 1]
    w = pts[:, 3]
    xr = np.percentile(pts[:, 0], [1, 99])
    zr = np.percentile(pts[:, 2], [1, 99])
    hr = np.percentile(hgt, [1, 99])
    bx = np.linspace(xr[0], xr[1], 80); bz = np.linspace(zr[0], zr[1], 80); bh = np.linspace(hr[0], hr[1], 60)
    F_, _, _ = np.histogram2d(pts[:, 0], hgt, bins=[bx, bh], weights=w)      # front: X × H
    S_, _, _ = np.histogram2d(pts[:, 2], hgt, bins=[bz, bh], weights=w)      # side:  Z × H
    T_, _, _ = np.histogram2d(pts[:, 0], pts[:, 2], bins=[bx, bz], weights=w)  # top: X × Z
    F_, S_, T_ = _smooth(F_), _smooth(S_), _smooth(T_)
    vm = max(F_.max(), S_.max(), T_.max()) or 1.
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.5, 1), height_ratios=(1, 1.15),
                          hspace=.08, wspace=.08)
    axf = fig.add_subplot(gs[0, 0])                      # front (camera view)
    axs = fig.add_subplot(gs[0, 1], sharey=axf)          # side (depth profile)
    axt = fig.add_subplot(gs[1, 0], sharex=axf)          # top (ground plane)
    axl = fig.add_subplot(gs[1, 1])                      # legend / notes
    axf.imshow(F_.T, origin='lower', cmap='inferno', vmin=0, vmax=vm,
               extent=(xr[0], xr[1], hr[0], hr[1]), aspect='auto')
    axf.set_ylabel('Height (m)'); axf.set_title('Front view (camera plane, X × H)')
    plt.setp(axf.get_xticklabels(), visible=False)
    axs.imshow(S_.T, origin='lower', cmap='inferno', vmin=0, vmax=vm,
               extent=(zr[0], zr[1], hr[0], hr[1]), aspect='auto')
    axs.set_xlabel('Z — depth (m)'); axs.set_title('Side view (Z × H)')
    plt.setp(axs.get_yticklabels(), visible=False)
    axt.imshow(T_.T, origin='lower', cmap='inferno', vmin=0, vmax=vm,
               extent=(xr[0], xr[1], zr[0], zr[1]), aspect='auto')
    axt.set_xlabel('X (m)'); axt.set_ylabel('Z — depth (m)')
    axt.set_title('Top view (ground plane, X × Z)')
    if np.linalg.norm(vec):
        bx0 = float(np.average(pts[:, 0], weights=w))
        bz0 = float(np.average(pts[:, 2], weights=w))
        axt.arrow(bx0, bz0, vec[0] * 3, vec[1] * 3, color='cyan', width=.25,
                  head_width=1.0, length_includes_head=True, zorder=5)
    axf.set_xlim(xr); axf.set_ylim(hr)
    axs.set_xlim(zr); axs.set_ylim(hr)
    axt.set_xlim(xr); axt.set_ylim(zr)
    ang = float(np.degrees(np.arctan2(vec[1], vec[0]))) if np.linalg.norm(vec) else 0.
    axl.axis('off')
    axl.text(0, .9, f'Frame {frame_id}', fontsize=12, weight='bold')
    axl.text(0, .72, f'points: {len(pts)}', fontsize=10)
    axl.text(0, .62, f'deflection: {ang:.1f}°  |v|: {np.linalg.norm(vec):.2f} m', fontsize=10)
    axl.text(0, .52, 'cyan arrow: 3-D deflection (×3)', fontsize=9, color='teal')
    axl.text(0, .42, 'color: field strength (shared scale)', fontsize=9)
    axl.text(0, .32, f'rendering: top {100 - int(round(RENDER_Q * 100))}% strength points', fontsize=9)
    fig.suptitle(f'Three-view reconstruction — frame {frame_id}', fontsize=12)
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def render_frame(path, frame_id, ir, field, pts, vec):
    h, w = field.shape
    v2, c2, k2 = _vector_2d(field)
    ang = float(np.degrees(np.arctan2(vec[1], vec[0]))) if np.linalg.norm(vec) else 0.
    mag = float(np.linalg.norm(vec))
    fig = plt.figure(figsize=(13, 10))

    # (a) IR + field overlay, apparent 2-D arrow
    axa = fig.add_subplot(2, 2, 1)
    ira = np.asarray(ir, np.float32)
    lo, hi = np.percentile(ira, [1, 99])
    axa.imshow(np.clip((ira - lo) / max(hi - lo, 1e-6), 0, 1), cmap='gray', vmin=0, vmax=1)
    fmax = float(field.max()) or 1.
    axa.imshow(np.ma.masked_less(field, .05), cmap='inferno', alpha=.75,
               extent=(0, ir.width, ir.height, 0), vmin=0, vmax=fmax)
    if np.linalg.norm(v2):
        axa.arrow(k2[0] * ir.width / w, k2[1] * ir.height / h,
                  v2[0] * 3 * ir.width / w, v2[1] * 3 * ir.height / h,
                  color='cyan', width=1.5, head_width=8, length_includes_head=True)
    axa.set_title('(a) Purified field $S\'$ on IR — apparent 2-D deflection')
    axa.axis('off')

    # (b) layered 3-D model
    axb = fig.add_subplot(2, 2, 2, projection='3d')
    if len(pts):
        _stacked_layers(axb, pts, vec, ang)

    # (c) ground-plane projection
    axc = fig.add_subplot(2, 2, 3)
    if len(pts):
        axc.scatter(pts[:, 0], pts[:, 2], c=pts[:, 3], cmap='inferno', s=1.5, alpha=.5, linewidths=0)
        if np.linalg.norm(vec):
            axc.arrow(0, 0, vec[0] * 3, vec[1] * 3, color='red', width=.12,
                      head_width=.8, length_includes_head=True, zorder=5)
    axc.axhline(0, color='gray', lw=.5); axc.axvline(0, color='gray', lw=.5)
    axc.set_xlabel('X (m)'); axc.set_ylabel('Z — depth (m)')
    axc.set_title(f'(c) Ground-plane deflection = {ang:.1f}°, |v| = {mag:.2f} m')
    axc.set_aspect('equal', adjustable='datalim')

    # (d) depth-layered deflection
    axd = fig.add_subplot(2, 2, 4)
    if len(pts):
        q1, q2 = np.percentile(pts[:, 2], [33, 66])
        layers = [('near', pts[:, 2] <= q1, '#1b9e77'),
                  ('mid', (pts[:, 2] > q1) & (pts[:, 2] <= q2), '#d95f02'),
                  ('far', pts[:, 2] > q2, '#7570b3')]
        axd.scatter(pts[:, 0], pts[:, 2], c='lightgray', s=1, alpha=.4, linewidths=0)
        for name, m, color in layers:
            p = pts[m]
            if len(p) < 4:
                continue
            wl = p[:, 3]
            cen = np.average(p[:, [0, 2]], axis=0, weights=wl)
            cor = p[wl >= np.percentile(wl, 90)][:, [0, 2]].mean(0)
            vl = cen - cor
            al = float(np.degrees(np.arctan2(vl[1], vl[0]))) if np.linalg.norm(vl) else 0.
            axd.arrow(cor[0], cor[1], vl[0] * 3, vl[1] * 3, color=color, width=.1,
                      head_width=.7, length_includes_head=True, zorder=5,
                      label=f'{name} ({p[:, 2].min():.0f}–{p[:, 2].max():.0f} m): {al:.1f}°')
        axd.legend(loc='best', frameon=False, fontsize=9)
    axd.axhline(0, color='gray', lw=.5); axd.axvline(0, color='gray', lw=.5)
    axd.set_xlabel('X (m)'); axd.set_ylabel('Z — depth (m)')
    axd.set_title('(d) Depth-layered deflection (near / mid / far)')
    axd.set_aspect('equal', adjustable='datalim')

    fig.suptitle(f'Frame {frame_id}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path.with_suffix('.png'), dpi=300)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    return ang, mag


def render_timeline(path, rows, ignition):
    f = np.array([r['frame'] for r in rows])
    ang = np.array([r['angle_deg'] for r in rows])
    R = np.array([r['R'] for r in rows])
    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    ax1.axvspan(f.min(), ignition, color='gray', alpha=.12)
    ax1.plot(f, ang, color='#d95f02', lw=1.2, label='deflection angle (deg)')
    ax1.axvline(ignition, color='black', ls='--', lw=1)
    ax1.text(ignition, ax1.get_ylim()[1] if ax1.get_ylim()[1] else 180, ' ignition', fontsize=9, va='top')
    ax1.set_xlabel('YHT frame index'); ax1.set_ylabel('Deflection angle (deg)', color='#d95f02')
    ax1.tick_params(axis='y', labelcolor='#d95f02')
    ax2 = ax1.twinx()
    ax2.plot(f, R, color='#1b9e77', lw=1.2, label='coherence R')
    ax2.set_ylabel('Directional coherence R', color='#1b9e77'); ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='#1b9e77')
    ax1.set_title('3-D field deflection around ignition (pre-fire noise → stable post-fire direction)')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=300)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    global ELEV, AZIM, RENDER_Q
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--ddm-ckpt', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--ir-root', type=Path, required=True)
    p.add_argument('--depth-cache', type=Path, required=True)
    p.add_argument('--phase2-package', type=Path,
                   default=Path(__file__).resolve().parents[1].parent / 'FieldGuide_Phase2_FRP_20260916')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tau', type=float, default=.045)
    p.add_argument('--frames', type=int, nargs='+', default=[13700, 13790, 14200])
    p.add_argument('--rgb-files', nargs='*', default=None,
                   help='explicit RGB paths (pair with --ir-files; bypasses numeric frame-id matching, for RGBT-3M etc.)')
    p.add_argument('--ir-files', nargs='*', default=None)
    p.add_argument('--render-q', type=float, default=.35,
                   help='rendering-only strength filter: show points above this quantile (analysis math unchanged)')
    p.add_argument('--timeline', action='store_true')
    p.add_argument('--t-start', type=int, default=13651)
    p.add_argument('--t-end', type=int, default=14701)
    p.add_argument('--t-stride', type=int, default=4)
    p.add_argument('--ignition', type=int, default=13771)
    p.add_argument('--elev', type=float, default=18)
    p.add_argument('--azim', type=float, default=-60)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ELEV, AZIM = a.elev, a.azim
    RENDER_Q = a.render_q
    if a.out.exists():
        raise FileExistsError(f'Refusing overwrite: {a.out}')
    os.environ['PHY_DDM'] = '1'
    os.environ.setdefault('PHY_DDM_VETO_T', '.005')
    os.environ.setdefault('PHY_DDM_VETO_C', '.25')
    os.environ.setdefault('PHY_DDM_FIELD_VETO_T', '0')
    os.environ['PHY_FRP_TAU'] = str(a.tau)
    if not a.phase2_package.is_dir():
        raise FileNotFoundError(a.phase2_package)
    sys.path[:0] = [str(a.phase2_package),
                    str(a.phase2_package.parent / 'FieldGuide_Phase2_FieldUse_20260916'),
                    str(Path(__file__).resolve().parents[1])]
    from smoke_frp_v2_7 import make
    from field_use_common import prepare_six, field_and_detections
    from eval_ddm_val import _cache, _depth_letterboxed
    from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
    from ultralytics.models.yolo.detect.field3d import backproject_field
    nms = __import__('ultralytics.utils.nms', fromlist=['non_max_suppression']).non_max_suppression

    a.out.mkdir(parents=True)
    active = [None]
    model, _, _ = make(a.weights, a.device, True)
    DDMFusionHook(depth_provider=lambda _rgb: active[0]).attach(model, a.ddm_ckpt)

    # Job list: explicit file pairs (--rgb-files/--ir-files) or numeric YHT frame ids.
    if a.rgb_files:
        if len(a.rgb_files) != len(a.ir_files):
            raise ValueError('--rgb-files and --ir-files must have equal length')
        jobs = [(Path(rp).stem, Path(rp), Path(ip)) for rp, ip in zip(a.rgb_files, a.ir_files)]
        rgb = {}
    else:
        rgb = {_frame_id(x): x for x in _images(a.rgb_root) if _frame_id(x) is not None}
        ir = {_frame_id(x): x for x in _images(a.ir_root) if _frame_id(x) is not None}
        jobs = [(f'{n:06d}', rgb[n], ir[n]) for n in a.frames if n in rgb and n in ir]

    def run_frame(name, rgbp, irp):
        dpath = _cache(a.depth_cache, a.rgb_root, rgbp)
        if not Path(irp).is_file() or not dpath.is_file():
            return None
        x, tr = prepare_six(rgbp, irp, torch, a.device)
        active[0] = _depth_letterboxed(dpath, tr, a.device)
        field_and_detections(model, torch, nms, False, x, tr, a.tau, {0, 1, 2})
        pur = getattr(model, '_last_t_pur', None)
        if pur is None:
            raise RuntimeError('purified field tap unavailable')
        field = pur[0, 0].detach().float().cpu().numpy()
        depth640 = active[0][0, 0].detach().float().cpu().numpy()
        return field, *_vector_3d(field, depth640, backproject_field)

    results = []
    for name, rgbp, irp in jobs:
        r = run_frame(name, rgbp, irp)
        if r is None:
            print(f'[PAPER-FIG] {name} missing (ir or depth), skipped', flush=True)
            continue
        field, pts, vec = r
        pts_r = pts[pts[:, 3] >= np.quantile(pts[:, 3], a.render_q)] if len(pts) and a.render_q > 0 else pts
        ang, mag = render_frame(a.out / f'fig3d_frame{name}', name,
                                Image.open(irp).convert('L'), field, pts_r, vec)
        if len(pts_r) >= 4:
            render_threeview(a.out / f'fig3view_frame{name}', name, pts_r, vec)
        results.append({'frame': name, 'angle_deg': ang, 'magnitude_m': mag,
                        'n_points': int(len(pts)), 'n_render': int(len(pts_r)),
                        'render_q': a.render_q})
        print(f'[PAPER-FIG] {name} angle={ang:.1f} mag={mag:.2f}m pts={len(pts)}', flush=True)

    timeline = []
    if a.timeline and rgb:
        history = []
        irx = {_frame_id(x): x for x in _images(a.ir_root) if _frame_id(x) is not None}
        for n in [k for k in sorted(rgb) if a.t_start <= k <= a.t_end and (k - a.t_start) % a.t_stride == 0]:
            if n not in irx:
                continue
            r = run_frame(f'{n:06d}', rgb[n], irx[n])
            if r is None:
                continue
            _, _, vec = r
            history.append(vec)
            history = history[-50:]
            ang, R = _angle_r(history, vec)
            timeline.append({'frame': n, 'angle_deg': ang, 'R': R})
        if timeline:
            render_timeline(a.out / 'fig3d_timeline', timeline, a.ignition)
            print(f'[PAPER-FIG] timeline n={len(timeline)}', flush=True)

    (a.out / 'paper_fig_meta.json').write_text(json.dumps(
        {'frames': results, 'timeline_n': len(timeline), 'ignition': a.ignition}, indent=2))
    print('[PAPER-FIG] ' + json.dumps({'frames_done': len(results), 'timeline_n': len(timeline)}))


if __name__ == '__main__':
    main()
