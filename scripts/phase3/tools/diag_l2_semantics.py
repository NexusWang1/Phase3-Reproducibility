"""L2 v3 semantics drift diagnostic (READ-ONLY; no training/eval state touched).

Hypotheses under test:
  H1 25% letterbox padding (depth=1.0, frame max) collapses top quantile
     boundaries -> padding lands in a middle stratum, last stratum empty.
  H2 Per-stratum z-score is scale-invariant: absolute edge co-occurrence
     magnitude is destroyed, so weak-co-occurrence hard negatives no longer
     produce LOW L2 (they sit at stratum mean -> sigmoid(0)=0.5).
  H3 Near-constant strata inflate z via the var+1e-6 floor -> saturation.

Parts:
  A installed-impl check (is v3 actually live?)
  B cue-level stats on RGBT val / M3FD val samples (+ GT-region reference)
  C FP-box level: legacy L2 vs v3 L2 box means on the 155 baseline FP boxes
  D checkpoint trajectory: M3FD val full-frame p_ddm for best.pt vs last.pt

Run (server):
  python -u "$PKG/tools/diag_l2_semantics.py" \
    --weights "$W" --phase2-package "$P2" \
    --rgbt-root "$RGBT" --rgbt-cache "$RGBT/depth_cache" --m3fd-root "$M3FD" \
    --ddm-run "$RUNS/ddm_v3_train_20261001" \
    --fp-jsonl "$RUNS/ddm_v2_eval_A_detections3d.jsonl" \
    --device cuda:2 --out "$RUNS/l2_semantics_diag_20261001"
"""
from __future__ import annotations
import argparse, inspect, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def stratum_stats(depth, k=8):
    """Return per-frame stratum occupancy + padding stratum, using the exact
    v3 partition rule (half-open, right=False). depth: 2D (H, W) tensor."""
    d = depth.flatten().float()
    q = torch.quantile(d, torch.linspace(0, 1, k + 1, device=d.device))
    inner = q[1:-1].contiguous()  # (k-1,)
    j = torch.searchsorted(inner, d.contiguous(), right=False)
    occ = torch.bincount(j, minlength=k).float() / j.numel()
    pad_mask = d == 1.0
    pad_strata = torch.bincount(j[pad_mask], minlength=k).float() / pad_mask.sum().clamp_min(1)
    return occ.cpu().numpy(), pad_strata.cpu().numpy(), q.cpu().numpy()


def cue_maps(cue, rgb, depth):
    with torch.no_grad():
        l1, l2, _ = cue(rgb, depth)
    return l1, l2


def zstats(rgb, depth, k=8):
    """Replicate v3 L2 internals: e, per-stratum z, saturation, plus legacy L2."""
    g = (0.2989 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])

    def sobel(x):
        kk = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        gx = F.conv2d(x, kk, padding=1)
        gy = F.conv2d(x, kk.transpose(-1, -2), padding=1)
        return (gx.square() + gy.square() + 1e-8).sqrt()

    gi, gd = sobel(g), sobel(depth)
    ei = torch.sigmoid((gi - gi.mean((-2, -1), keepdim=True)) / gi.std((-2, -1), keepdim=True).clamp_min(1e-6))
    ed = torch.sigmoid((gd - gd.mean((-2, -1), keepdim=True)) / gd.std((-2, -1), keepdim=True).clamp_min(1e-6))
    e = ei * ed
    q = torch.quantile(depth.flatten(2), torch.linspace(0, 1, k + 1, device=depth.device), dim=2).squeeze(-1).permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    j_idx = torch.searchsorted(q[:, 1:-1, 0, 0].contiguous(), depth.flatten(2).squeeze(1).contiguous(), right=False)
    sat, e_std = [], []
    legacy = torch.zeros_like(e)
    for jj in range(k):
        mask = (j_idx == jj).view_as(e).to(e.dtype)
        n = mask.sum()
        if n < 2:
            continue
        mu = (e * mask).sum() / n
        var = ((e - mu).square() * mask).sum() / n
        z = (e - mu) / torch.sqrt(var + 1e-6)
        sat.append(((z.abs() > 4).float() * mask).sum() / n)
        e_std.append(var.sqrt())
        legacy = legacy + e * mask  # half-open single coverage == v3 partition
    sat = torch.stack(sat).mean().item() if sat else 0.0
    e_std_med = torch.stack(e_std).median().item() if e_std else 0.0
    # legacy closed-interval L2 (v2 semantics): sum over ALL closed bins / k
    l2_closed = sum(
        (e * ((depth >= q[:, jj:jj + 1]) & (depth <= q[:, jj + 1:jj + 2])).float())
        for jj in range(k)) / k
    return sat, e_std_med, e, l2_closed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--phase2-package', type=Path, required=True)
    p.add_argument('--rgbt-root', type=Path, required=True)
    p.add_argument('--rgbt-cache', type=Path, required=True)
    p.add_argument('--m3fd-root', type=Path, required=True)
    p.add_argument('--ddm-run', type=Path, required=True)
    p.add_argument('--fp-jsonl', type=Path, default=None)
    p.add_argument('--n-frames', type=int, default=120)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    pkg = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(pkg))
    sys.path.insert(0, str(a.phase2_package))
    from tools.ddm_dataset import RGBTDDM, M3FDDM, cache_path
    from tools.ddm_geometry import rgb_letterboxed, depth_letterboxed, letterbox_transform
    from ultralytics.nn.modules.ddm_cues import DDM3Cue as CueInstalled
    from tools.ddm_cues_v2_legacy import DDM3Cue as CueLegacy

    dev = torch.device(a.device)
    report = {}

    # ---- A. installed impl check ----
    src = inspect.getsource(CueInstalled.forward)
    report['A_installed_impl'] = 'v3_halfopen_zscore' if 'searchsorted' in src else 'LEGACY_CLOSED_INTERVAL'
    print('[DIAG-A] installed forward uses:', report['A_installed_impl'])

    cue_new = CueInstalled().to(dev).eval()   # L1/L2 are parameter-free
    cue_old = CueLegacy().to(dev).eval()

    # ---- B. cue-level stats ----
    def scan(ds, tag):
        idx = np.linspace(0, len(ds) - 1, min(a.n_frames, len(ds))).astype(int)
        agg = defaultdict(list)
        for i in idx:
            b = ds[int(i)]
            rgb = b['rgb'][None].to(dev); depth = b['depth'][None].to(dev)
            if depth.ndim == 3:
                depth = depth[:, None]
            _, l2v3 = cue_maps(cue_new, rgb, depth)
            _, l2old = cue_maps(cue_old, rgb, depth)
            occ, pad_s, q = stratum_stats(depth[0, 0].cpu())
            sat, e_std, e, l2_closed = zstats(rgb, depth)
            agg['occ'].append(occ)
            agg['pad_strata'].append(pad_s)
            agg['l2_v3_mean'].append(float(l2v3.mean()))
            agg['l2_legacy_mean'].append(float(l2old.mean()))
            agg['sat_frac'].append(sat)
            agg['e_std_median'].append(e_std)
            pv = np.corrcoef(l2v3.flatten().cpu().numpy(), l2old.flatten().cpu().numpy())[0, 1]
            agg['pearson_v3_legacy'].append(float(pv))
            t = b.get('target')
            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.sum() > 0:
                m = t[None, None].to(dev) if t.ndim == 2 else t[None].to(dev)
                m = F.interpolate(m.float(), l2v3.shape[-2:], mode='nearest') > 0
                agg['gt_region_l2_v3'].append(float(l2v3[m].mean()))
                agg['gt_region_l2_legacy'].append(float(l2old[m].mean()))
        occ = np.mean(agg['occ'], axis=0)
        pad_s = np.mean(agg['pad_strata'], axis=0)
        out = {
            'n_frames': len(idx),
            'stratum_occupancy_mean': [round(float(x), 4) for x in occ],
            'empty_strata_frames_frac': round(float(np.mean([np.sum(np.array(o) < 1e-6) for o in agg['occ']])), 3),
            'padding_stratum_dist': [round(float(x), 4) for x in pad_s],
            'l2_v3_mean': round(float(np.mean(agg['l2_v3_mean'])), 4),
            'l2_legacy_mean': round(float(np.mean(agg['l2_legacy_mean'])), 4),
            'z_saturation_frac_mean': round(float(np.mean(agg['sat_frac'])), 4),
            'e_stratum_std_median': round(float(np.median(agg['e_std_median'])), 5),
            'pearson_v3_vs_legacy_mean': round(float(np.mean(agg['pearson_v3_legacy'])), 4),
        }
        if agg['gt_region_l2_v3']:
            out['gt_region_l2_v3'] = round(float(np.mean(agg['gt_region_l2_v3'])), 4)
            out['gt_region_l2_legacy'] = round(float(np.mean(agg['gt_region_l2_legacy'])), 4)
        print(f'[DIAG-B] {tag}:', json.dumps(out))
        return out

    report['B_rgbt_val'] = scan(RGBTDDM(a.rgbt_root, a.rgbt_cache, 'val'), 'rgbt_val')
    report['B_m3fd_val'] = scan(M3FDDM(a.m3fd_root, 'val'), 'm3fd_val')

    # ---- C. FP-box level legacy vs v3 L2 ----
    if a.fp_jsonl and a.fp_jsonl.is_file():
        rows = [json.loads(l) for l in a.fp_jsonl.open(encoding='utf-8')]
        fps = [r for r in rows if r['dataset'] == 'm3fd_hardneg' and r['cls'] in (0, 1)]
        by_img = defaultdict(list)
        for r in fps:
            by_img[r['image']].append(r['box_xyxy'])
        recs = []
        mroot = a.m3fd_root.resolve()
        for img, boxes in sorted(by_img.items()):
            rel = Path(img).resolve().relative_to(mroot)
            x, tr = rgb_letterboxed(mroot / rel, 640)
            d = depth_letterboxed(cache_path(mroot / 'depth_cache', rel), tr, 'cpu', 640)
            rgb = x[None].to(dev); depth = d.to(dev)
            _, l2v3 = cue_maps(cue_new, rgb, depth)
            _, l2old = cue_maps(cue_old, rgb, depth)
            for (x0, y0, x1, y1) in boxes:
                xa, xb = max(0, int(x0)), min(640, int(np.ceil(x1)))
                ya, yb = max(0, int(y0)), min(640, int(np.ceil(y1)))
                if xb - xa < 2 or yb - ya < 2:
                    continue
                recs.append({'img': str(rel), 'l2_v3': float(l2v3[0, 0, ya:yb, xa:xb].mean()),
                             'l2_legacy': float(l2old[0, 0, ya:yb, xa:xb].mean())})
        v3m = np.array([r['l2_v3'] for r in recs]); olm = np.array([r['l2_legacy'] for r in recs])
        report['C_fp_boxes'] = {
            'n_boxes': len(recs), 'n_images': len(by_img),
            'l2_v3_box_median': round(float(np.median(v3m)), 4),
            'l2_legacy_box_median': round(float(np.median(olm)), 4),
            'l2_v3_box_p25_p75': [round(float(x), 4) for x in np.percentile(v3m, [25, 75])],
            'l2_legacy_box_p25_p75': [round(float(x), 4) for x in np.percentile(olm, [25, 75])],
            'v3_box_frac_above_0.4': round(float((v3m > 0.4).mean()), 4),
            'legacy_box_frac_below_0.2': round(float((olm < 0.2).mean()), 4),
        }
        print('[DIAG-C]', json.dumps(report['C_fp_boxes']))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
        bins = np.linspace(0, 1, 41)
        ax.hist(olm, bins=bins, alpha=0.62, color='#1f3b57',
                label=f"v2 legacy L2 (median {np.median(olm):.3f})")
        ax.hist(v3m, bins=bins, alpha=0.62, color='#e76f51',
                label=f"v3 z-score L2 (median {np.median(v3m):.3f})")
        ax.set_xlabel('FP box-mean L2 (155 baseline hard-neg boxes)')
        ax.set_ylabel('boxes'); ax.legend(frameon=False); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(a.out / 'fp_box_l2_hist.png', facecolor='white')
        with (a.out / 'fp_box_l2_perbox.json').open('w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
    else:
        report['C_fp_boxes'] = 'skipped: no --fp-jsonl'

    # ---- D. checkpoint trajectory on M3FD val ----
    from smoke_frp_v2_7 import make
    from ultralytics.nn.modules.ddm_branch import DDMBranch
    model, _, _ = make(Path(a.weights), a.device, True)
    model.eval()
    [z.requires_grad_(False) for z in model.parameters()]
    mv = M3FDDM(a.m3fd_root, 'val')
    loader = torch.utils.data.DataLoader(mv, 8, shuffle=False, num_workers=4, pin_memory=True)

    def pmean(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        cues = CueInstalled().to(dev).eval(); cues.load_state_dict(ck['cues'])
        br = DDMBranch(ck['p3_channels']).to(dev).eval(); br.load_state_dict(ck['branch'])
        vals = []
        with torch.no_grad():
            for b in loader:
                rgb = b['rgb'].to(dev); depth = b['depth'].to(dev)
                with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                    p3 = model._rgb_backbone(rgb)[1]
                pred = br(*cues(rgb, depth), p3.float())
                vals.extend(pred.mean((-2, -1)).flatten().cpu().tolist())
        return np.array(vals)

    p_best = pmean(a.ddm_run / 'best.pt')
    p_last = pmean(a.ddm_run / 'last.pt')
    report['D_ckpt_trajectory'] = {
        'best_epoch': int(torch.load(a.ddm_run / 'best.pt', map_location='cpu')['epoch']),
        'last_epoch': int(torch.load(a.ddm_run / 'last.pt', map_location='cpu')['epoch']),
        'm3fd_val_p_mean_best': round(float(p_best.mean()), 6),
        'm3fd_val_p_mean_last': round(float(p_last.mean()), 6),
        'm3fd_val_p_median_best': round(float(np.median(p_best)), 6),
        'm3fd_val_p_median_last': round(float(np.median(p_last)), 6),
        'p_frac_below_0.005_best': round(float((p_best < 0.005).mean()), 4),
        'p_frac_below_0.005_last': round(float((p_last < 0.005).mean()), 4),
    }
    print('[DIAG-D]', json.dumps(report['D_ckpt_trajectory']))

    with (a.out / 'diag.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('[DIAG] saved:', a.out / 'diag.json')


if __name__ == '__main__':
    main()
