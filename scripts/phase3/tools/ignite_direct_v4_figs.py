"""IGNITE v4 逐帧过程图：新增火区 + 预测方向箭头 + 投影分布直方图。

放置：FieldGuide_Phase3_DDM3D_20260921/tools/ignite_direct_v4_figs.py
（与 ignite_direct_v4.py 同目录；field3d 相对路径引用方式一致）
运行：
  PYTHONPATH="$ROOT:$PKG" python -u "$PKG/tools/ignite_direct_v4_figs.py" \
    --ig /path/to/IGNITE-fire-dataset \
    --out "$ROOT/runs/phase3/ignite_v4_figs_v1" --topk 3

每序列选 raw_static_2d 口径下"正例且新增面积最大"的 topk 帧对出图。
服务器 matplotlib 无 CJK 字体，本图全部英文标注（论文可直接用）。
统计口径与 ignite_direct_v4.py 完全一致（同函数同参数），图上数字可回溯到 .npy。
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

# 单色 colormap：可见像素一律品红（spring(1)=黄 会融进 hot 底图，勿用 spring）
NEW_CMAP = ListedColormap(['#ff00ff'])

_f3d = Path(__file__).resolve().parents[1] / 'ultralytics/models/yolo/detect/field3d.py'
_spec = importlib.util.spec_from_file_location('phase3_field3d', _f3d)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

W, H = 640, 512
CX, CY = W / 2, H / 2

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None


def load_tiff(path):
    return np.asarray(Image.open(path)).astype(np.float64)


def load_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def field_from_tiff(tiff, t80):
    hi = np.percentile(tiff, 99.5)
    return np.clip((tiff - t80) / max(hi - t80, 1e-9), 0, 1).astype(np.float32)


def deflection_vec_2d(s, threshold=.3):
    ys, xs = np.where(s > threshold)
    if len(xs) < 4:
        return None
    w = s[ys, xs]
    center = np.average(np.c_[xs, ys], axis=0, weights=w)
    core = np.c_[xs, ys][w >= np.percentile(w, 90)].mean(0)
    v = center - core
    return v if np.linalg.norm(v) > 0 else None


def show_thermal(ax, tiff):
    lo, hi = np.percentile(tiff, [1, 99.5])
    ax.imshow(np.clip((tiff - lo) / max(hi - lo, 1e-9), 0, 1), cmap="hot", vmin=0, vmax=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ig', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--t80', type=float, default=8800.0)
    p.add_argument('--delta', type=int, default=10)
    p.add_argument('--min-area', type=int, default=50)
    p.add_argument('--topk', type=int, default=3)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    rows = [json.loads(l) for l in (a.ig / 'metadata.jsonl').open(encoding='utf-8')]
    seqs = {}
    for r in rows:
        seqs.setdefault(r['sequence_id'], []).append(r)

    index = {}
    for sid, rs in sorted(seqs.items()):
        rs.sort(key=lambda r: r['video_time_s'])
        tiffs = [load_tiff(a.ig / r['thermal_tiff_file_name']) for r in rs]
        masks = [load_mask(a.ig / r['mask_80c_file_name']) for r in rs]
        cands = []
        for k in range(len(rs) - a.delta):
            j = k + a.delta
            if masks[k].sum() < a.min_area:
                continue
            s_static = field_from_tiff(tiffs[k], a.t80)
            v2 = deflection_vec_2d(s_static)
            if v2 is None:
                continue
            grown = binary_dilation(masks[k], iterations=3) if binary_dilation else masks[k]
            new = masks[j] & ~grown
            if new.sum() < a.min_area:
                continue
            ys, xs = np.where(new)
            cyx = np.array([np.where(masks[k])[1].mean(), np.where(masks[k])[0].mean()])
            pts2 = np.c_[xs, ys].astype(float) - cyx[None, :]
            proj = pts2 @ (v2 / np.linalg.norm(v2))
            med, frac = float(np.median(proj)), float((proj > 0).mean())
            if frac > 0.5:
                cands.append((int(new.sum()), k, j, v2, new, grown, proj, med, frac))
        cands.sort(key=lambda t: -t[0])
        picks = cands[:a.topk]
        index[sid] = []
        for area, k, j, v2, new, grown, proj, med, frac in picks:
            fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
            t0, t1 = rs[k]['video_time_s'], rs[j]['video_time_s']
            fig.suptitle(f"IGNITE seq {sid} · t={t0:.0f}s -> t+{a.delta}s | static-2D deflection predictor",
                         fontsize=13, fontweight="bold")
            ys_k, xs_k = np.where(masks[k])
            cx2, cy2 = xs_k.mean(), ys_k.mean()
            vn = v2 / np.linalg.norm(v2) * 120  # 固定箭头长度便于读图

            ax = axes[0]
            show_thermal(ax, tiffs[k])
            ax.contour(masks[k], levels=[.5], colors='deepskyblue', linewidths=1.2)
            ax.plot(cx2, cy2, 'o', color='cyan', ms=6)
            ax.arrow(cx2, cy2, vn[0], vn[1], head_width=9, head_length=12, fc='cyan', ec='cyan', lw=2)
            ax.set_title(f"frame k (t={t0:.0f}s): fire mask + predicted direction", fontsize=10)

            ax = axes[1]
            show_thermal(ax, tiffs[j])
            ax.contour(grown, levels=[.5], colors='gray', linewidths=1.0, linestyles='--')
            ax.imshow(np.ma.masked_where(~new, new), cmap=NEW_CMAP, alpha=0.75, vmin=0, vmax=1)
            ys_n, xs_n = np.where(new)
            ax.plot(xs_n.mean(), ys_n.mean(), 'x', color='lime', ms=9, mew=2.5)
            ax.arrow(cx2, cy2, vn[0], vn[1], head_width=9, head_length=12, fc='cyan', ec='cyan', lw=2)
            ax.set_title(f"frame k+{a.delta} (t={t1:.0f}s): NEW fire px (magenta), centroid X", fontsize=10)

            ax = axes[2]
            ax.hist(proj, bins=40, color='steelblue', edgecolor='white', lw=.4)
            ax.axvline(0, color='red', ls='--', lw=1.2)
            ax.axvline(med, color='green', lw=2)
            ax.set_title(f"projection of NEW px onto predicted dir\n"
                         f"n={area}, median={med:+.1f}px, frac>0={frac * 100:.1f}%", fontsize=10)
            ax.set_xlabel("signed projection (px)")
            for a_ in axes[:2]:
                a_.set_xticks([]); a_.set_yticks([])
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fn = f"{sid}_k{k:04d}_j{j:04d}.png"
            fig.savefig(a.out / fn, bbox_inches="tight", dpi=140)
            plt.close(fig)
            index[sid].append({"file": fn, "k": k, "j": j, "t0": t0, "t1": t1,
                               "new_area": area, "median_proj_px": med, "frac_positive": frac})
            print(f"[V4-FIG] {fn} n={area} med={med:+.1f}px frac={frac:.3f}", flush=True)
    (a.out / 'index.json').write_text(json.dumps(index, indent=2))
    print('[V4-FIG] done ->', a.out, flush=True)


if __name__ == '__main__':
    main()
