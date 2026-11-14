"""Dump letterboxed depth at field resolution (160x160) for the YHT field-sequence frames.

Read-only analysis utility. Pairs with runs/phase3/yht_fields_*/fields/*.npy.
Depth geometry: identical to prepare_six letterbox, then bilinear 640 -> 160.

Run:
  PYTHONPATH="$ROOT:$PKG" python -u "$PKG/tools/dump_depth160.py" \
    --rgb-root "/path/to/YHT/254p RGB Images" \
    --depth-cache "/path/to/YHT/depth_cache/254p RGB Images" \
    --start 13651 --end 14701 --stride 1 \
    --out "$ROOT/runs/phase3/yht_depth160"
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FIELD = 160

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--depth-cache', type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    (a.out / 'depth160').mkdir(parents=True)
    saved, skipped = 0, []
    for n in range(a.start, a.end + 1, a.stride):
        rgb = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            c = a.rgb_root / f'254p RGB Frame ({n}){ext}'
            if c.is_file():
                rgb = c
                break
        if rgb is None:
            skipped.append({'frame_id': n, 'reason': 'missing_rgb'})
            continue
        dpath = (a.depth_cache / rgb.relative_to(a.rgb_root)).with_suffix('.png')
        if not dpath.is_file():
            skipped.append({'frame_id': n, 'reason': 'missing_depth'})
            continue
        with Image.open(rgb) as im:
            width, height = im.size
        scale = min(640 / width, 640 / height)
        nw, nh = round(width * scale), round(height * scale)
        px, py = (640 - nw) // 2, (640 - nh) // 2
        d = torch.from_numpy(np.asarray(Image.open(dpath), np.uint16).copy()).float().div(65535)[None, None]
        d = F.interpolate(d, (nh, nw), mode='bilinear', align_corners=False)
        canvas = torch.ones(1, 1, 640, 640)
        canvas[..., py:py + nh, px:px + nw] = d
        d160 = F.interpolate(canvas, (FIELD, FIELD), mode='bilinear', align_corners=False)[0, 0]
        np.save(a.out / 'depth160' / f'{n:06d}.npy', d160.numpy().astype(np.float32))
        saved += 1
    meta = {'start': a.start, 'end': a.end, 'stride': a.stride, 'saved': saved, 'skipped': skipped,
            'geometry': 'letterbox(prepare_six) -> bilinear 640->160; pad=1.0(far); value=relative depth, 1=far'}
    (a.out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ('saved',)} | {'skipped': len(skipped)}))

if __name__ == '__main__':
    main()
