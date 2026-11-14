"""One letterbox geometry path shared by DDM training and evaluation.

RGB resize/padding follows Phase2 field_use_common.prepare_six (PIL bilinear,
round(), fill=114). Depth interpolation follows the existing v1 evaluator.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def letterbox_transform(width, height, size=640):
    scale = min(size / width, size / height)
    nw, nh = round(width * scale), round(height * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    return width, height, nw, nh, px, py


def rgb_letterboxed(path: Path, size=640):
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        tr = letterbox_transform(*rgb.size, size)
        _, _, nw, nh, px, py = tr
        canvas = Image.new('RGB', (size, size), (114, 114, 114))
        canvas.paste(rgb.resize((nw, nh), Image.Resampling.BILINEAR), (px, py))
        arr = np.asarray(canvas).astype(np.float32) / 255
    return torch.from_numpy(arr).permute(2, 0, 1), tr


def depth_letterboxed(path: Path, tr, device='cpu', size=640):
    if not Path(path).is_file():
        raise FileNotFoundError(f'missing offline depth cache: {path}')
    with Image.open(path) as im:
        if im.mode not in {'I;16', 'I'}:
            raise ValueError(f'expected uint16 depth cache, got {im.mode}: {path}')
        arr = np.asarray(im, dtype=np.uint16).copy()
    width, height, nw, nh, px, py = tr
    if arr.shape != (height, width):
        raise ValueError(f'depth/RGB size mismatch: {path} {arr.shape} != {(height, width)}')
    d = torch.from_numpy(arr).float().div(65535)[None, None]
    d = F.interpolate(d, (nh, nw), mode='bilinear', align_corners=False)
    canvas = torch.ones(1, 1, size, size)
    canvas[..., py:py + nh, px:px + nw] = d
    return canvas.to(device)
