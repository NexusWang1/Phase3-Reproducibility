"""Frozen DAv2-L depth service with native-safetensors and HF fallback paths.

Set ``PHY_DEPTH_WEIGHTS=/abs/model.safetensors`` (and optionally
``PHY_DEPTH_REPO=/abs/Depth-Anything-V2``) for the offline native DAv2 path.
Without PHY_DEPTH_WEIGHTS, the original Transformers/HF loader is used. Both
paths return float32 relative depth in [0,1], where 1 means far.
"""
from __future__ import annotations
import contextlib
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F


class DepthService:
    MODEL_ID = 'depth-anything/Depth-Anything-V2-Large'

    def __init__(self, device='cuda:0', cache_dir=None):
        self.device = torch.device(device)
        self.cache_dir = str(cache_dir or os.getenv('PHY_DEPTH_CACHE_DIR', 'weights_external'))
        self.model = self.processor = None
        self.native = False

    @staticmethod
    def _as_numpy(rgb):
        if torch.is_tensor(rgb):
            x = rgb.detach().cpu()
            x = x.permute(1, 2, 0) if x.ndim == 3 and x.shape[0] in (1, 3) else x
            return (x.float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        return np.asarray(rgb, dtype=np.uint8)

    @staticmethod
    def _normalize(depth):
        lo = depth.amin((-2, -1), keepdim=True)
        hi = depth.amax((-2, -1), keepdim=True)
        # Native DAv2 output is disparity-like (larger=near). Preserve 1=far.
        return 1 - (depth - lo) / (hi - lo).clamp_min(1e-6)

    @staticmethod
    def _repo_path():
        raw = Path(os.getenv('PHY_DEPTH_REPO', 'weights_external/Depth-Anything-V2'))
        if raw.is_absolute():
            return raw
        cwd = Path.cwd() / raw
        if cwd.is_dir():
            return cwd
        # Portable package-relative fallback; PHY_DEPTH_REPO takes precedence.
        return Path(__file__).resolve().parents[3] / raw

    def _load_native(self, weights):
        repo = self._repo_path()
        if not repo.is_dir():
            raise FileNotFoundError(f'PHY_DEPTH_REPO not found: {repo}')
        sys.path.insert(0, str(repo))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError(f'native DAv2 import failed from {repo}: {e}') from e
        state = load_file(str(weights), device='cpu')
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f'[DEPTH-NATIVE-LOAD] failed; first_keys={list(state)[:5]}', flush=True)
            raise RuntimeError('native DAv2 safetensors state_dict load failed') from e
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.native = True
        print(f'[DEPTH-NATIVE-LOAD] weights={weights} repo={repo} keys={len(state)}', flush=True)

    def _load(self):
        if self.model is not None:
            return
        weights = os.getenv('PHY_DEPTH_WEIGHTS')
        if weights:
            path = Path(weights)
            if path.suffix != '.safetensors':
                raise ValueError(f'PHY_DEPTH_WEIGHTS must name .safetensors: {path}')
            if not path.is_file():
                raise FileNotFoundError(path)
            self._load_native(path)
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError('DepthService requires transformers: pip install transformers') from e
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, cache_dir=self.cache_dir)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID, cache_dir=self.cache_dir).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _native_tensor(self, image):
        h, w = image.shape[:2]
        scale = 518 / min(h, w)
        nh = int(math.ceil(h * scale / 14) * 14)
        nw = int(math.ceil(w * scale / 14) * 14)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255)[None]
        tensor = F.interpolate(tensor, (nh, nw), mode='bilinear', align_corners=False)
        mean = tensor.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = tensor.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    @torch.inference_mode()
    def infer(self, rgbs):
        self._load()
        single = not isinstance(rgbs, (list, tuple))
        rgbs = [rgbs] if single else rgbs
        images = [self._as_numpy(x) for x in rgbs]
        if self.native:
            out = []
            # Native DAv2's variable-size preprocessing is intentionally
            # per-image; do not pad heterogeneous images into a fake batch.
            for image in images:
                x = self._native_tensor(image).to(self.device)
                amp = torch.autocast('cuda', enabled=self.device.type == 'cuda') if self.device.type == 'cuda' else contextlib.nullcontext()
                with amp:
                    pred = self.model(x)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                if pred.ndim == 3:
                    pred = pred[:, None]
                pred = F.interpolate(pred.float(), image.shape[:2], mode='bicubic', align_corners=False)
                out.append(self._normalize(pred)[0, 0].float().cpu())
            return out[0] if single else out
        hw = [x.shape[:2] for x in images]
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            pred = self.model(**inputs).predicted_depth[:, None]
        out = [self._normalize(F.interpolate(pred[i:i+1], size=size, mode='bicubic', align_corners=False))[0, 0].float().cpu() for i, size in enumerate(hw)]
        return out[0] if single else out
