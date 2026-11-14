"""Isolated DDM ablation adapters; never alter the baseline model modules."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

VARIANTS=('full','no_l1','no_l2','no_l3','shuffle_depth','rgb_only','depth_only')

def cue_class(base,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    class AblatedCue(base):
        def forward(self,rgb,depth):
            if depth.ndim==3:depth=depth[:,None]
            if variant=='shuffle_depth':
                flat=depth.flatten(2)
                depth=flat[:,:,torch.randperm(flat.shape[-1],device=depth.device)].view_as(depth)
            elif variant=='rgb_only':depth=torch.zeros_like(depth)
            elif variant=='depth_only':rgb=torch.zeros_like(rgb)
            l1,l2,l3=super().forward(rgb,depth)
            if variant in ('no_l1','depth_only'):l1=torch.zeros_like(l1)
            if variant=='no_l2':l2=torch.zeros_like(l2)
            if variant=='no_l3':l3=torch.zeros_like(l3)
            return l1,l2,l3
    AblatedCue.__name__='DDM3Cue'
    return AblatedCue

def branch_class(base,variant):
    class AblatedBranch(base):
        def forward(self,l1,l2,l3,p3):
            return super().forward(l1,l2,l3,torch.zeros_like(p3) if variant=='depth_only' else p3)
    AblatedBranch.__name__='DDMBranch'
    return AblatedBranch

def patch_model_classes(variant):
    import ultralytics.nn.modules.ddm_cues as cues_module
    import ultralytics.nn.modules.ddm_branch as branch_module
    original_cues=cues_module.DDM3Cue
    original_branch=branch_module.DDMBranch
    cues_module.DDM3Cue=cue_class(original_cues,variant)
    branch_module.DDMBranch=branch_class(original_branch,variant)
    # Hook versions imported before the patch need the same replacements.
    import ultralytics.nn.modules.ddm_fusion_hook as hook
    hook.DDM3Cue=cues_module.DDM3Cue
    hook.DDMBranch=branch_module.DDMBranch
    return cues_module.DDM3Cue,branch_module.DDMBranch

def _stretched_rgb(path,size):
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)

def _stretched_depth(path,size):
    with Image.open(path) as im:
        if im.mode not in {'I;16','I'}:raise ValueError(f'Expected uint16 depth: {path}')
        a=np.asarray(im,dtype=np.uint16).copy()
    x=torch.from_numpy(a).float()[None,None].div(65535.)
    return F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)[0]

def _stretched_target(label,size):
    target=torch.zeros(1,size,size)
    if not label.is_file():return target
    for line in label.read_text(encoding='utf-8-sig').splitlines():
        vals=line.split()
        if len(vals)!=5 or int(float(vals[0])) not in {0,1}:continue
        _,cx,cy,w,h=map(float,vals)
        x0=max(0,int((cx-w/2)*size));x1=min(size,int((cx+w/2)*size))
        y0=max(0,int((cy-h/2)*size));y1=min(size,int((cy+h/2)*size))
        target[:,y0:y1,x0:x1]=1
    return target

class StretchTrainDataset(torch.utils.data.Dataset):
    """Reproduce the v1 training-side direct RGB/depth square stretch only.

    The validation loader is intentionally left on the v2 aligned geometry.
    """
    def __init__(self,base):
        self.base=base
        self.rows=base.rows
        self.skipped=base.skipped
        self.size=base.size
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        row=self.rows[i]
        rgb,depth=row[:2]
        kind=type(self.base).__name__
        if kind=='RGBTDDM':target=_stretched_target(row[2],self.size);label='rgbt'
        elif kind=='YHTDDM':target=torch.tensor(float(row[2]));label='yht'
        elif kind=='M3FDDM':target=torch.zeros(1,self.size,self.size);label='m3fd'
        else:raise TypeError(f'Unsupported dataset: {kind}')
        return {'rgb':_stretched_rgb(rgb,self.size),'depth':_stretched_depth(depth,self.size),
                'target':target,'kind':label,'path':str(rgb)}
