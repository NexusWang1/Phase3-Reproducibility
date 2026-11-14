"""Pre/post L2 algebra and cue-isolation audit on 20 paired RGBT frames."""
from __future__ import annotations
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from tools.ddm_dataset import RGBTDDM
from tools.ddm_cues_v2_legacy import DDM3Cue as LegacyCue, _gray, _sobel
_cue_path=PKG/'ultralytics/nn/modules/ddm_cues.py'
_spec=importlib.util.spec_from_file_location('ddm_cues_v3_audit',_cue_path)
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
DDM3Cue=_module.DDM3Cue

def components(rgb,depth,k):
    gi=_sobel(_gray(rgb));gd=_sobel(depth)
    ei=torch.sigmoid((gi-gi.mean((-2,-1),keepdim=True))/gi.std((-2,-1),keepdim=True).clamp_min(1e-6))
    ed=torch.sigmoid((gd-gd.mean((-2,-1),keepdim=True))/gd.std((-2,-1),keepdim=True).clamp_min(1e-6))
    e=ei*ed
    q=torch.quantile(depth.flatten(2),torch.linspace(0,1,k+1,device=depth.device),dim=2).squeeze(-1).permute(1,0).unsqueeze(-1).unsqueeze(-1)
    return e,q

def pearson(a,b):
    a=a.flatten().double();b=b.flatten().double();a=a-a.mean();b=b-b.mean()
    return float(((a*b).sum()/(a.square().sum()*b.square().sum()).sqrt().clamp_min(1e-12)).item())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rgbt-root',type=Path,required=True)
    p.add_argument('--rgbt-cache',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--count',type=int,default=20)
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(f'Refusing overwrite: {a.out}')
    ds=RGBTDDM(a.rgbt_root,a.rgbt_cache,'train')
    if len(ds)<a.count:raise RuntimeError(f'Need {a.count} paired frames, found {len(ds)}')
    indices=random.Random(a.seed).sample(range(len(ds)),a.count)
    torch.manual_seed(a.seed)
    old=LegacyCue().to(a.device).eval();new=DDM3Cue().to(a.device).eval()
    new.load_state_dict(old.state_dict(),strict=True)
    rows=[]
    with torch.inference_mode():
        for i in indices:
            sample=ds[i];rgb=sample['rgb'][None].to(a.device);depth=sample['depth'][None].to(a.device)
            old_l1,old_l2,old_l3=old(rgb,depth)
            new_l1,new_l2,new_l3=new(rgb,depth)
            e,q=components(rgb,depth,old.k)
            j_idx=torch.searchsorted(q[:,1:-1,0,0].contiguous(),depth.flatten(2).squeeze(1).contiguous(),right=False)
            cover=torch.zeros_like(e);new_cover=torch.zeros_like(e);raw=torch.zeros_like(e);layer_means=[]
            for j in range(old.k):
                lo=q[:,j:j+1];hi=q[:,j+1:j+2]
                cover+=((depth>=lo)&(depth<=hi)).to(e.dtype)
                mask=(j_idx==j).view_as(e).to(e.dtype)
                new_cover+=mask
                n=mask.sum((-2,-1),keepdim=True)
                mu=(e*mask).sum((-2,-1),keepdim=True)/(n+1e-6)
                var=((e-mu).square()*mask).sum((-2,-1),keepdim=True)/(n+1e-6)
                z=(e-mu)/torch.sqrt(var+1e-6)
                raw+=mask*z
                if n.item():layer_means.append(float((z*mask).sum().div(n).abs()))
            final_layer_means=[]
            for j in range(old.k):
                mask=(j_idx==j).view_as(e).to(e.dtype)
                n=mask.sum()
                if n.item():final_layer_means.append(float((raw*mask).sum().div(n).abs()))
            padding=(depth==1).flatten(2).squeeze(1)
            padding_count=padding.sum().item()
            row={'path':sample['path'],
                 'legacy_minus_e_over_k_max_abs':float((old_l2-e/old.k).abs().max()),
                 'legacy_overlap_pixel_ratio':float((cover>1).float().mean()),
                 'legacy_formula_max_abs':float((old_l2-e*cover/old.k).abs().max()),
                 'new_cover_min':float(new_cover.min()),
                 'new_cover_max':float(new_cover.max()),
                 'new_l2_vs_sigmoid_raw_max_abs':float((new_l2-torch.sigmoid(raw)).abs().max()),
                 'new_l2_vs_e_pearson':pearson(new_l2,e),
                 'padding_pixel_fraction':float(padding.float().mean()),
                 'padding_stratum0_fraction':float(padding.float().mean()),
                 'padding_actual_stratum0_fraction':float(((padding)&(j_idx==0)).sum().item()/padding_count) if padding_count else 0.,
                 'padding_last_stratum_fraction':float(((padding)&(j_idx==old.k-1)).sum().item()/padding_count) if padding_count else 0.,
                 'max_abs_layer_z_mean':max(layer_means,default=0.),
                 'max_abs_layer_final_raw_mean':max(final_layer_means,default=0.),
                 'l1_torch_equal':torch.equal(old_l1,new_l1),
                 'l3_torch_equal':torch.equal(old_l3,new_l3)}
            rows.append(row);print(f'[L2-AUDIT] {len(rows)}/{a.count} {json.dumps(row)}',flush=True)
    median=float(np.median([r['new_l2_vs_e_pearson'] for r in rows]))
    max_old=max(r['legacy_minus_e_over_k_max_abs'] for r in rows)
    max_layer=max(r['max_abs_layer_z_mean'] for r in rows)
    max_final_layer=max(r['max_abs_layer_final_raw_mean'] for r in rows)
    report={'count':len(rows),'seed':a.seed,'legacy_e_over_k_max_abs':max_old,
            'legacy_exact_e_over_k':max_old<1e-6,
            'legacy_boundary_overlap_frames':sum(r['legacy_overlap_pixel_ratio']>0 for r in rows),
            'new_cover_all_one':all(r['new_cover_min']==1 and r['new_cover_max']==1 for r in rows),
            'new_l2_e_pearson_median':median,
            'padding_stratum0_fraction':float(np.mean([r['padding_stratum0_fraction'] for r in rows])),
            'max_abs_layer_z_mean':max_layer,
            'max_abs_layer_final_raw_mean':max_final_layer,
            'l1_equal_all':all(r['l1_torch_equal'] for r in rows),
            'l3_equal_all':all(r['l3_torch_equal'] for r in rows),
            'rows':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert max(r['legacy_formula_max_abs'] for r in rows)<1e-6,'legacy formula reproduction failed'
    assert report['new_cover_all_one'],'new strata do not partition every pixel exactly once'
    assert max(r['new_l2_vs_sigmoid_raw_max_abs'] for r in rows)<1e-6,'new L2 formula reproduction failed'
    assert report['l1_equal_all'] and report['l3_equal_all'],'L1/L3 changed'
    assert max_layer<1e-4,'L2 within-layer z-score mean not zero'
    assert max_final_layer<1e-5,'Final l2_raw layer mean not zero'
    assert median<.95,f'L2 still nearly affine to co-occurrence: median Pearson={median}'
    print(f'[L2-AUDIT] PASS {a.out}',flush=True)

if __name__=='__main__':main()
