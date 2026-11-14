"""Read-only DDM hard-negative FP zoning probe, using the gate detection path."""
from __future__ import annotations
import argparse,json,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def _cache(root,dataset,rgb):return (Path(root)/Path(rgb).relative_to(dataset)).with_suffix('.png')
def _depth_letterboxed(path,tr,device):
    if not path.is_file():raise FileNotFoundError(f'missing offline depth cache: {path}')
    d=torch.from_numpy(np.asarray(Image.open(path),dtype=np.uint16).copy()).float().div(65535)[None,None]
    _,_,nw,nh,px,py=tr;d=F.interpolate(d,(nh,nw),mode='bilinear',align_corners=False);canvas=torch.ones(1,1,640,640);canvas[...,py:py+nh,px:px+nw]=d
    return canvas.to(device)
def _box_slice(box,tr,shape):
    """Same expanded-box geometry as field_use_common.box_field_slice."""
    width,height,nw,nh,px,py=tr;fh,fw=shape;x0,y0,x1,y1=map(float,box[:4]);dx=(x1-x0)*.25;dy=(y1-y0)*.25
    x0=max(0.,x0-dx);y0=max(0.,y0-dy);x1=min(float(width),x1+dx);y1=min(float(height),y1+dy)
    left=max(0,min(fw-1,int(np.floor((px+x0*nw/width)*fw/640))));top=max(0,min(fh-1,int(np.floor((py+y0*nh/height)*fh/640))))
    right=max(left+1,min(fw,int(np.ceil((px+x1*nw/width)*fw/640))));bottom=max(top+1,min(fh,int(np.ceil((py+y1*nh/height)*fh/640))))
    return slice(top,bottom),slice(left,right)
def _components(mask):
    """8-connected components, matching frozen FRP connectivity."""
    h,w=mask.shape;seen=np.zeros_like(mask,bool);out=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        seen[y,x]=True;stack=[(y,x)];group=[]
        while stack:
            yy,xx=stack.pop();group.append((yy,xx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
        out.append(group)
    return out
def _summary(values):
    return {'count':len(values),'mean':float(np.mean(values)) if values else None,'min':float(np.min(values)) if values else None,'max':float(np.max(values)) if values else None,'median':float(np.median(values)) if values else None}
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--ddm-ckpt',type=Path,required=True);p.add_argument('--m3fd-root',type=Path,required=True);p.add_argument('--hardneg-manifest',type=Path,required=True);p.add_argument('--m3fd-cache',type=Path);p.add_argument('--phase2-package',type=Path,default=Path(__file__).parent.parent/'FieldGuide_Phase2_FRP_20260916');p.add_argument('--device',default='cuda:0');p.add_argument('--tau',type=float,default=.045);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.environ.setdefault('PHY_DDM','1')
    if not a.phase2_package.is_dir():raise FileNotFoundError(a.phase2_package)
    out=a.out/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False);mcache=a.m3fd_cache or a.m3fd_root/'depth_cache'
    sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916')]
    from smoke_frp_v2_7 import make
    from field_use_common import prepare_six,field_and_detections
    from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
    active=[None];model,_,_=make(a.weights,a.device,True);DDMFusionHook(depth_provider=lambda _rgb:active[0]).attach(model,a.ddm_ckpt)
    records=[];rows=[json.loads(x) for x in a.hardneg_manifest.read_text().splitlines() if x.strip()]
    for i,row in enumerate(rows,1):
        rgb=a.m3fd_root/row['rgb'];ir=a.m3fd_root/row['ir_gray'];x,tr=prepare_six(rgb,ir,torch,a.device);active[0]=_depth_letterboxed(_cache(mcache,a.m3fd_root,rgb),tr,a.device)
        _,det=field_and_detections(model,torch,__import__('ultralytics.utils.nms',fromlist=['non_max_suppression']).non_max_suppression,False,x,tr,a.tau,{0,1,2})
        aux=model._last_frp_aux;t=model._last_t[0,0].detach().float().cpu().numpy();amb=aux['ambiguous'][0,0].detach().cpu().numpy().astype(bool);pddm=aux.get('p_ddm')
        if pddm is None:raise RuntimeError('DDM is disabled: p_ddm unavailable')
        pddm=pddm[0,0].detach().float().cpu().numpy();groups=_components(amb)
        # Production topk=0 scores every ambiguous component; retain the
        # explicit ranking rule for a non-default diagnostic run.
        ranked=sorted(enumerate(groups),key=lambda q:float(t[tuple(np.asarray(q[1]).T)].max()),reverse=True);keep={k for k,_ in ranked[:model.frp.topk or len(ranked)]}
        for box in det.tolist():
            sy,sx=_box_slice(box,tr,t.shape);region=t[sy,sx];zone='high' if float(region.max())>=model.frp.t_high else ('amb' if amb[sy,sx].any() else 'low')
            overlaps=[(k,sum(sy.start<=y<sy.stop and sx.start<=x<sx.stop for y,x in g)) for k,g in enumerate(groups)];k,n=max(overlaps,key=lambda z:z[1]) if overlaps else (None,0)
            scored=bool(zone=='amb' and n and k in keep);comp=np.asarray(groups[k]) if k is not None and n else None
            vals=pddm[comp[:,0],comp[:,1]] if comp is not None else pddm[sy,sx]
            records.append({'image':row.get('id',str(row['rgb'])),'rgb':row['rgb'],'box_xyxy':[float(v) for v in box[:4]],'class':int(box[5]),'confidence':float(box[4]),'zone':zone,'scored':scored,'direct_pass':not scored,'p_ddm_mean':float(vals.mean()),'p_ddm_max':float(vals.max()),'p_ddm_min':float(vals.min()),'t_max':float(region.max())})
        print(f'[DDM-FP-ZONES] {i}/{len(rows)} fp={len(records)}',flush=True)
    (out/'fp_zone_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    # Gate hardneg_fp is smoke/fire only.  Keep person records for diagnosis,
    # but never let them contaminate the preregistered 155-count histogram.
    event=[r for r in records if r['class'] in {0,1}];zone={z:[r for r in event if r['zone']==z] for z in ('high','amb','low')};hist={z:{'total':len(v),'scored':sum(r['scored'] for r in v),'direct_pass':sum(r['direct_pass'] for r in v)} for z,v in zone.items()}
    person=[r for r in records if r['class']==2];summary={'weights':str(a.weights),'ddm_ckpt':str(a.ddm_ckpt),'tau':a.tau,'n_hardneg':len(rows),'fp_total':len(event),'fp_records_all_classes':len(records),'zone_histogram':hist,'p_ddm_by_zone':{z:_summary([r['p_ddm_mean'] for r in v]) for z,v in zone.items()},'direct_high_pddm_ge_tau':sum(r['zone']=='high' and r['direct_pass'] and r['p_ddm_mean']>=a.tau for r in event),'theoretical_extra_intercept_if_ddm_covers_direct_high':sum(r['zone']=='high' and r['direct_pass'] and r['p_ddm_mean']>=a.tau for r in event),'person_fp':{'total':len(person),'zone_histogram':{z:sum(r['zone']==z for r in person) for z in zone},'records':person}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print(f'[DDM-FP-ZONES] out={out}')
if __name__=='__main__':main()
