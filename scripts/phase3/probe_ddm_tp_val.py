"""Read-only RGBT validation detector-box P_ddm probe for veto calibration."""
from __future__ import annotations
import argparse,json,os,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def _box_slice(box,tr,shape):
    """Map an unexpanded original-image detection box to a field map."""
    width,height,nw,nh,px,py=tr;fh,fw=shape;x0,y0,x1,y1=map(float,box[:4])
    l=max(0,min(fw-1,int(np.floor((px+x0*nw/width)*fw/640))));t=max(0,min(fh-1,int(np.floor((py+y0*nh/height)*fh/640))))
    r=max(l+1,min(fw,int(np.ceil((px+x1*nw/width)*fw/640))));b=max(t+1,min(fh,int(np.ceil((py+y1*nh/height)*fh/640))))
    return slice(t,b),slice(l,r)
def _iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);inter=max(0,x1-x0)*max(0,y1-y0)
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def _q(v):
    return {f'p{k}':float(np.percentile(v,k)) if v else None for k in (1,5,10,25,50)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--ddm-ckpt',type=Path,required=True);p.add_argument('--rgbt-root',type=Path,required=True);p.add_argument('--rgbt-cache',type=Path,required=True);p.add_argument('--phase2-package',type=Path,default=Path(__file__).parent.parent/'FieldGuide_Phase2_FRP_20260916');p.add_argument('--device',default='cuda:0');p.add_argument('--tau',type=float,default=.045);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.environ.setdefault('PHY_DDM','1');out=a.out/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    if not a.phase2_package.is_dir():raise FileNotFoundError(a.phase2_package)
    sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916'),str(Path(__file__).parent)]
    from smoke_frp_v2_7 import make
    from field_use_common import paired_paths,prepare_six,field_and_detections
    from eval_ddm_val import _cache,_depth_letterboxed,_label_boxes_cls
    from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
    active=[None];model,_,_=make(a.weights,a.device,True);DDMFusionHook(depth_provider=lambda _rgb:active[0]).attach(model,a.ddm_ckpt);records=[]
    for index,(rgb,ir,label) in enumerate(paired_paths(a.rgbt_root,'val'),1):
        x,tr=prepare_six(rgb,ir,torch,a.device);active[0]=_depth_letterboxed(_cache(a.rgbt_cache,a.rgbt_root,rgb),tr,a.device)
        _,det=field_and_detections(model,torch,__import__('ultralytics.utils.nms',fromlist=['non_max_suppression']).non_max_suppression,False,x,tr,a.tau,{0,1,2})
        pddm=model._last_frp_aux.get('p_ddm')
        if pddm is None:raise RuntimeError('DDM is disabled: p_ddm unavailable')
        pddm=pddm[0,0].detach().float().cpu().numpy();gts=_label_boxes_cls(label,tr[0],tr[1],{0,1,2});used=set();key=str(rgb)
        for box in sorted(det.tolist(),key=lambda z:-z[4]):
            c=int(box[5]);hits=[(j,g) for j,g in enumerate(gts) if j not in used and int(g[4])==c and _iou(box,g)>=.5]
            hit=next(iter(hits),None)
            if hit:used.add(hit[0])
            sy,sx=_box_slice(box,tr,pddm.shape);vals=pddm[sy,sx]
            records.append({'image':key,'box_xyxy':[float(v) for v in box[:4]],'class':c,'confidence':float(box[4]),'matched_gt':bool(hit),'matched_gt_class':int(hit[1][4]) if hit else None,'iou':float(_iou(box,hit[1])) if hit else 0.,'p_ddm_mean':float(vals.mean()),'p_ddm_max':float(vals.max()),'p_ddm_min':float(vals.min())})
        print(f'[DDM-TP-VAL] {index} detections={len(records)}',flush=True)
    (out/'tp_val_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    by_class={str(c):{'tp':sum(r['matched_gt'] and r['class']==c for r in records),'fp':sum(not r['matched_gt'] and r['class']==c for r in records)} for c in (0,1,2)}
    tp=[r['p_ddm_mean'] for r in records if r['matched_gt']];fp=[r['p_ddm_mean'] for r in records if not r['matched_gt']];theta={}
    for th in (.005,.01,.02,.03,.045,.06):theta[str(th)]={'tp_survival':float(np.mean(np.asarray(tp)>=th)) if tp else 0.,'fp_kill':float(np.mean(np.asarray(fp)<th)) if fp else 0.}
    summary={'weights':str(a.weights),'ddm_ckpt':str(a.ddm_ckpt),'tau':a.tau,'n_records':len(records),'by_class':by_class,'tp_p_ddm_mean_quantiles':_q(tp),'fp_p_ddm_mean_quantiles':_q(fp),'veto_thresholds':theta}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print(f'[DDM-TP-VAL] out={out}')
if __name__=='__main__':main()
