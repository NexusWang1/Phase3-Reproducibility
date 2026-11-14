"""Dump fair box-level MSP, raw-logit energy and P3 Mahalanobis scores."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

from eval_ddm_val import _box_slice, _label_boxes_cls


def _pairs(root, split):
    from field_use_common import paired_paths
    return list(paired_paths(root, split))


def _fit(model, tap, pairs, seed, limit, device):
    from field_use_common import prepare_six
    rng=np.random.default_rng(seed)
    selected=np.sort(rng.choice(len(pairs),size=min(limit,len(pairs)),replace=False))
    sums={c:None for c in (0,1,2)}; counts={c:0 for c in (0,1,2)}
    second=None
    for j,index in enumerate(selected,1):
        rgb,ir,label=pairs[int(index)]
        x,tr=prepare_six(rgb,ir,torch,device)
        with torch.inference_mode():model(x)
        feat=tap.pop()
        if feat is None:raise RuntimeError(f'P3 hook did not fire: {rgb}')
        boxes=_label_boxes_cls(label,tr[0],tr[1],{0,1,2})
        for box in boxes:
            sy,sx=_box_slice(box,tr,feat.shape[-2:])
            vector=feat[0,:,sy,sx].mean((1,2)).double().cpu().numpy()
            if not np.isfinite(vector).all():raise RuntimeError(f'Non-finite P3 fit vector: {rgb}')
            c=int(box[4]);sums[c]=vector.copy() if sums[c] is None else sums[c]+vector
            counts[c]+=1
            second=np.outer(vector,vector) if second is None else second+np.outer(vector,vector)
        if j==1 or j%100==0:print(f'[OOD-FIT] {j}/{len(selected)} boxes={sum(counts.values())}',flush=True)
    if any(n==0 for n in counts.values()):raise RuntimeError(f'Missing train class for Gaussian fit: {counts}')
    n=sum(counts.values());means=np.stack([sums[c]/counts[c] for c in (0,1,2)])
    global_mean=sum(sums.values())/n
    # Pooled within-class covariance, not a mixture between-class covariance.
    within=second-sum(counts[c]*np.outer(means[c],means[c]) for c in (0,1,2))
    covariance=within/max(1,n-3)
    lam=.1;diag=np.diag(np.diag(covariance));covariance=(1-lam)*covariance+lam*diag
    covariance+=np.eye(covariance.shape[0])*max(1e-8,float(np.trace(covariance))/covariance.shape[0]*1e-6)
    precision=np.linalg.inv(covariance)
    fit={'means':means,'precision':precision}
    meta={'seed':seed,'selected_frames':len(selected),'available_train_frames':len(pairs),
          'selected_index_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
          'ordered_train_rgb_sha256':hashlib.sha256('\n'.join(str(item[0]) for item in pairs).encode()).hexdigest(),
          'feature_tap':'Detect input P3 (index 1), forward_pre_hook',
          'feature_dim':int(means.shape[1]),'class_box_counts':counts,
          'covariance':'pooled within-class, sample denominator N-3, 0.1 diagonal shrinkage + 1e-6 trace ridge',
          'shrinkage_lambda':lam,'train_only':True,'global_fit_mean_norm':float(np.linalg.norm(global_mean))}
    return fit,meta


def _energy_logits(preds,anchors,conf,state):
    """Select the class-logit branch once, then audit exact NMS anchor matching."""
    candidates={}
    for name in ('one2many','one2one'):
        branch=preds.get(name)
        if isinstance(branch,dict) and isinstance(branch.get('scores'),torch.Tensor):
            scores=branch['scores']
            if scores.ndim==3 and scores.shape[0]==1:candidates[name]=scores[0]
    if not candidates:raise RuntimeError('Raw Detect one2many/one2one class logits unavailable')
    def matched(logits):
        valid=(anchors>=0)&(anchors<logits.shape[-1])
        probabilities=np.full(len(anchors),np.nan,dtype=float)
        if valid.any():
            index=torch.as_tensor(anchors[valid],device=logits.device,dtype=torch.long)
            probabilities[valid]=torch.sigmoid(logits[:,index].float()).amax(0).detach().cpu().numpy()
        hits=np.isfinite(probabilities)&(np.abs(probabilities-conf)<1e-3)
        finite=np.isfinite(probabilities)
        error=float(np.abs(probabilities[finite]-conf[finite]).mean()) if finite.any() else float('inf')
        return int(hits.sum()),error
    if state['branch'] is None and len(anchors):
        # Primary key is match rate; mean error only resolves a tie.
        checks={name:matched(logits) for name,logits in candidates.items()}
        state['branch']=max(candidates,key=lambda name:(checks[name][0],-checks[name][1],name=='one2many'))
        print(f"[OOD-ENERGY] fixed branch={state['branch']}",flush=True)
    if state['branch'] is not None and state['branch'] not in candidates:
        raise RuntimeError(f"Chosen Energy branch disappeared: {state['branch']}")
    if state['branch'] is None:return None
    logits=candidates[state['branch']]
    hits,_=matched(logits)
    state['matched']+=hits;state['total']+=len(anchors)
    return logits


def _score_image(model,tap,rgb,ir,device,tau,fit,dataset,energy_state,label=None):
    from field_use_common import prepare_six
    from ultralytics.utils.nms import non_max_suppression
    x,tr=prepare_six(rgb,ir,torch,device)
    with torch.inference_mode():result=model(x)
    feat=tap.pop()
    if feat is None:raise RuntimeError(f'P3 hook did not fire: {rgb}')
    if not isinstance(result,tuple) or not isinstance(result[1],dict):
        raise RuntimeError('Raw Detect logits unavailable; cannot map Energy to kept anchors')
    raw,preds=result
    with torch.inference_mode():
        boxes,indices=non_max_suppression(raw,conf_thres=min(.001,tau),iou_thres=.7,nc=3,
                                           multi_label=False,agnostic=False,max_det=300,return_idxs=True)
    det=boxes[0].detach().float().cpu().numpy()
    anchors=indices[0].detach().long().cpu().numpy().reshape(-1)
    if len(det)!=len(anchors):raise RuntimeError('NMS box/anchor mapping length mismatch')
    valid=(det[:,4]>=tau)&np.isin(det[:,5].astype(int),[0,1,2])
    det,anchors=det[valid].copy(),anchors[valid]
    width,height,nw,nh,px,py=tr
    if len(det):
        det[:,[0,2]]=np.clip((det[:,[0,2]]-px)*width/nw,0,width)
        det[:,[1,3]]=np.clip((det[:,[1,3]]-py)*height/nh,0,height)
        valid=(det[:,2]>det[:,0])&(det[:,3]>det[:,1]);det,anchors=det[valid],anchors[valid]
    logits=_energy_logits(preds,anchors,det[:,4],energy_state)
    rows=[];null_energy=0;null_maha=0
    for box,index in zip(det,anchors):
        energy=None;maha=None
        if logits is not None and 0<=index<logits.shape[-1]:
            value=-torch.logsumexp(logits[:,int(index)].float(),dim=0)
            if torch.isfinite(value):energy=float(value)
        if energy is None:null_energy+=1
        sy,sx=_box_slice(box,tr,feat.shape[-2:])
        vector=feat[0,:,sy,sx].mean((1,2)).double().cpu().numpy()
        if np.isfinite(vector).all():
            diff=fit['means']-vector[None,:]
            distances=np.einsum('cd,de,ce->c',diff,fit['precision'],diff)
            value=float(np.sqrt(max(0.,float(distances.min()))))
            if np.isfinite(value):maha=value
        if maha is None:null_maha+=1
        rows.append({'image':str(rgb),'dataset':dataset,'box_xyxy':[float(v) for v in box[:4]],
                     'conf':float(box[4]),'cls':int(box[5]),'energy':energy,'maha':maha})
    gt=[] if label is None else [dict(image=str(rgb),box_xyxy=[float(v) for v in box[:4]],cls=int(box[4]))
                                       for box in _label_boxes_cls(label,width,height,{0,1,2})]
    return rows,gt,null_energy,null_maha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--rgbt-root',type=Path,required=True)
    p.add_argument('--m3fd-root',type=Path,required=True);p.add_argument('--hardneg-manifest',type=Path,required=True)
    p.add_argument('--phase2-package',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    p.add_argument('--tau',type=float,default=.045);p.add_argument('--seed',type=int,default=20260929)
    p.add_argument('--fit-frames',type=int,default=3000);p.add_argument('--expected-val',type=int,default=905)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(f'Refusing overwrite: {a.out}')
    if not 0<a.fit_frames<=3000:raise ValueError('--fit-frames must be in 1..3000')
    for path in (a.weights,a.hardneg_manifest):
        if not path.is_file():raise FileNotFoundError(path)
    if not a.phase2_package.is_dir():raise FileNotFoundError(a.phase2_package)
    sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916')]
    from smoke_frp_v2_7 import make
    os.environ.update(PHY_DDM='0',PHY_DR='0',PHY_FRP='1')
    model,_,_=make(a.weights,a.device,True)
    tap=[None]
    def capture(_module,args):tap[0]=args[0][1].detach()
    handle=model.model[-1].register_forward_pre_hook(capture)
    class Tap:
        def pop(self):
            value=tap[0];tap[0]=None;return value
    reader=Tap()
    train=_pairs(a.rgbt_root,'train');val=_pairs(a.rgbt_root,'val')
    if len(val)!=a.expected_val:raise RuntimeError(f'RGBT val frames={len(val)}, expected={a.expected_val}')
    hard=[json.loads(row) for row in a.hardneg_manifest.read_text(encoding='utf-8').splitlines() if row.strip()]
    if len(hard)!=592:raise RuntimeError(f'M3FD hardneg frames={len(hard)}, expected=592')
    fit,meta=_fit(model,reader,train,a.seed,a.fit_frames,a.device)
    meta.update(weights=str(a.weights),rgbt_root=str(a.rgbt_root),m3fd_root=str(a.m3fd_root),
                hardneg_manifest=str(a.hardneg_manifest),tau=a.tau,
                energy='negative logsumexp of selected raw pre-sigmoid Detect class logits, exact NMS anchor index',
                n_rgbt_val=len(val),n_m3fd_hardneg=len(hard))
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'maha_fit.npz',**fit)
    (a.out/'fit_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    n={'rgbt_val':0,'m3fd_hardneg':0};events=persons=null_e=null_m=0
    energy_state={'branch':None,'matched':0,'total':0}
    with (a.out/'ood_scores.jsonl').open('w',encoding='utf-8') as scores,(a.out/'gt_records.jsonl').open('w',encoding='utf-8') as gts:
        cases=[(rgb,ir,label,'rgbt_val') for rgb,ir,label in val]
        cases +=[(a.m3fd_root/row['rgb'],a.m3fd_root/row['ir_gray'],None,'m3fd_hardneg') for row in hard]
        for j,(rgb,ir,label,dataset) in enumerate(cases,1):
            rows,truth,ne,nm=_score_image(model,reader,rgb,ir,a.device,a.tau,fit,dataset,energy_state,label)
            for row in rows:scores.write(json.dumps(row,allow_nan=False)+'\n')
            for row in truth:gts.write(json.dumps(row,allow_nan=False)+'\n')
            n[dataset]+=len(rows);null_e+=ne;null_m+=nm
            if dataset=='m3fd_hardneg':
                events+=sum(z['cls'] in (0,1) for z in rows)
                persons+=sum(z['cls']==2 for z in rows)
            if j==1 or j%100==0:print(f'[OOD-DUMP] {j}/{len(cases)} boxes={sum(n.values())}',flush=True)
    handle.remove()
    total=sum(n.values())
    summary={'n_boxes':n,'n_gt_rows':sum(1 for _ in (a.out/'gt_records.jsonl').open(encoding='utf-8')),
             'hardneg_event_fp':events,'hardneg_person_fp':persons,
             'energy_null_rate':null_e/max(1,total),'maha_null_rate':null_m/max(1,total),
             'energy_branch':energy_state['branch'],
             'energy_anchor_conf_match_rate':energy_state['matched']/max(1,energy_state['total'])}
    (a.out/'dump_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('[OOD-DUMP] '+json.dumps(summary),flush=True)
    if summary['n_gt_rows']==0 or max(summary['energy_null_rate'],summary['maha_null_rate'])>=.01:
        raise RuntimeError('OOD dump completeness failed; see dump_summary.json')
    if energy_state['branch'] is None or summary['energy_anchor_conf_match_rate']<.99:
        raise RuntimeError(f'Energy anchor alignment failed: {summary}; see dump_summary.json')
    if abs(events-155)>1 or abs(persons-373)>1:
        raise RuntimeError(f'Baseline FP mismatch: events={events} person={persons}, expected 155/373')

if __name__=='__main__':main()
