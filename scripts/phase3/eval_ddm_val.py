"""Offline-cache DDM validation on RGBT validation and M3FD hard negatives.

The evaluator deliberately never instantiates DepthService: every inference
receives the uint16 depth cache paired to its RGB image, and a missing cache is
a fatal reproducibility error.
"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def _cache(cache_root:Path, dataset_root:Path, rgb:Path)->Path:
    return (cache_root/rgb.relative_to(dataset_root)).with_suffix('.png')
def _depth_letterboxed(path:Path,tr,device:str)->torch.Tensor:
    """Load uint16 depth using the exact letterbox geometry of prepare_six."""
    from tools.ddm_geometry import depth_letterboxed
    return depth_letterboxed(path,tr,device)
def _label_boxes_cls(label:Path,width:int,height:int,classes:set):
    """Class-preserving local YOLO label parser: [x0,y0,x1,y1,cls]."""
    boxes=[]
    if not label.is_file():return boxes
    for row in label.read_text(encoding='utf-8-sig').splitlines():
        vals=row.split()
        if len(vals)!=5 or int(float(vals[0])) not in classes:continue
        c=int(float(vals[0]));_,cx,cy,bw,bh=map(float,vals)
        boxes.append([(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height,c])
    return boxes
def _box_slice(box,tr,shape):
    """Map a detection box (no expansion) from original pixels to a field map."""
    width,height,nw,nh,px,py=tr;fh,fw=shape;x0,y0,x1,y1=map(float,box[:4])
    left=max(0,min(fw-1,int(np.floor((px+x0*nw/width)*fw/640))));top=max(0,min(fh-1,int(np.floor((py+y0*nh/height)*fh/640))))
    right=max(left+1,min(fw,int(np.ceil((px+x1*nw/width)*fw/640))));bottom=max(top+1,min(fh,int(np.ceil((py+y1*nh/height)*fh/640))))
    return slice(top,bottom),slice(left,right)
def _apply_box_veto(det,pddm,tr,veto_t,veto_c,enabled):
    """Return event detections after the opt-in low-DDM/low-confidence veto."""
    if not enabled or veto_t<=0:return det,0
    keep=[];vetoed=0
    for box in det:
        if int(box[5]) in {0,1} and float(box[4])<veto_c:
            sy,sx=_box_slice(box,tr,pddm.shape)
            if float(pddm[sy,sx].mean())<veto_t:vetoed+=1;continue
        keep.append(box)
    return np.asarray(keep,dtype=det.dtype).reshape(-1,det.shape[1]),vetoed
def _iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);inter=max(0,x1-x0)*max(0,y1-y0)
    return inter/(max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))
def _ap(pred,gt,iou=.5):
    """Class-aware AP; rows carry image keys to forbid cross-image matches."""
    n=len(gt);used=set();tp=[]
    for p in sorted(pred,key=lambda z:-z[4]):
        candidates=[(k,g) for k,g in enumerate(gt) if k not in used and int(g[4])==int(p[5]) and g[5]==p[6]]
        hit=next(((k,g) for k,g in candidates if _iou(p,g)>=iou),None)
        if hit:used.add(hit[0]);tp.append(1)
        else:tp.append(0)
    if not n:return 0.0
    if not tp:return 0.0
    tp=np.cumsum(tp);fp=np.arange(1,len(tp)+1)-tp;rec=tp/n;pre=tp/np.maximum(tp+fp,1)
    return float(np.trapz(np.r_[1.0,np.maximum.accumulate(pre[::-1])[::-1]],np.r_[0,rec]))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--ddm-ckpt',type=Path)
    p.add_argument('--rgbt-root',type=Path,required=True);p.add_argument('--rgbt-cache',type=Path,required=True)
    p.add_argument('--m3fd-root',type=Path,required=True);p.add_argument('--hardneg-manifest',type=Path,required=True)
    p.add_argument('--phase2-package',type=Path,default=Path(__file__).parent.parent/'FieldGuide_Phase2_FRP_20260916')
    p.add_argument('--device',default='cuda:0');p.add_argument('--tau',type=float,default=.045);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--depth-ablation',choices=('full','shuffle','const','rgb_only'),default='full')
    p.add_argument('--shuffle-table',type=Path);p.add_argument('--m3fd-cohort',choices=('m3fd_val','m3fd_test'))
    p.add_argument('--require-veto-only-hook',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(f'Refusing overwrite: {a.out}')
    if a.depth_ablation=='shuffle' and (a.shuffle_table is None or a.m3fd_cohort is None):
        raise ValueError('shuffle needs --shuffle-table and --m3fd-cohort')
    os.environ['PHY_DDM_CUE_MODE']='rgb_only' if a.depth_ablation=='rgb_only' else 'depth'
    shuffle=None
    if a.depth_ablation=='shuffle':
        shuffle=json.loads(a.shuffle_table.read_text(encoding='utf-8'))
        if shuffle.get('seed')!=20261003:raise RuntimeError('Expected preregistered depth shuffle seed 20261003')
    if not a.phase2_package.is_dir():raise FileNotFoundError(f'phase2 package: {a.phase2_package}')
    sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916')]
    from smoke_frp_v2_7 import make
    from field_use_common import paired_paths,prepare_six,field_and_detections
    from tools.ddm_geometry import letterbox_transform
    from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
    from ultralytics.models.yolo.detect.field3d import box_camera_coordinate
    if float(os.getenv('PHY_F3D_SCALE','50'))!=50.:raise RuntimeError('3D output requires nominal PHY_F3D_SCALE=50')
    active=[None]
    hook=DDMFusionHook(depth_provider=lambda _rgb: active[0])
    model,_,_=make(a.weights,a.device,True);hook.attach(model,a.ddm_ckpt)
    if a.require_veto_only_hook and getattr(hook,'fusion_mode',None)!='veto_only':
        raise RuntimeError('Installed DDM hook is not the q-fusion-removed veto-only implementation')
    veto_t=float(os.getenv('PHY_DDM_VETO_T','0'));veto_c=float(os.getenv('PHY_DDM_VETO_C','.25'));veto_enabled=os.getenv('PHY_DDM','0') in {'1','true','True'}
    pred=[];gt=[];cc_total=[];cc_scored=[];pddm=[];lat=[];n_vetoed=[0];detections3d=[];n_post_veto=[0];hardneg_per_image=[]
    def infer(rgb,ir,cache_root,dataset_root,label=None,fp_classes=None,dataset=None,cohort=None):
        x,tr=prepare_six(rgb,ir,torch,a.device)
        if a.depth_ablation=='rgb_only':
            active[0]=torch.full((1,1,640,640),.5,device=a.device)
        else:
            source_rgb=rgb;source_tr=tr
            if a.depth_ablation=='shuffle':
                key=rgb.resolve().relative_to(dataset_root.resolve()).as_posix()
                try:source_key=shuffle['cohorts'][cohort][key]
                except KeyError as e:raise RuntimeError(f'shuffle mapping missing: {cohort}/{key}') from e
                if source_key==key:raise RuntimeError(f'shuffle self-pair: {cohort}/{key}')
                source_rgb=dataset_root/source_key
                with Image.open(source_rgb) as im:source_tr=letterbox_transform(*im.size)
            dpath=_cache(cache_root,dataset_root,source_rgb)
            active[0]=_depth_letterboxed(dpath,source_tr,a.device)
            if a.depth_ablation=='const':active[0]=torch.full_like(active[0],float(active[0].mean()))
        started=time.perf_counter()
        _,det=field_and_detections(model,torch,__import__('ultralytics.utils.nms',fromlist=['non_max_suppression']).non_max_suppression,False,x,tr,a.tau,{0,1,2})
        if x.is_cuda:torch.cuda.synchronize(x.device)
        lat.append(time.perf_counter()-started);aux=model._last_frp_aux
        cc_total.append(int(aux['cc_count_total']));cc_scored.append(int(aux['cc_count_scored']))
        pddm_map=aux.get('p_ddm')
        n=0
        if pddm_map is not None:
            pddm_map=pddm_map[0,0].detach().float().cpu().numpy();pddm.extend(pddm_map.flatten().tolist())
            det,n=_apply_box_veto(det,pddm_map,tr,veto_t,veto_c,veto_enabled);n_vetoed[0]+=n
        n_post_veto[0]+=len(det)
        depth640=active[0][0,0].detach().float().cpu().numpy()
        width,height,nw,nh,px,py=tr
        for box in det:
            x0,y0,x1,y1=map(float,box[:4]);box_lb=[px+x0*nw/width,py+y0*nh/height,px+x1*nw/width,py+y1*nh/height]
            coord=box_camera_coordinate(depth640,box_lb)
            p_mean=None
            if pddm_map is not None:
                sy,sx=_box_slice(box,tr,pddm_map.shape);p_mean=float(pddm_map[sy,sx].mean())
            detections3d.append({'image':str(rgb),'dataset':dataset,'depth_ablation':a.depth_ablation,'box_xyxy':[x0,y0,x1,y1],
                                 'conf':float(box[4]),'cls':int(box[5]),'p_ddm_box_mean':p_mean,
                                 **coord,'scale_note':'nominal, PHY_F3D_SCALE=50, not metric'})
        key=str(rgb)
        if label is not None:
            pred.extend([list(z)+[key] for z in det.tolist()])
            gt.extend([list(z)+[key] for z in _label_boxes_cls(label,tr[0],tr[1],{0,1,2})])
        if fp_classes is None:return len(det)
        counts={c:sum(int(z[5])==c for z in det) for c in fp_classes}
        hardneg_per_image.append({'image':str(rgb),'depth_ablation':a.depth_ablation,
                                  'smoke_fp':counts.get(0,0),'fire_fp':counts.get(1,0),
                                  'event_fp':counts.get(0,0)+counts.get(1,0),
                                  'person_fp':counts.get(2,0),'n_vetoed':n})
        return counts
    rgbt_pairs=list(paired_paths(a.rgbt_root,'val'))
    for rgb,ir,label in rgbt_pairs:infer(rgb,ir,a.rgbt_cache,a.rgbt_root,label,dataset='rgbt_val',cohort='rgbt_val')
    # M3FD caches are deliberately rooted at <m3fd-root>/depth_cache.
    mcache=a.m3fd_root/'depth_cache';hard_fp=0;hard_fp_person=0
    hard_rows=[json.loads(x) for x in a.hardneg_manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    for row in hard_rows:
        fp=infer(a.m3fd_root/row['rgb'],a.m3fd_root/row['ir_gray'],mcache,a.m3fd_root,fp_classes={0,1,2},dataset='m3fd_hardneg',cohort=a.m3fd_cohort)
        hard_fp+=fp[0]+fp[1];hard_fp_person+=fp[2]
    # Standard validation mAP: each class is pooled across all RGBT validation images.
    aps=[];classes=sorted({int(z[4]) for z in gt})
    for th in np.arange(.5,.96,.05):
        class_ap=[_ap([z for z in pred if int(z[5])==c],[z for z in gt if int(z[4])==c],float(th)) for c in classes]
        aps.append(float(np.mean(class_ap)) if class_ap else 0.)
    out={'weights':str(a.weights),'ddm_ckpt':str(a.ddm_ckpt) if a.ddm_ckpt else None,'tau':a.tau,
         'depth_ablation':a.depth_ablation,'shuffle_table':str(a.shuffle_table) if a.shuffle_table else None,
         'hook_fusion_mode':getattr(hook,'fusion_mode','legacy_q_fusion'),
         'phy_ddm':os.getenv('PHY_DDM'),'phy_ddm_w1':os.getenv('PHY_DDM_W1'),'phy_ddm_w2':os.getenv('PHY_DDM_W2'),'phy_ddm_w3':os.getenv('PHY_DDM_W3'),
         'phy_ddm_veto_t':veto_t,'phy_ddm_veto_c':veto_c,'n_vetoed':n_vetoed[0],
         'mAP50':float(aps[0]),'AP50_95':float(np.mean(aps)),'hardneg_fp':int(hard_fp),'hardneg_fp_person':int(hard_fp_person),'cc_count_total_mean':float(np.mean(cc_total)),'cc_count_scored_mean':float(np.mean(cc_scored)),
         'fps':float(len(lat)/sum(lat)) if lat else 0.,'p_ddm_mean':float(np.mean(pddm)) if pddm else None,'p_ddm_max':float(np.max(pddm)) if pddm else None,
         'n_rgbt_images':len(rgbt_pairs),'n_hardneg':len(hard_rows),
         'n_detections3d':len(detections3d),
         'z_median_rgbt':float(np.median([z['Z'] for z in detections3d if z['dataset']=='rgbt_val'])) if any(z['dataset']=='rgbt_val' for z in detections3d) else None,
         'z_median_hardneg':float(np.median([z['Z'] for z in detections3d if z['dataset']=='m3fd_hardneg'])) if any(z['dataset']=='m3fd_hardneg' for z in detections3d) else None}
    if len(detections3d)!=n_post_veto[0]:raise RuntimeError('3D output count differs from post-veto detections')
    if len(hardneg_per_image)!=len(hard_rows) or sum(row['event_fp'] for row in hardneg_per_image)!=hard_fp:
        raise RuntimeError('Hard-negative per-image FP rows do not match aggregate')
    a.out.mkdir(parents=True)
    (a.out/'detections3d.jsonl').write_text(''.join(json.dumps(z,allow_nan=False)+'\n' for z in detections3d),encoding='utf-8')
    (a.out/'hardneg_per_image.jsonl').write_text(''.join(json.dumps(z,allow_nan=False)+'\n' for z in hardneg_per_image),encoding='utf-8')
    (a.out/'metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('[DDM-VAL] '+json.dumps(out))
if __name__=='__main__':main()
