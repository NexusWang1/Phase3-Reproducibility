"""YHT paired-frame direction drift under known input rotations/translations."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from tools.ddm_dataset import images,frame_id
from eval_ddm_val import _cache,_depth_letterboxed

def perturb(x,depth,angle=0.,dx=0.,dy=0.):
 rad=np.deg2rad(angle);c=float(np.cos(rad));s=float(np.sin(rad));size=x.shape[-1]
 affine=x.new_tensor([[c,-s,-2*dx/size],[s,c,-2*dy/size]])[None]
 grid=F.affine_grid(affine,x.shape,align_corners=False)
 fill=114/255
 xp=F.grid_sample(x-fill,grid,align_corners=False,padding_mode='zeros')+fill
 dp=F.grid_sample(depth-1,grid,align_corners=False,padding_mode='zeros')+1
 return xp,dp

def angle_between(a,b):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if min(np.linalg.norm(a),np.linalg.norm(b))<1e-9:return None
 return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--ddm-ckpt',type=Path,required=True);p.add_argument('--phase2-package',type=Path,required=True);p.add_argument('--rgb-root',type=Path,required=True);p.add_argument('--ir-root',type=Path,required=True);p.add_argument('--depth-cache',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--stride',type=int,default=8);p.add_argument('--pair-gap',type=int,default=5);p.add_argument('--max-pairs',type=int,default=20);p.add_argument('--device',default='cuda:1');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 if os.getenv('PHY_DDM')!='1':raise RuntimeError('PHY_DDM=1 required')
 sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916')]
 from smoke_frp_v2_7 import make
 from field_use_common import prepare_six
 from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
 from ultralytics.models.yolo.detect.field3d import estimate_direction_from_fields
 rgb={frame_id(f):f for f in images(a.rgb_root)};ir={frame_id(f):f for f in images(a.ir_root)}
 ids=[i for i in sorted(rgb) if i in ir and a.start<=i<=a.end and (i-a.start)%a.stride==0]
 if len(ids)<=a.pair_gap:raise RuntimeError('Not enough paired YHT frames')
 active=[None];model,_,_=make(a.weights,a.device,True);DDMFusionHook(depth_provider=lambda _:active[0]).attach(model,a.ddm_ckpt)
 variants=[(f'rot_{x:+d}',float(x),0.,0.) for x in (-5,-2,2,5)]
 variants += [(f'trans_x_{x:+d}',0.,float(x),0.) for x in (-16,-8,8,16)]
 variants += [(f'trans_y_{x:+d}',0.,0.,float(x)) for x in (-16,-8,8,16)]
 def infer(fid,angle=0.,dx=0.,dy=0.):
  x,tr=prepare_six(rgb[fid],ir[fid],torch,a.device)
  active[0]=_depth_letterboxed(_cache(a.depth_cache,a.rgb_root,rgb[fid]),tr,a.device)
  if angle or dx or dy:x,active[0]=perturb(x,active[0],angle,dx,dy)
  with torch.inference_mode():model(x)
  field=model._last_t_pur
  if field is None:raise RuntimeError(f'No purified field for {fid}')
  return field[0,0].detach().float().cpu().numpy()
 rows=[]
 for index in range(min(a.max_pairs,len(ids)-a.pair_gap)):
  previous,current=ids[index],ids[index+a.pair_gap]
  base=np.asarray(estimate_direction_from_fields(infer(previous),infer(current))['direction']['vector'])
  if np.linalg.norm(base)<1e-9:continue
  for tag,angle,dx,dy in variants:
   vec=np.asarray(estimate_direction_from_fields(infer(previous,angle,dx,dy),infer(current,angle,dx,dy))['direction']['vector'])
   rad=np.deg2rad(angle);c,s=np.cos(rad),np.sin(rad)
   corrected=np.array([[c,-s],[s,c]])@vec
   rows.append({'previous':previous,'current':current,'perturbation':tag,'rotation_deg':angle,
                'dx_px':dx,'dy_px':dy,'raw_drift_deg':angle_between(base,vec),
                'camera_corrected_drift_deg':angle_between(base,corrected)})
  print(f'[YHT-POSE] pair {previous}->{current}',flush=True)
 if not rows:raise RuntimeError('No nonzero direction pairs')
 summary={}
 for tag,_,_,_ in variants:
  group=[r for r in rows if r['perturbation']==tag]
  for metric in ('raw_drift_deg','camera_corrected_drift_deg'):
   values=[r[metric] for r in group if r[metric] is not None]
   summary.setdefault(tag,{})[metric+'_median']=float(np.median(values)) if values else None
  summary[tag]['n']=len(group)
 a.out.mkdir(parents=True)
 (a.out/'pose_robustness.json').write_text(json.dumps({'n_input_pairs':len({(r['previous'],r['current']) for r in rows}),
    'pair_gap':a.pair_gap,'size':640,'protocol':'same affine applied to RGB, IR and depth after shared letterbox; raw and inverse-rotation-corrected camera-frame drift','summary':summary,'rows':rows},indent=2),encoding='utf-8')
 print('[YHT-POSE] '+json.dumps(summary),flush=True)
if __name__=='__main__':main()
