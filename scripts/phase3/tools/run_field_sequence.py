"""Generate ordered, offline-depth YHT purified-field sequences for direction eval."""
from __future__ import annotations
import argparse,json,os,sys,types
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--ddm-ckpt',type=Path,required=True);p.add_argument('--rgb-root',type=Path,required=True);p.add_argument('--ir-root',type=Path,required=True);p.add_argument('--depth-cache',type=Path,required=True);p.add_argument('--yht-labels',type=Path,required=True);p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--stride',type=int,default=1);p.add_argument('--phase2-package',type=Path,default=Path(__file__).resolve().parents[1].parent/'FieldGuide_Phase2_FRP_20260916');p.add_argument('--device',default='cuda:0');p.add_argument('--tau',type=float,default=.045);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(f'Refusing overwrite: {a.out}')
 if a.stride<1:raise ValueError('--stride must be >=1')
 os.environ.setdefault('PHY_DDM','1');os.environ['PHY_FRP_TAU']=str(a.tau)
 if not a.phase2_package.is_dir():raise FileNotFoundError(a.phase2_package)
 sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916'),str(Path(__file__).resolve().parents[1])]
 from smoke_frp_v2_7 import make
 from field_use_common import prepare_six
 from tools.ddm_dataset import images,frame_id,parse_yht_labels
 from eval_ddm_val import _cache,_depth_letterboxed
 from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
 from ultralytics.models.yolo.detect.field3d import field_centroids
 labels=parse_yht_labels(a.yht_labels);positive=sorted(k for k,v in labels.items() if v)
 if not positive:raise RuntimeError('YHT labels contain no positive frame')
 ignition=positive[0];start=a.start if a.start is not None else ignition-50;end=a.end if a.end is not None else ignition+1000
 rgb={frame_id(x):x for x in images(a.rgb_root) if frame_id(x) is not None};ir={frame_id(x):x for x in images(a.ir_root) if frame_id(x) is not None}
 selected=[n for n in sorted(rgb) if start<=n<=end and (n-start)%a.stride==0]
 if not selected:raise RuntimeError(f'No RGB frames in requested range {start}..{end}')
 a.out.mkdir(parents=True);fields=a.out/'fields';fields.mkdir();active=[None];model,_,_=make(a.weights,a.device,True);DDMFusionHook(depth_provider=lambda _rgb:active[0]).attach(model,a.ddm_ckpt)
 # The model normally exposes this tap.  Keep an instance-local capture as a
 # compatibility fallback without changing the hook or frozen FRP module.
 captured=[None];forward=model.frp.forward
 def capture(_frp,t,i2p,r2):
  pur,aux=forward(t,i2p,r2);captured[0]=pur;return pur,aux
 model.frp.forward=types.MethodType(capture,model.frp)
 frames=[];skips=[]
 for pos,n in enumerate(selected,1):
  if n not in ir:skips.append({'frame_id':n,'reason':'missing_ir'});continue
  dpath=_cache(a.depth_cache,a.rgb_root,rgb[n])
  if not dpath.is_file():skips.append({'frame_id':n,'reason':'missing_depth'});continue
  try:
   x,tr=prepare_six(rgb[n],ir[n],torch,a.device);active[0]=_depth_letterboxed(dpath,tr,a.device);captured[0]=None
   with torch.inference_mode():model(x)
   pur=getattr(model,'_last_t_pur',None)
   if pur is None:pur=captured[0]
   if pur is None:raise RuntimeError('model did not expose purified field')
   field=pur[0,0].detach().float().cpu().numpy();np.save(fields/f'{n:06d}.npy',field)
   frames.append({'frame_id':n,'file':f'{n:06d}.npy','shape':list(field.shape),'components':len(field_centroids(field))})
  except Exception as e:skips.append({'frame_id':n,'reason':'inference_error','detail':str(e)});continue
  if pos%50==0:print(f'[FIELD-SEQUENCE] {pos}/{len(selected)} saved={len(frames)} skipped={len(skips)}',flush=True)
 meta={'weights':str(a.weights),'ddm_ckpt':str(a.ddm_ckpt),'phy_ddm':os.getenv('PHY_DDM'),'phy_ddm_field_veto_t':os.getenv('PHY_DDM_FIELD_VETO_T','0'),'tau':a.tau,'ignition_frame':ignition,'start':start,'end':end,'stride':a.stride,'requested_frames':len(selected),'saved_frames':len(frames),'skipped_frames':len(skips),'component_count_mean':float(np.mean([x['components'] for x in frames])) if frames else None,'frames':frames,'skips':skips}
 (a.out/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('[FIELD-SEQUENCE] '+json.dumps({k:meta[k] for k in ('saved_frames','skipped_frames','component_count_mean','phy_ddm_field_veto_t')}))
if __name__=='__main__':main()
