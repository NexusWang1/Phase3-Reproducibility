"""Render Phase3 YHT field/3D demo video frames and M3FD veto comparisons."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def _frame_id(path):
 import re
 n=re.findall(r'\d+',Path(path).stem);return int(n[-1]) if n else None
def _images(root):return sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
def _vector_3d(field,depth640,backproject):
 s=F.interpolate(torch.from_numpy(field)[None,None],(640,640),mode='bilinear',align_corners=False)[0,0].numpy()
 pts=backproject(depth640,s,threshold=.05,topk=10000)
 if len(pts)<4:return pts,np.zeros(2)
 w=pts[:,3];center=np.average(pts[:,[0,2]],axis=0,weights=w);core=pts[w>=np.percentile(w,90)][:, [0,2]].mean(0)
 return pts,center-core
def _render_yht(path,rgb,ir,det,kept,vetoed,field,pddm,points,vec,history,components,boxes3d,c_dir):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(16,12),dpi=120);h,w=np.asarray(rgb).shape[:2]
 ax[0,0].imshow(rgb);ax[0,0].set_title('RGB + detection / DDM veto')
 for z,coord in zip(kept,boxes3d):
  x0,y0,x1,y1,cf,c=z;ax[0,0].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='lime',lw=2));ax[0,0].text(x0,y0,f'{int(c)} {cf:.2f} Z={coord["Z"]:.1f}',color='lime',bbox={'facecolor':'black','alpha':.5})
 for z,mean in vetoed:
  x0,y0,x1,y1,cf,c=z;ax[0,0].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='red',ls='--',lw=2));ax[0,0].text(x0,y0,f'veto {cf:.2f}/{mean:.3f}',color='red',bbox={'facecolor':'black','alpha':.5})
 ax[0,0].axis('off');ax[0,1].imshow(ir,cmap='gray');ax[0,1].imshow(field,cmap='inferno',alpha=.6,extent=(0,w,h,0));ax[0,1].set_title(f"Purified field S′ | components={components}");ax[0,1].axis('off')
 im=ax[1,0].imshow(pddm,cmap='viridis',vmin=0,vmax=1);ax[1,0].set_title(f'P_ddm mean={pddm.mean():.3f} max={pddm.max():.3f}');ax[1,0].axis('off');fig.colorbar(im,ax=ax[1,0],fraction=.046)
 if len(points):ax[1,1].scatter(points[:,0],points[:,2],c=points[:,3],s=1,cmap='inferno');ax[1,1].quiver(0,0,vec[0]*3,vec[1]*3,color='red',angles='xy',scale_units='xy',scale=1,width=.008)
 units=np.asarray([v/np.linalg.norm(v) for v in history if np.linalg.norm(v)])
 R=float(np.linalg.norm(units.mean(0))) if len(units) else 0.;angle=float(np.degrees(np.arctan2(vec[1],vec[0]))) if np.linalg.norm(vec) else 0.
 ax[1,1].set_title(f'3D ground plane | deflection={angle:.1f}° R={R:.2f} c_dir={c_dir:.2f}');ax[1,1].set_xlabel('X (nominal)');ax[1,1].set_ylabel('Z (nominal)');fig.tight_layout();fig.savefig(path);plt.close(fig)
 return angle,R
def _render_m3fd(path,rgb,box,mean,kept):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(14,6),dpi=120);x0,y0,x1,y1,cf,c=box
 for a,title,on in zip(ax,['Veto off: FP retained','Veto on: low-DDM FP removed'],[True,False]):
  a.imshow(rgb);color='red' if on else 'gray';style='-' if on else '--';a.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color=color,ls=style,lw=3));a.text(x0,y0,('FP ' if on else 'REMOVED ')+f'{cf:.2f} pddm={mean:.4f}',color=color,bbox={'facecolor':'white','alpha':.8});
  if not on:a.plot([x0,x1],[y0,y1],color='gray',lw=3);a.plot([x0,x1],[y1,y0],color='gray',lw=3)
  a.set_title(title);a.axis('off')
 fig.tight_layout();fig.savefig(path);plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--ddm-ckpt',type=Path,required=True);p.add_argument('--rgb-root',type=Path,required=True);p.add_argument('--ir-root',type=Path,required=True);p.add_argument('--depth-cache',type=Path,required=True);p.add_argument('--m3fd-root',type=Path,required=True);p.add_argument('--fp-records',type=Path,required=True);p.add_argument('--phase2-package',type=Path,default=Path(__file__).resolve().parents[1].parent/'FieldGuide_Phase2_FRP_20260916');p.add_argument('--device',default='cuda:0');p.add_argument('--tau',type=float,default=.045);p.add_argument('--start',type=int,default=13680);p.add_argument('--end',type=int,default=14500);p.add_argument('--stride',type=int,default=8);p.add_argument('--ignition-frame',type=int,default=13771);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(f'Refusing overwrite: {a.out}')
 os.environ['PHY_DDM']='1';os.environ.setdefault('PHY_DDM_VETO_T','.005');os.environ.setdefault('PHY_DDM_VETO_C','.25');os.environ.setdefault('PHY_DDM_FIELD_VETO_T','0');os.environ['PHY_FRP_TAU']=str(a.tau)
 if not a.phase2_package.is_dir():raise FileNotFoundError(a.phase2_package)
 sys.path[:0]=[str(a.phase2_package),str(a.phase2_package.parent/'FieldGuide_Phase2_FieldUse_20260916'),str(Path(__file__).resolve().parents[1])]
 from smoke_frp_v2_7 import make
 from field_use_common import prepare_six,field_and_detections
 from eval_ddm_val import _cache,_depth_letterboxed,_apply_box_veto,_box_slice
 from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
 from ultralytics.models.yolo.detect.field3d import backproject_field,field_centroids,box_camera_coordinate,deflection_confidence
 if float(os.getenv('PHY_F3D_SCALE','50'))!=50.:raise RuntimeError('3D output requires nominal PHY_F3D_SCALE=50')
 outframes=a.out/'yht_frames';compare=a.out/'m3fd_veto_compare';outframes.mkdir(parents=True);compare.mkdir()
 rgb={_frame_id(x):x for x in _images(a.rgb_root) if _frame_id(x) is not None};ir={_frame_id(x):x for x in _images(a.ir_root) if _frame_id(x) is not None}
 active=[None];model,_,_=make(a.weights,a.device,True);DDMFusionHook(depth_provider=lambda _rgb:active[0]).attach(model,a.ddm_ckpt);history=[];frames=[];skipped=[]
 for n in [k for k in sorted(rgb) if a.start<=k<=a.end and (k-a.start)%a.stride==0]:
  if n not in ir:skipped.append(n);continue
  dpath=_cache(a.depth_cache,a.rgb_root,rgb[n])
  if not dpath.is_file():skipped.append(n);continue
  x,tr=prepare_six(rgb[n],ir[n],torch,a.device);active[0]=_depth_letterboxed(dpath,tr,a.device)
  _,det=field_and_detections(model,torch,__import__('ultralytics.utils.nms',fromlist=['non_max_suppression']).non_max_suppression,False,x,tr,a.tau,{0,1,2});aux=model._last_frp_aux;pddm=aux['p_ddm'][0,0].detach().float().cpu().numpy();kept,nv=_apply_box_veto(det,pddm,tr,float(os.environ['PHY_DDM_VETO_T']),float(os.environ['PHY_DDM_VETO_C']),True)
  vetoed=[]
  for z in det:
   if not any(np.array_equal(z,k) for k in kept):sy,sx=_box_slice(z,tr,pddm.shape);vetoed.append((z,float(pddm[sy,sx].mean())))
  pur=getattr(model,'_last_t_pur',None)
  if pur is None:raise RuntimeError('purified field tap unavailable')
  field=pur[0,0].detach().float().cpu().numpy();depth640=active[0][0,0].detach().float().cpu().numpy();points,vec=_vector_3d(field,depth640,backproject_field)
  c_dir=deflection_confidence(field,vec,history,kept_classes={int(z[5]) for z in kept});history.append(vec);history=history[-50:]
  width,height,nw,nh,px,py=tr;boxes3d=[]
  for box in kept:
   x0,y0,x1,y1=map(float,box[:4]);box_lb=[px+x0*nw/width,py+y0*nh/height,px+x1*nw/width,py+y1*nh/height]
   sy,sx=_box_slice(box,tr,pddm.shape);coord=box_camera_coordinate(depth640,box_lb)
   boxes3d.append({'image':str(rgb[n]),'dataset':'yht_demo','box_xyxy':[x0,y0,x1,y1],
                   'conf':float(box[4]),'cls':int(box[5]),'p_ddm_box_mean':float(pddm[sy,sx].mean()),
                   **coord,'scale_note':'nominal, PHY_F3D_SCALE=50, not metric'})
  angle,R=_render_yht(outframes/f'{n:06d}.png',Image.open(rgb[n]).convert('RGB'),Image.open(ir[n]).convert('L'),det,kept,vetoed,field,pddm,points,vec,history,len(field_centroids(field)),boxes3d,c_dir);frames.append({'frame_id':n,'vetoed':nv,'kept_count':len(kept),'angle_deg':angle,'R':R,'c_dir':c_dir,'boxes3d':boxes3d})
  print(f'[PHASE3-DEMO] YHT frame={n} saved={len(frames)} skipped={len(skipped)}',flush=True)
 # Visualize only smoke/fire records that the registered v1.1 rule removes.
 rows=[json.loads(x) for x in a.fp_records.read_text().splitlines() if x.strip()];chosen=[r for r in rows if r['class'] in {0,1} and r['confidence']<float(os.environ['PHY_DDM_VETO_C']) and r['p_ddm_mean']<float(os.environ['PHY_DDM_VETO_T'])];chosen=sorted(chosen,key=lambda r:r['confidence'],reverse=True)[:6]
 rendered=[]
 for i,r in enumerate(chosen):
  rgbpath=a.m3fd_root/r['rgb'];
  if not rgbpath.is_file():continue
  _render_m3fd(compare/f'{i+1:02d}_{Path(r["rgb"]).stem}.png',Image.open(rgbpath).convert('RGB'),r['box_xyxy']+[r['confidence'],r['class']],r['p_ddm_mean'],False);rendered.append(r['rgb'])
 # H264 is optional; PNG frames remain a complete portable handoff.
 video=None
 try:
  import imageio.v2 as imageio
  video=a.out/'yht_demo.mp4'
  with imageio.get_writer(video,fps=8,codec='libx264') as writer:
   for frame in sorted(outframes.glob('*.png')):writer.append_data(imageio.imread(frame))
 except Exception as e:print(f'[PHASE3-DEMO] mp4 unavailable; PNG sequence retained: {e}')
 pre=[row['c_dir'] for row in frames if row['frame_id']<a.ignition_frame];post=[row['c_dir'] for row in frames if row['frame_id']>=a.ignition_frame]
 comparison={'ignition_frame':a.ignition_frame,'pre_fire_n':len(pre),'pre_fire_mean':float(np.mean(pre)) if pre else None,'after_ignition_n':len(post),'after_ignition_mean':float(np.mean(post)) if post else None}
 meta={'frames_saved':len(frames),'frames_skipped':len(skipped),'start':a.start,'end':a.end,'stride':a.stride,'field_veto_t':os.environ['PHY_DDM_FIELD_VETO_T'],'box_veto_t':os.environ['PHY_DDM_VETO_T'],'box_veto_c':os.environ['PHY_DDM_VETO_C'],'m3fd_compares':len(rendered),'video':str(video) if video and video.is_file() else None,'c_dir_comparison':comparison,'frames':frames,'skipped':skipped};(a.out/'demo_meta.json').write_text(json.dumps(meta,indent=2));print('[PHASE3-C_DIR] '+json.dumps(comparison));print('[PHASE3-DEMO] '+json.dumps({k:meta[k] for k in ('frames_saved','frames_skipped','m3fd_compares','video')}))
if __name__=='__main__':main()
