"""Resumable uint16 depth cache builder with optional depth-direction visual QA."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics.nn.modules.depth_service import DepthService
def valid_png(path):
 try:
  with Image.open(path) as im:return im.mode in {'I;16','I'} and im.size[0]>0
 except Exception:return False
def visual(rgb,depth,path):
 import matplotlib.pyplot as plt
 path.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].imshow(rgb);ax[0].set_title('RGB');ax[1].imshow(depth,cmap='magma',vmin=0,vmax=1);ax[1].set_title('relative depth: 1=far');[a.axis('off') for a in ax];fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch',type=int,default=4);p.add_argument('--smoke-out',type=Path);p.add_argument('--smoke-n',type=int,default=10);a=p.parse_args()
 paths=[x for x in a.images.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}];out=a.dataset_root/'depth_cache';out.mkdir(parents=True,exist_ok=True);todo=[];skipped=0
 for src in paths:
  dst=out/src.relative_to(a.dataset_root).with_suffix('.png')
  if valid_png(dst):skipped+=1
  else:todo.append((src,dst))
 svc=DepthService(a.device);t=time.perf_counter();created=0;visuals=[]
 for i in range(0,len(todo),a.batch):
  group=todo[i:i+a.batch];rgb=[np.asarray(Image.open(src).convert('RGB')) for src,_ in group];ds=svc.infer(rgb)
  for src,d,image in zip([x[0] for x in group],ds,rgb):
   dst=out/src.relative_to(a.dataset_root).with_suffix('.png');dst.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((d.numpy().clip(0,1)*65535).astype(np.uint16)).save(dst);created+=1
   if a.smoke_out and len(visuals)<a.smoke_n:visual(image,d.numpy(),a.smoke_out/f'{len(visuals):03d}_{src.stem}.png');visuals.append(str(src))
 meta={'model':svc.MODEL_ID,'images_total':len(paths),'created':created,'skipped':skipped,'seconds':time.perf_counter()-t,'format':'uint16 PNG; depth=uint16/65535; 1=far','visuals':visuals};(out/'depth_cache_meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
if __name__=='__main__':main()
