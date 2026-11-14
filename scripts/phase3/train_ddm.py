"""Train only DDM3Cue(L3) + DDMBranch from cached RGB/depth pairs."""
import argparse,csv,json,math,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tools.ddm_dataset import RGBTDDM,YHTDDM,M3FDDM
from ultralytics.nn.modules.ddm_cues import DDM3Cue
from ultralytics.nn.modules.ddm_branch import DDMBranch

def auc_ap(scores,target):
 s=scores.detach().flatten().float().cpu();y=target.detach().flatten().float().cpu();order=torch.argsort(s,descending=True);y=y[order];pos=y.sum().item();neg=len(y)-pos
 if not pos or not neg:return float('nan'),float('nan')
 tp=torch.cumsum(y,0);fp=torch.arange(1,len(y)+1)-tp;auc=float((1-fp[y.bool()].sum()/(pos*neg)).item());ap=float((tp[y.bool()]/(tp[y.bool()]+fp[y.bool()])).sum().item()/pos);return auc,ap
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--phase2-package',type=Path,required=True);p.add_argument('--rgbt-root',type=Path,required=True);p.add_argument('--rgbt-cache',type=Path,required=True);p.add_argument('--yht-rgb',type=Path,required=True);p.add_argument('--yht-cache',type=Path,required=True);p.add_argument('--yht-labels',type=Path,required=True);p.add_argument('--m3fd-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--rgbt-yht-ratio',default='2:4');p.add_argument('--m3fd-ratio',type=int,default=1);p.add_argument('--seed',type=int,default=20260922);p.add_argument('--w-rgbt',type=float,default=1);p.add_argument('--w-yht-mil',type=float,default=1);p.add_argument('--w-yht-zero',type=float,default=1);p.add_argument('--w-m3fd-zero',type=float,default=1);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device(a.device);sys.path.insert(0,str(a.phase2_package));from smoke_frp_v2_7 import make
 model,_,_=make(Path(a.weights),a.device,True);model.eval();[z.requires_grad_(False) for z in model.parameters()]
 tr=RGBTDDM(a.rgbt_root,a.rgbt_cache,'train');va=RGBTDDM(a.rgbt_root,a.rgbt_cache,'val');yh=YHTDDM(a.yht_rgb,a.yht_cache,a.yht_labels);mt=M3FDDM(a.m3fd_root,'train');mv=M3FDDM(a.m3fd_root,'val')
 assert len(tr) and len(va) and len(yh) and len(mt) and len(mv),'empty dataset after depth alignment'
 a.out.mkdir(parents=True)
 gr=torch.Generator().manual_seed(a.seed);gy=torch.Generator().manual_seed(a.seed+1);gm=torch.Generator().manual_seed(a.seed+3)
 lr=DataLoader(tr,a.batch,shuffle=True,num_workers=4,pin_memory=True,generator=gr);ly=DataLoader(yh,a.batch,shuffle=True,num_workers=4,pin_memory=True,generator=gy);lm=DataLoader(mt,a.batch,shuffle=True,num_workers=4,pin_memory=True,generator=gm);lv=DataLoader(va,a.batch,shuffle=False,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(a.seed+2));lvm=DataLoader(mv,a.batch,shuffle=False,num_workers=4,pin_memory=True)
 cues=DDM3Cue().to(dev);branch=None;opt=None;best=-1;best_m3fd_mean=None;rows=[];r,y=map(int,a.rgbt_yht_ratio.split(':'));m=a.m3fd_ratio
 if min(r,y,m)<1:raise ValueError('RGBT:YHT:M3FD ratios must be positive')
 cycles=max(math.ceil(len(lr)/r),math.ceil(len(ly)/y),math.ceil(len(lm)/m));steps=cycles*(r+y+m)
 print(f'[DDM-V2-MIX] requested RGBT:YHT:M3FD={r}:{y}:{m}; steps/epoch={steps}; actual={cycles*r}:{cycles*y}:{cycles*m}; frames={len(tr)}:{len(yh)}:{len(mt)}')
 def forward(rgb,depth):
  nonlocal branch,opt
  # The frozen backbone may use AMP; DDM cues/branch and their BCE remain fp32.
  with torch.no_grad():
   with torch.autocast(dev.type,enabled=dev.type=='cuda'):p3=model._rgb_backbone(rgb)[1]
  p3=p3.float()
  l1,l2,l3=cues(rgb,depth); 
  if branch is None:
   branch=DDMBranch(p3.shape[1]).to(dev);n=sum(z.numel() for z in list(cues.parameters())+list(branch.parameters()));assert n<1600000,n;print(f'[DDM-TRAINABLE] {n:,}')
   opt=torch.optim.AdamW(list(cues.parameters())+list(branch.parameters()),lr=a.lr,weight_decay=.01)
  return branch(l1,l2,l3,p3)
 for epoch in range(1,a.epochs+1):
  cues.train();branch.train() if branch else None;losses=[];it_r=iter(lr);it_y=iter(ly);it_m=iter(lm)
  for step in range(steps):
   slot=step%(r+y+m);kind='rgbt' if slot<r else 'yht' if slot<r+y else 'm3fd'
   try:batch=next(it_r if kind=='rgbt' else it_y if kind=='yht' else it_m)
   except StopIteration:
    if kind=='rgbt':it_r=iter(lr);batch=next(it_r)
    elif kind=='yht':it_y=iter(ly);batch=next(it_y)
    else:it_m=iter(lm);batch=next(it_m)
   rgb=batch['rgb'].to(dev);depth=batch['depth'].to(dev)
   pred=forward(rgb,depth)
   assert torch.isfinite(pred).all(), f'non-finite pred, paths={batch["path"]}'
   if kind=='rgbt':
    target=F.interpolate(batch['target'].to(dev),(pred.shape[-2:]),mode='nearest').float();loss=a.w_rgbt*F.binary_cross_entropy(pred,target)
   elif kind=='yht':
    target=batch['target'].to(dev).float();loss=(a.w_yht_mil*F.binary_cross_entropy(pred.amax((-2,-1)).flatten(),target) if target.any() else a.w_yht_zero*F.binary_cross_entropy(pred,torch.zeros_like(pred)))
   else:
    loss=a.w_m3fd_zero*F.binary_cross_entropy(pred,torch.zeros_like(pred))
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
  cues.eval();branch.eval();ss=[];tt=[]
  with torch.no_grad():
   for batch in lv:
    pred=forward(batch['rgb'].to(dev),batch['depth'].to(dev));ss.append(pred.cpu());tt.append(F.interpolate(batch['target'],pred.shape[-2:],mode='nearest'))
  auc,ap=auc_ap(torch.cat(ss),torch.cat(tt))
  if not np.isfinite(ap):raise RuntimeError(f'RGBT val AP non-finite at epoch {epoch}; refusing checkpoint selection')
  row={'epoch':epoch,'loss':float(np.mean(losses)),'val_auc':auc,'val_ap':ap,'m3fd_val_p_mean':None};rows.append(row)
  if ap>best:
   best=ap;hard=[]
   with torch.no_grad():
    for batch in lvm:
     pred=forward(batch['rgb'].to(dev),batch['depth'].to(dev))
     hard.extend({'path':path,'p_ddm_mean':float(value)} for path,value in zip(batch['path'],pred.mean((-2,-1)).flatten().cpu()))
   best_m3fd_mean=float(np.mean([z['p_ddm_mean'] for z in hard]));row['m3fd_val_p_mean']=best_m3fd_mean
   (a.out/'m3fd_val_per_frame.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in hard))
   torch.save({'cues':cues.state_dict(),'branch':branch.state_dict(),'p3_channels':branch.net[0].in_channels-3,'epoch':epoch,'val_ap':ap},a.out/'best.pt')
  torch.save({'cues':cues.state_dict(),'branch':branch.state_dict(),'p3_channels':branch.net[0].in_channels-3,'epoch':epoch,'val_ap':ap},a.out/'last.pt');print(json.dumps(row))
 with (a.out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (a.out/'summary.json').write_text(json.dumps({'train_rgbt':len(tr),'val_rgbt':len(va),'yht':len(yh),'train_m3fd':len(mt),'val_m3fd':len(mv),'rgbt_skipped':tr.skipped,'yht_skipped':yh.skipped,'best_val_ap':best,'best_m3fd_val_p_mean':best_m3fd_mean,'m3fd_val_report_only':True,'actual_steps_per_epoch':{'rgbt':cycles*r,'yht':cycles*y,'m3fd':cycles*m}},indent=2))
if __name__=='__main__':main()
