"""Cache-only paired RGB/depth datasets and DDM supervision targets."""
from __future__ import annotations
import csv
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tools.ddm_geometry import rgb_letterboxed, depth_letterboxed

IMAGE_EXT={'.jpg','.jpeg','.png','.bmp'}
def images(root):return sorted(x for x in Path(root).rglob('*') if x.suffix.lower() in IMAGE_EXT)
def tensor_rgb(path,size=640):
 return rgb_letterboxed(path,size)[0]
def tensor_depth(path,tr,size=640):
 return depth_letterboxed(path,tr,'cpu',size)[0]
def raster_yolo(label,tr,size=640,event=frozenset({0,1})):
 t=torch.zeros(1,size,size)
 if not label.is_file():return t
 width,height,nw,nh,px,py=tr
 for line in label.read_text(encoding='utf-8-sig').splitlines():
  v=line.split()
  if len(v)!=5 or int(float(v[0])) not in event:continue
  _,cx,cy,w,h=map(float,v)
  x0=max(0,int(px+(cx-w/2)*nw));x1=min(size,int(px+(cx+w/2)*nw))
  y0=max(0,int(py+(cy-h/2)*nh));y1=min(size,int(py+(cy+h/2)*nh))
  t[:,y0:y1,x0:x1]=1
 return t
def cache_path(cache_root,relative):return (Path(cache_root)/relative).with_suffix('.png')
def parse_yht_labels(path):
 """Accept frame-index lines/ranges; positive tokens include fire/smoke/YY/YN."""
 text=Path(path).read_text(encoding='utf-8-sig',errors='ignore');out={}
 for line in text.splitlines():
  if not re.match(r'^\s*\d',line):continue
  nums=[int(x) for x in re.findall(r'\d+',line)]
  if not nums:continue
  low=line.lower();positive=any(x in low for x in ('fire','smoke','火','烟','yy','yn','ny'))
  negative=any(x in low for x in ('none','normal','无','nn'))
  value=1 if positive and not negative else 0
  start,end=(nums[0],nums[-1]) if len(nums)>1 else (nums[0],nums[0])
  for i in range(start,end+1):out[i]=value
 return out
def frame_id(path):
 n=re.findall(r'\d+',Path(path).stem);return int(n[-1]) if n else None

class RGBTDDM(Dataset):
 def __init__(self,root,cache_root,split='train',size=640):
  self.root=Path(root);self.cache=Path(cache_root);self.size=size;self.rows=[];base=self.root/'images'/split
  self.split=split;event={0,1}
  for rgb in images(base):
   rel=rgb.relative_to(self.root);depth=cache_path(self.cache,rel);label=(self.root/'labels'/rgb.relative_to(self.root/'images')).with_suffix('.txt')
   if depth.is_file():self.rows.append((rgb,depth,label))
  self.skipped=sum(1 for _ in images(base))-len(self.rows)
  self.pos_frames=sum(1 for _,_,lb in self.rows if lb.is_file() and lb.stat().st_size>0)
  if self.pos_frames==0:raise RuntimeError(f'all targets empty - label path broken: {self.root}')
  print(f'[RGBTDDM] split={self.split} rows={len(self.rows)} pos_frames={self.pos_frames} event={sorted(event)}')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  rgb,d,label=self.rows[i];x,tr=rgb_letterboxed(rgb,self.size);return {'rgb':x,'depth':tensor_depth(d,tr,self.size),'target':raster_yolo(label,tr,self.size),'kind':'rgbt','path':str(rgb)}

class YHTDDM(Dataset):
 def __init__(self,rgb_root,cache_root,labels,size=640):
  self.root=Path(rgb_root);self.cache=Path(cache_root);self.size=size;lab=parse_yht_labels(labels);self.rows=[]
  for rgb in images(self.root):
   depth=cache_path(self.cache,rgb.relative_to(self.root));fid=frame_id(rgb)
   if depth.is_file() and fid in lab:self.rows.append((rgb,depth,lab[fid]))
  self.skipped=sum(1 for _ in images(self.root))-len(self.rows);self.labels=lab
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  rgb,d,target=self.rows[i];x,tr=rgb_letterboxed(rgb,self.size);return {'rgb':x,'depth':tensor_depth(d,tr,self.size),'target':torch.tensor(float(target)),'kind':'yht','path':str(rgb)}

class M3FDDM(Dataset):
 """Scene-disjoint zero-target hard negatives; test block is never admitted."""
 def __init__(self,root,split='train',size=640):
  if split not in {'train','val'}:raise ValueError(f'M3FD split {split} not permitted')
  self.root=Path(root).resolve();self.split=split;self.size=size;self.rows=[]
  manifest=self.root/'manifest.csv'
  with manifest.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  if not rows or not {'id','split','scene_block','rgb'}.issubset(rows[0]):raise RuntimeError(f'bad M3FD manifest: {manifest}')
  train_paths={row['rgb'] for row in rows if row['split'].strip().lower()=='train'}
  holdout_paths={row['rgb'] for row in rows if row['split'].strip().lower() in {'val','test'}}
  train_ids={row['id'] for row in rows if row['split'].strip().lower()=='train'}
  holdout_ids={row['id'] for row in rows if row['split'].strip().lower() in {'val','test'}}
  if train_paths & holdout_paths or train_ids & holdout_ids:raise RuntimeError('M3FD train/val/test leakage in manifest')
  for row in rows:
   part=row['split'].strip().lower();block=int(row['scene_block'])
   if part not in {'train','val','test'}:raise RuntimeError(f'unknown M3FD split: {part}')
   if (part=='test' and block!=0) or (part=='val' and block!=1) or (part=='train' and block not in {2,3,4}):
    raise RuntimeError(f'M3FD split/block mismatch: {row}')
   if part!=split:continue
   if split=='train' and (part in {'val','test'} or block in {0,1}):raise RuntimeError(f'M3FD leakage: {row}')
   rel=Path(row['rgb']);rgb=(self.root/rel).resolve();depth=cache_path(self.root/'depth_cache',rel)
   if self.root not in rgb.parents or not rgb.is_file() or not depth.is_file():
    raise FileNotFoundError(f'M3FD RGB/depth pair missing or outside root: {rgb} | {depth}')
   self.rows.append((rgb,depth,row['id']))
  if not self.rows:raise RuntimeError(f'empty M3FD {split}: {manifest}')
  print(f'[M3FDDM] split={split} rows={len(self.rows)} blocks={"2/3/4" if split=="train" else "1"}')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  rgb,depth,_=self.rows[i];x,tr=rgb_letterboxed(rgb,self.size)
  return {'rgb':x,'depth':tensor_depth(depth,tr,self.size),'target':torch.zeros(1,self.size,self.size),'kind':'m3fd','path':str(rgb)}
