"""Four paired direction baselines and hierarchical block bootstrap for IGNITE.

Endpoint is the original v4 raw, static, 2-D signed-median projection of new
fire pixels k+delta; prediction uses the same v4 thermal-field deflection.
"""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from tools.ignite_direct_v4 import load_tiff,load_mask,field_from_tiff,deflection_vec_2d,projection_stats

def binary_dilation(mask,iterations=3):
 """SciPy default cross-connectivity dilation, without an extra dependency."""
 out=mask.copy()
 for _ in range(iterations):
  expanded=out.copy()
  expanded[1:] |= out[:-1];expanded[:-1] |= out[1:]
  expanded[:,1:] |= out[:,:-1];expanded[:,:-1] |= out[:,1:]
  out=expanded
 return out

def sign_pvalue(diffs):
 x=np.asarray(diffs);x=x[np.abs(x)>1e-12];n=len(x)
 if not n:return None,0
 k=min(int((x>0).sum()),int((x<0).sum()))
 return min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n),n

def unit(v):
 if v is None:return None
 n=np.linalg.norm(v)
 return None if not np.isfinite(n) or n<=1e-12 else v/n

def positive(points,direction):
 v=unit(direction)
 if v is None or len(points)<10:return None
 return int(np.median(points@v)>0)

def hier_boot(rows,key,reps,seed):
 rng=np.random.default_rng(seed);seqs=defaultdict(lambda:defaultdict(list))
 for r in rows:
  if r.get(key) is not None:seqs[r['sequence']][r['block']].append(float(r[key]))
 names=sorted(seqs)
 if len(names)<2:raise RuntimeError('Need at least two sequences for sequence bootstrap')
 values=[]
 for _ in range(reps):
  sample=[]
  for sid in rng.choice(names,size=len(names),replace=True):
   blocks=seqs[sid];keys=list(blocks)
   for bid in rng.choice(keys,size=len(keys),replace=True):sample.extend(blocks[bid])
  values.append(np.mean(sample))
 return [float(x) for x in np.percentile(values,[2.5,97.5])]

def main():
 p=argparse.ArgumentParser();p.add_argument('--ignite-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--delta',type=int,default=10);p.add_argument('--t80',type=float,default=8800);p.add_argument('--min-area',type=int,default=50);p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--seed',type=int,default=20260930);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 records=[json.loads(x) for x in (a.ignite_root/'metadata.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
 groups=defaultdict(list)
 for r in records:groups[r['sequence_id']].append(r)
 if len(groups)!=4:raise RuntimeError(f'Expected 4 IGNITE sequences, found {len(groups)}')
 rng=np.random.default_rng(a.seed);rows=[]
 for sid,items in sorted(groups.items()):
  items.sort(key=lambda r:r['video_time_s'])
  @lru_cache(maxsize=2*a.delta+4)
  def thermal(index):return load_tiff(a.ignite_root/items[index]['thermal_tiff_file_name'])
  @lru_cache(maxsize=2*a.delta+4)
  def masks(index):return load_mask(a.ignite_root/items[index]['mask_80c_file_name'])
  for k in range(a.delta,len(items)-a.delta):
   current=masks(k)
   if current.sum()<a.min_area:continue
   newly=masks(k+a.delta)&~binary_dilation(current,iterations=3)
   if newly.sum()<a.min_area:continue
   ys,xs=np.where(current);cy,cx=ys.mean(),xs.mean()
   ny,nx=np.where(newly);points=np.c_[nx-cx,ny-cy]
   field=field_from_tiff(thermal(k),a.t80)
   predict=deflection_vec_2d(field)
   if unit(predict) is None:continue
   # Centroid -> hottest in the current segmented fire region.
   hot_index=np.argmax(np.where(current,thermal(k),-np.inf))
   hy,hx=np.unravel_index(hot_index,current.shape)
   hot=np.array([hx-cx,hy-cy],dtype=float)
   # Weighted image-plane ascent of the radiometric temperature field.
   gy,gx=np.gradient(thermal(k));weights=np.clip(thermal(k)-a.t80,0,None)*current
   grad=np.array([(gx*weights).sum(),(gy*weights).sum()],dtype=float)
   # Previous observed spread (k-delta -> k), no future information.
   prior=masks(k-a.delta)
   py,px=np.where(prior)
   prev_new=current&~binary_dilation(prior,iterations=3)
   vy,vx=np.where(prev_new)
   prev=np.array([vx.mean()-px.mean(),vy.mean()-py.mean()]) if len(px) and len(vx)>=a.min_area else None
   theta=rng.uniform(0,2*np.pi,size=100)
   random_rate=float(np.mean([positive(points,np.array([np.cos(t),np.sin(t)])) for t in theta]))
   result={'sequence':sid,'frame_index':k,'time_s':float(items[k]['video_time_s']),
           'block':int(float(items[k]['video_time_s'])//60),'new_fire_pixels':int(newly.sum()),
           'pred_positive':positive(points,predict),'random_rate':random_rate,
           'hottest_positive':positive(points,hot),'gradient_positive':positive(points,grad),
           'previous_positive':positive(points,prev)}
   rows.append(result)
  print(f'[IGNITE-DIR] {sid}: {sum(r["sequence"]==sid for r in rows)} valid pairs',flush=True)
 if not rows:raise RuntimeError('No valid frame pairs')
 keys=('pred_positive','random_rate','hottest_positive','gradient_positive','previous_positive')
 summary={}
 for key in keys:
  subset=[r for r in rows if r[key] is not None]
  if not subset:raise RuntimeError(f'No valid pairs for baseline {key}')
  summary[key]={'n':len(subset),'rate':float(np.mean([r[key] for r in subset])),
                'hierarchical_sequence_60s_block_ci95':hier_boot(subset,key,a.bootstrap,a.seed+len(summary))}
  if key!='pred_positive':
   diffs=[r['pred_positive']-r[key] for r in subset]
   pval,n=sign_pvalue(diffs)
   summary[key]['paired_difference_vs_prediction']=float(np.mean(diffs))
   summary[key]['paired_sign_p_two_sided']=pval
   summary[key]['paired_sign_non_ties']=n
 stride_rows=[r for r in rows if r['frame_index']%a.delta==0]
 if not stride_rows:raise RuntimeError('No non-overlapping frame pairs')
 sensitivity={'stride':a.delta,'pooled_n':len(stride_rows),
              'pooled_pred_positive_rate':float(np.mean([r['pred_positive'] for r in stride_rows])),
              'by_sequence':{sid:{'n':sum(r['sequence']==sid for r in stride_rows),
                                   'rate':float(np.mean([r['pred_positive'] for r in stride_rows if r['sequence']==sid])) if any(r['sequence']==sid for r in stride_rows) else None}
                             for sid in sorted(groups)}}
 report={'ignite_root':str(a.ignite_root.resolve()),
         'protocol':'IGNITE raw static 2D signed-median projection, new fire k+delta; no test reuse for training',
         'delta':a.delta,'random_directions_per_pair':100,'bootstrap_reps':a.bootstrap,
         'seed':a.seed,'n_pairs':len(rows),'baselines':summary,'nonoverlap':sensitivity}
 a.out.mkdir(parents=True)
 (a.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 with (a.out/'per_pair.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('[IGNITE-DIR] '+json.dumps({'n_pairs':len(rows),'nonoverlap_n':len(stride_rows)}),flush=True)

if __name__=='__main__':main()
