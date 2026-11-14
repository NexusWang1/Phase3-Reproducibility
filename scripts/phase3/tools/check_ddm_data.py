"""Preflight cached-depth alignment and YHT frame-label audit."""
import argparse,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from tools.ddm_dataset import images,cache_path,parse_yht_labels
def audit(root,cache,n=200):
 rows=images(root);sample=random.Random(20260922).sample(rows,min(n,len(rows)));missing=[];values=[]
 for rgb in sample:
  d=cache_path(cache,rgb.relative_to(root))
  if not d.is_file():missing.append(str(rgb));continue
  x=np.asarray(Image.open(d),np.uint16);values.extend([float(x.mean()),*np.percentile(x,[1,50,99]).tolist()])
 return {'total':len(rows),'sampled':len(sample),'aligned':len(sample)-len(missing),'alignment_rate':(len(sample)-len(missing))/len(sample) if sample else 0,'missing':missing[:20],'depth_stats':{'min':min(values) if values else None,'max':max(values) if values else None,'mean':float(np.mean(values)) if values else None}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--rgbt-root',type=Path,required=True);p.add_argument('--rgbt-cache',type=Path,required=True);p.add_argument('--yht-rgb',type=Path,required=True);p.add_argument('--yht-cache',type=Path,required=True);p.add_argument('--yht-labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=200);a=p.parse_args()
 report={'rgbt_train':audit(a.rgbt_root/'images'/'train',a.rgbt_cache/'images'/'train',a.n),'rgbt_val':audit(a.rgbt_root/'images'/'val',a.rgbt_cache/'images'/'val',a.n),'yht':audit(a.yht_rgb,a.yht_cache,a.n)};labels=parse_yht_labels(a.yht_labels);report['yht_labels']={'frames':len(labels),'positive':sum(labels.values()),'negative':len(labels)-sum(labels.values())};report['passed']=all(x['alignment_rate']==1 for x in (report['rgbt_train'],report['rgbt_val'],report['yht'])) and len(labels)==53451;a.out.mkdir(parents=True,exist_ok=True);(a.out/'check_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));
 if not report['passed']:raise RuntimeError('DDM data preflight failed')
if __name__=='__main__':main()
