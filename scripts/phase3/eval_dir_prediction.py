"""Self-supervised field direction evaluation for ordered .npy field sequences."""
import argparse,json
from pathlib import Path
import numpy as np
from ultralytics.models.yolo.detect.field3d import estimate_direction_from_fields
def iou(a,b):return float(np.logical_and(a,b).sum()/(np.logical_or(a,b).sum() or 1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--fields',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threshold',type=float,default=.05);a=p.parse_args();files=sorted(a.fields.glob('*.npy'));rows=[]
 for dt in (1,3,5):
  for i in range(len(files)-dt):
   prev=np.load(files[i]);curr=np.load(files[i+dt]);e=estimate_direction_from_fields(prev,curr,threshold=a.threshold);v=np.asarray(e['direction']['vector']);ys,xs=np.where(curr>a.threshold);truth=np.array([xs.mean(),ys.mean()])-np.array([np.where(prev>a.threshold)[1].mean(),np.where(prev>a.threshold)[0].mean()]) if len(xs) else np.zeros(2);n=np.linalg.norm(truth);angle=float(np.degrees(np.arccos(np.clip(np.dot(v,truth)/(np.linalg.norm(v)*n),-1,1)))) if n and np.linalg.norm(v) else None;rows.append({'dt':dt,'previous':files[i].name,'current':files[i+dt].name,'angle_error_deg':angle,'frontier_iou':iou(prev>a.threshold,curr>a.threshold),'direction':e})
 out={'n':len(rows),'by_dt':{str(dt):{'angle_error_deg_mean':float(np.mean([r['angle_error_deg'] for r in rows if r['dt']==dt and r['angle_error_deg'] is not None])),'frontier_iou_mean':float(np.mean([r['frontier_iou'] for r in rows if r['dt']==dt]))} for dt in (1,3,5)},'rows':rows};a.out.mkdir(parents=True,exist_ok=True);(a.out/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['by_dt']))
if __name__=='__main__':main()
