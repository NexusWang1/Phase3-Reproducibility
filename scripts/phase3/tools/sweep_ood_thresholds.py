"""Deterministic offline veto sweep; only stdlib and NumPy, no torch or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

trapz = getattr(np, 'trapezoid', None) or np.trapz  # numpy>=2 removed trapz; identical numerics


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def ap(pred,gt,threshold=.5):
    """Equivalent to eval_ddm_val._ap, with image/class indexing for sweeps."""
    if not gt:return 0.0
    by_image=defaultdict(list)
    for j,row in enumerate(gt):by_image[row['image']].append((j,row))
    used=set();hits=[]
    for row in sorted(pred,key=lambda z:-z['conf']):
        found=False
        for j,truth in by_image[row['image']]:
            if j not in used and iou(row['box_xyxy'],truth['box_xyxy'])>=threshold:
                used.add(j);found=True;break
        hits.append(int(found))
    if not hits:return 0.0
    tp=np.cumsum(hits);fp=np.arange(1,len(hits)+1)-tp
    recall=tp/len(gt);precision=tp/np.maximum(tp+fp,1)
    return float(trapz(np.r_[1.,np.maximum.accumulate(precision[::-1])[::-1]],np.r_[0.,recall]))


def map50(pred,gt):
    classes=sorted({row['cls'] for row in gt})
    return float(np.mean([ap([r for r in pred if r['cls']==c],
                             [r for r in gt if r['cls']==c]) for c in classes])) if classes else 0.


def sweep(scores,gt,method,thresholds,baseline_map):
    out=[]
    for threshold in thresholds:
        kept=[];fp=0;person=0;null=0
        for row in scores:
            cls=row['cls'];score=row.get(method)
            veto=False
            if cls in (0,1):
                if score is None:null+=1
                elif method=='conf':veto=score<float(threshold)
                else:veto=score>float(threshold)
            if veto:continue
            if row['dataset']=='rgbt_val':kept.append(row)
            elif cls in (0,1):fp+=1
            elif cls==2:person+=1
        value=map50(kept,gt)
        out.append({'threshold':float(threshold),'map50':value,'delta_map':baseline_map-value,
                    'hardneg_fp':fp,'hardneg_person_fp':person,'null_event_scores_kept':null})
    return out


def operating_points(points):
    lossless=[p for p in points if p['delta_map']<=2e-5+1e-12]
    equivalent=[p for p in points if p['hardneg_fp']<=51]
    return {'map_lossless':min(lossless,key=lambda p:(p['hardneg_fp'],-p['map50'])) if lossless else None,
            'fp51_equivalent':min(equivalent,key=lambda p:(p['delta_map'],-p['hardneg_fp'])) if equivalent else None}


def fmt(method,label,point,base_fp):
    if point is None:return f'| {method} | {label} | unavailable | unavailable | unavailable | unavailable | unavailable |\n'
    fp=point['hardneg_fp'];marker='<span style="color:red">'+str(fp)+'</span>' if label=='mAP-lossless' and fp<=51 else str(fp)
    return f"| {method} | {label} | {point['threshold']:.8g} | {point['map50']:.8f} | {point['delta_map']:.8g} | {marker} | {(base_fp-fp)/base_fp:.1%} |\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--in-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path)
    a=p.parse_args();out=a.out_dir or a.in_dir
    if not a.in_dir.is_dir():raise FileNotFoundError(a.in_dir)
    for name in ('sweep_results.json','comparison_table.md'):
        if (out/name).exists():raise FileExistsError(f'Refusing overwrite: {out/name}')
    score_path=a.in_dir/'ood_scores.jsonl';gt_path=a.in_dir/'gt_records.jsonl'
    scores=read_jsonl(score_path);gt=read_jsonl(gt_path)
    if not scores or not gt:raise RuntimeError('Empty scores or GT')
    baseline_pred=[r for r in scores if r['dataset']=='rgbt_val']
    base_map=map50(baseline_pred,gt)
    base_fp=sum(r['dataset']=='m3fd_hardneg' and r['cls'] in (0,1) for r in scores)
    base_person=sum(r['dataset']=='m3fd_hardneg' and r['cls']==2 for r in scores)
    grids={'conf':np.round(np.arange(.045,.500001,.005),6).tolist()}
    for method in ('energy','maha'):
        values=np.array([r[method] for r in baseline_pred if r['cls'] in (0,1) and r.get(method) is not None],float)
        if not len(values):raise RuntimeError(f'No RGBT-val {method} scores for threshold grid')
        margin=max(1e-9,float(values.max()-values.min())*1e-6)
        grids[method]=np.unique(np.r_[values.min()-margin,
                                        np.quantile(values,np.linspace(0,1,48)),
                                        values.max()+margin]).tolist()
    results={};points={}
    for method,grid in grids.items():
        results[method]=sweep(scores,gt,method,grid,base_map)
        points[method]=operating_points(results[method])
    first=results['conf'][0]
    if first['hardneg_fp']!=base_fp or abs(first['map50']-base_map)>1e-12:
        raise RuntimeError('conf θ=tau does not reproduce no-veto baseline')
    if abs(base_fp-155)>1 or abs(base_map-.9968)>.001:
        raise RuntimeError(f'Baseline protocol mismatch: FP={base_fp}, mAP50={base_map}')
    payload={'protocol':{'person_exempt':True,'event_classes':[0,1],'tau':.045,
                         'energy_and_maha_grid':'50 deterministic points from RGBT val event scores, including finite outer endpoints',
                         'ap':'same greedy-by-confidence, first-unmatched-IoU>=0.5 and trapezoid as eval_ddm_val._ap'},
             'baseline':{'map50':base_map,'hardneg_fp':base_fp,'hardneg_person_fp':base_person},
             'ddm_reference':{'threshold':.005,'map50':.996773,'hardneg_fp':51,'delta_map':base_map-.996773},
             'input_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (score_path,gt_path)},
             'grids':grids,'sweeps':results,'operating_points':points}
    out.mkdir(parents=True,exist_ok=True)
    (out/'sweep_results.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    table='| Method | Operating point | Threshold | mAP@0.5 | ΔmAP | Hard-neg FP | FP reduction |\n|---|---|---:|---:|---:|---:|---:|\n'
    table+=f'| No veto | baseline | — | {base_map:.8f} | 0 | {base_fp} | — |\n'
    for method in ('conf','energy','maha'):
        table+=fmt(method,'mAP-lossless',points[method]['map_lossless'],base_fp)
        table+=fmt(method,'FP=51 equivalent',points[method]['fp51_equivalent'],base_fp)
    table+=f'| DDM v2 (reference) | θ=0.005 | 0.005 | 0.99677300 | {base_map-.996773:.8g} | 51 | {(base_fp-51)/base_fp:.1%} |\n'
    table+='\nRed FP values identify a classical baseline reaching ≤51 FP at ΔmAP≤2e-5. An unavailable point means the scanned grid did not reach it.\n'
    contenders=[(method,point['hardneg_fp']) for method,pair in points.items()
                if (point:=pair['map_lossless']) is not None]
    if contenders:
        winner,lowest=min(contenders,key=lambda item:item[1])
        verdict='weakens' if lowest<=51 else 'supports'
        table+=f'\nAt the lossless mAP operating point, {winner} is closest to DDM v2: {lowest} FP versus 51 (gap {lowest-51:+d}). This {verdict} the claimed DDM advantage under this protocol.\n'
    else:table+='\nNo classical baseline had a feasible lossless-mAP point on the registered grid.\n'
    (out/'comparison_table.md').write_text(table,encoding='utf-8')
    print('[OOD-SWEEP] '+json.dumps({'baseline_map50':base_map,'baseline_fp':base_fp,'operating_points':points}),flush=True)

if __name__=='__main__':main()
