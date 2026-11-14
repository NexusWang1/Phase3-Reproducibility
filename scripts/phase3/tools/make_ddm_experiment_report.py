"""Evidence-only v3/v2 comparison and ablation table; never invent missing runs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

ARMS={
 'geom_aligned_hn_on':('aligned','on','full'),
 'geom_aligned_hn_off':('aligned','off','full'),
 'geom_stretch_hn_on':('v1_stretch','on','full'),
 'geom_stretch_hn_off':('v1_stretch','off','full'),
 'no_l1':('aligned','on','no_l1'),
 'no_l2':('aligned','on','no_l2'),
 'no_l3':('aligned','on','no_l3'),
 'shuffle_depth':('aligned','on','shuffle_depth'),
 'rgb_only':('aligned','on','rgb_only'),
 'depth_only':('aligned','on','depth_only')}

def read(path):
 if not path.is_file():return None
 return json.loads(path.read_text(encoding='utf-8'))
def best_auc(run):
 path=run/'results.csv'
 if not path.is_file():return None
 with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if not rows:return None
 best=max(rows,key=lambda x:float(x['val_ap']))
 return float(best['val_auc'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--v3-run',type=Path,required=True);p.add_argument('--v3-eval',type=Path,required=True);p.add_argument('--l2-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 v3=read(a.v3_run/'summary.json');v3eval=read(a.v3_eval/'metrics.json');audit=read(a.l2_audit)
 if not v3 or not v3eval or not audit:raise RuntimeError('v3 training/eval/L2 audit not all present')
 baseline_map=0.9967877821325909
 lines=['# DDM v3 L2 repair and orthogonal ablation','',
        'This report is generated from result files. Missing server experiments are marked pending, not estimated.','',
        '| Metric | v2 reference | v3 fixed θ=0.005 |','|---|---:|---:|',
        f"| RGBT val AP | 0.8305 | {v3['best_val_ap']:.6f} |",
        f"| RGBT val AUC | 0.960 | {best_auc(a.v3_run):.6f} |",
        f"| mAP@0.5 | 0.996773 | {v3eval['mAP50']:.6f} |",
        f"| M3FD hard-negative event FP | 51 | {v3eval['hardneg_fp']} |",
        f"| M3FD person FP | not specified here | {v3eval['hardneg_fp_person']} |",
        f"| M3FD val mean p_DDM | not specified here | {v3['best_m3fd_val_p_mean']} |",
        '',f"L2 audit: {a.l2_audit}; new L2/e Pearson median {audit['new_l2_e_pearson_median']:.6f}; L1 equal={audit['l1_equal_all']}, L3 equal={audit['l3_equal_all']}; legacy e/k exact={audit['legacy_exact_e_over_k']} (boundary-overlap frames={audit['legacy_boundary_overlap_frames']})."]
 onepage=lines+['',f'v3 training: {a.v3_run}',f'v3 fixed-θ reference evaluation: {a.v3_eval}',
   'The fixed θ=0.005 result is compared without changing tau=0.045. Re-selected θ is documented separately in the val-only selection JSON. If any v3 metric worsens, report it as observed.']
 lines += ['', '## Ablation (same repaired-v3 cues, except stated variant)', '',
        '| Arm | Train geometry | M3FD zero target | Cue/input variant | Val AP | Val AUC | Selected θ | Hard-negative FP | ΔmAP vs no-veto | Fixed-θ FP |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
 rows=[]
 for name,(geometry,hardneg,variant) in ARMS.items():
  run=a.runs_root/f'ddm_ablation_{name}';summary=read(run/'summary.json')
  selection=read(run/'theta_scan/selected.json');fixed=read(run/'eval_fixed/metrics.json')
  theta=selection['selected']['theta'] if selection else None
  metrics=read(run/'eval_selected/metrics.json') if theta!=.005 else fixed
  if summary:
   config=summary.get('ablation',{})
   if config.get('variant')!=variant or config.get('geometry_train')!=geometry or config.get('m3fd_zero_target')!=(hardneg=='on'):
    raise RuntimeError(f'Ablation config mismatch: {run}')
  ap=summary.get('best_val_ap') if summary else None;auc=best_auc(run) if summary else None
  fp=metrics.get('hardneg_fp') if metrics else None
  delta=baseline_map-metrics['mAP50'] if metrics else None
  row={'arm':name,'geometry':geometry,'m3fd_zero':hardneg,'variant':variant,
       'val_ap':ap,'val_auc':auc,'selected_theta':theta,'hardneg_fp':fp,'delta_map':delta,
       'fixed_theta_fp':fixed['hardneg_fp'] if fixed else None,
       'train_path':str(run),'eval_path':str(run/'eval_selected/metrics.json') if theta!=.005 else str(run/'eval_fixed/metrics.json')}
  rows.append(row)
  fmt=lambda x,s:format(x,s) if x is not None else 'pending'
  lines.append(f"| {name} | {geometry} | {hardneg} | {variant} | {fmt(ap,'.6f')} | {fmt(auc,'.6f')} | {fmt(theta,'.6f')} | {fmt(fp,'d')} | {fmt(delta,'.8f')} | {fmt(fixed['hardneg_fp'] if fixed else None,'d')} |")
 lines += ['', 'θ=0.005 is the fixed historical comparison. Any val-reselected θ must be reported separately with the block-1-only selection JSON; block-0 test was not used for selection. A v2 selection rule was not archived, so the new explicit rule is not mislabeled as a reproduction.',
           '', 'Metrics below v2 are retained without post-hoc protocol edits.']
 a.out.mkdir(parents=True)
 (a.out/'v3_vs_v2_onepage.md').write_text('\n'.join(onepage)+'\n',encoding='utf-8')
 (a.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 with (a.out/'ablation_table.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(f'[DDM-REPORT] {a.out}')
if __name__=='__main__':main()
