"""Select one DDM veto theta using only block-1 M3FD and RGBT val metrics.

Predeclared rule: minimize hard-negative event FP subject to an absolute mAP50
drop <= max_drop against no-veto baseline; tie-break by lower drop then theta.
This is explicit because the earlier v2 theta-selection rule was not archived.
"""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--eval-root',type=Path,required=True);p.add_argument('--val-manifest',type=Path,required=True);p.add_argument('--baseline-map',type=float,required=True);p.add_argument('--max-drop',type=float,default=.001);p.add_argument('--expected-val-count',type=int,default=402);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 raw_manifest=a.val_manifest.read_bytes()
 val_rows=[json.loads(x) for x in raw_manifest.decode('utf-8').splitlines() if x.strip()]
 if len(val_rows)!=a.expected_val_count or any(r.get('split')!='val' or r.get('scene_block')!=1 for r in val_rows):
  raise RuntimeError('Selection manifest is not the expected block-1-only validation set')
 rows=[]
 for path in sorted(a.eval_root.glob('theta_*/metrics.json')):
  m=json.loads(path.read_text(encoding='utf-8'))
  if m['n_hardneg']!=a.expected_val_count:raise RuntimeError(f'{path}: expected block-1 count {a.expected_val_count}, got {m["n_hardneg"]}')
  if m['tau']!=.045:raise RuntimeError(f'{path}: tau != 0.045')
  if m['phy_ddm']!='1':raise RuntimeError(f'{path}: PHY_DDM != 1')
  row={'theta':float(m['phy_ddm_veto_t']),'map50':m['mAP50'],'map_drop':a.baseline_map-m['mAP50'],
       'hardneg_fp':m['hardneg_fp'],'person_fp':m['hardneg_fp_person'],'path':str(path)}
  rows.append(row)
 if not rows:raise RuntimeError(f'No theta_*/metrics.json under {a.eval_root}')
 if len({r['theta'] for r in rows})!=len(rows):raise RuntimeError('Duplicate theta')
 if len({r['person_fp'] for r in rows})!=1:raise RuntimeError('Person FP changed across theta; veto semantics violated')
 eligible=[r for r in rows if r['map_drop']<=a.max_drop]
 if not eligible:raise RuntimeError('No theta meets mAP constraint')
 best=min(eligible,key=lambda r:(r['hardneg_fp'],r['map_drop'],r['theta']))
 out={'selection_split':'RGBT val + M3FD scene block 1 only','test_block_0_used':False,
      'val_manifest':str(a.val_manifest),'val_manifest_sha256':hashlib.sha256(raw_manifest).hexdigest(),
      'rule':'min hardneg FP subject to absolute mAP50 drop <= max_drop; tie lower drop then lower theta',
      'baseline_map':a.baseline_map,'max_drop':a.max_drop,'expected_val_count':a.expected_val_count,
      'selected':best,'grid':rows,
      'limitation':'This explicit rule is not claimed to reproduce an undocumented v2 theta selection.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
 print('[DDM-THETA] '+json.dumps(best))
if __name__=='__main__':main()
