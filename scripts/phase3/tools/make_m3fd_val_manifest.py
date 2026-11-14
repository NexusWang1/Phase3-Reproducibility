"""Materialize the scene-block-1-only M3FD manifest for theta selection."""
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--m3fd-root',type=Path,required=True);p.add_argument('--test-hardneg',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 root=a.m3fd_root.resolve();rows=list(csv.DictReader((root/'manifest.csv').open(encoding='utf-8-sig',newline='')))
 val=[];test_ids=set();test_rgb=set()
 for r in rows:
  split=r['split'].strip().lower();block=int(r['scene_block'])
  if split=='test':
   if block!=0:raise RuntimeError(f'Test scene outside block 0: {r}')
   test_ids.add(r['id']);test_rgb.add(r['rgb'].replace('\\','/'));continue
  if split!='val':continue
  if block!=1:raise RuntimeError(f'Val scene outside block 1: {r}')
  rgb=Path(r['rgb']);ir=Path(r.get('tir') or str(rgb).replace('images/','image/',1))
  if not (root/rgb).is_file() or not (root/ir).is_file():raise FileNotFoundError((root/rgb,root/ir))
  val.append({'id':r['id'],'rgb':str(rgb).replace('\\','/'),'ir_gray':str(ir).replace('\\','/'),
              'scene_block':block,'split':'val'})
 if not val or {r['id'] for r in val}&test_ids:raise RuntimeError('Empty/leaking val manifest')
 if a.test_hardneg:
  hard=[json.loads(x) for x in a.test_hardneg.read_text(encoding='utf-8').splitlines() if x.strip()]
  invalid=[r['rgb'] for r in hard if r['rgb'].replace('\\','/') not in test_rgb]
  if invalid:raise RuntimeError(f'Hard-negative evaluation manifest includes non-test-block-0 RGB: {invalid[:3]}')
  if len({r['rgb'] for r in hard})!=len(hard):raise RuntimeError('Duplicate hard-negative RGB')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(''.join(json.dumps(r)+'\n' for r in val),encoding='utf-8')
 print(f'[M3FD-VAL] val_block1={len(val)} test_block0_excluded={len(test_ids)} test_hardneg_checked={bool(a.test_hardneg)} out={a.out}')
if __name__=='__main__':main()
