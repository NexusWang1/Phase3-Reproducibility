"""New-only wrapper around the frozen v2 trainer for registered DDM ablations.

All other training arguments are forwarded to train_ddm.py unchanged. M3FD-off
keeps the same image/optimizer-step schedule but sets its BCE weight to zero.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

from tools.ddm_ablation_lib import VARIANTS,StretchTrainDataset,patch_model_classes

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant',choices=VARIANTS,default='full')
    parser.add_argument('--geometry',choices=('aligned','v1_stretch'),default='aligned')
    parser.add_argument('--m3fd-zero',choices=('on','off'),default='on')
    settings,remaining=parser.parse_known_args()
    if '--out' not in remaining:raise ValueError('The forwarded --out directory is required')
    if '--w-m3fd-zero' in remaining:raise ValueError('Use --m3fd-zero on/off; do not override its BCE weight')
    out=Path(remaining[remaining.index('--out')+1])
    if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
    if '--seed' in remaining and remaining[remaining.index('--seed')+1]!='20260922':
        raise ValueError('Ablation seed must match v2: 20260922')
    if '--seed' not in remaining:remaining+=['--seed','20260922']
    remaining+=['--w-m3fd-zero','1' if settings.m3fd_zero=='on' else '0']
    cue,branch=patch_model_classes(settings.variant)
    import train_ddm as trainer
    trainer.DDM3Cue=cue
    trainer.DDMBranch=branch
    if settings.geometry=='v1_stretch':
        rgbt,yht,m3fd=trainer.RGBTDDM,trainer.YHTDDM,trainer.M3FDDM
        trainer.RGBTDDM=lambda *a,**k: StretchTrainDataset(rgbt(*a,**k)) if (len(a)>2 and a[2]=='train') or k.get('split')=='train' else rgbt(*a,**k)
        trainer.YHTDDM=lambda *a,**k: StretchTrainDataset(yht(*a,**k))
        trainer.M3FDDM=lambda *a,**k: StretchTrainDataset(m3fd(*a,**k)) if (len(a)>1 and a[1]=='train') or k.get('split')=='train' else m3fd(*a,**k)
    config={'variant':settings.variant,'geometry_train':settings.geometry,
            'geometry_val':'aligned_v2','m3fd_zero_target':settings.m3fd_zero=='on',
            'm3fd_off_semantics':'same M3FD forward/optimizer schedule; M3FD BCE weight=0',
            'seed':20260922,'cue_version':'L2 within-stratum z-score v3'}
    print('[DDM-ABLATION] '+json.dumps(config),flush=True)
    old_argv=sys.argv
    try:
        sys.argv=['train_ddm.py']+remaining
        trainer.main()
    finally:sys.argv=old_argv
    for name in ('best.pt','last.pt'):
        path=out/name
        state=torch.load(path,map_location='cpu',weights_only=False)
        state['ablation']=config
        torch.save(state,path)
    summary=out/'summary.json'
    doc=json.loads(summary.read_text(encoding='utf-8'))
    doc['ablation']=config
    summary.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    print(f'[DDM-ABLATION] complete {out}',flush=True)

if __name__=='__main__':main()
