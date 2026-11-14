"""Run the unchanged v2 reference evaluator with checkpoint-matched cue masks."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tools.ddm_ablation_lib import patch_model_classes

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ddm-ckpt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    settings,remaining=parser.parse_known_args()
    if settings.out.exists():raise FileExistsError(f'Refusing overwrite: {settings.out}')
    if os.getenv('PHY_DDM')!='1':raise RuntimeError('Set PHY_DDM=1 for ablation evaluation')
    ckpt=torch.load(settings.ddm_ckpt,map_location='cpu',weights_only=False)
    config=ckpt.get('ablation')
    if not config:raise RuntimeError(f'Checkpoint has no ablation metadata: {settings.ddm_ckpt}')
    variant=config['variant']
    torch.manual_seed(int(config['seed']))
    if torch.cuda.is_available():torch.cuda.manual_seed_all(int(config['seed']))
    patch_model_classes(variant)
    import eval_ddm_val
    old_argv=sys.argv
    try:
        sys.argv=['eval_ddm_val.py','--ddm-ckpt',str(settings.ddm_ckpt),'--out',str(settings.out)]+remaining
        eval_ddm_val.main()
    finally:sys.argv=old_argv
    metrics=settings.out/'metrics.json'
    data=json.loads(metrics.read_text(encoding='utf-8'))
    data['ablation']=config
    metrics.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(f'[DDM-ABLATION-EVAL] variant={variant} {metrics}',flush=True)

if __name__=='__main__':main()
