"""Fail-closed bitwise comparison against Phase2 RGB and Phase3 depth inference."""
import argparse
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from tools.ddm_dataset import RGBTDDM,M3FDDM
from tools.ddm_geometry import rgb_letterboxed
from eval_ddm_val import _depth_letterboxed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rgbt-root',type=Path)
    p.add_argument('--rgbt-cache',type=Path)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--phase2-field-use',type=Path,default=Path(__file__).resolve().parents[2]/'FieldGuide_Phase2_FieldUse_20260916')
    a=p.parse_args()
    if not a.phase2_field_use.is_dir():raise FileNotFoundError(a.phase2_field_use)
    sys.path.insert(0,str(a.phase2_field_use))
    from field_use_common import prepare_six
    temporary=None
    if a.synthetic:
        temporary=tempfile.TemporaryDirectory(prefix='ddm_v2_alignment_')
        root=Path(temporary.name);cache=root/'depth_cache'
        for i in range(20):
            width,height=(83+i,57+i) if i%2 else (61+i,91+i)
            rgb=np.empty((height,width,3),dtype=np.uint8)
            yy,xx=np.indices((height,width));rgb[...,0]=(xx+i)%256;rgb[...,1]=(yy*2+i)%256;rgb[...,2]=(xx+yy)%256
            gray=((xx+yy+i)%256).astype(np.uint8);depth=((xx+yy*3+i)*113%65536).astype(np.uint16)
            for sub in ('images/train','image/train','labels/train'):(root/sub).mkdir(parents=True,exist_ok=True)
            (cache/'images/train').mkdir(parents=True,exist_ok=True)
            name=f'{i:05d}'
            Image.fromarray(rgb).save(root/'images/train'/f'{name}.png')
            Image.fromarray(gray).save(root/'image/train'/f'{name}.png')
            Image.fromarray(depth).save(cache/'images/train'/f'{name}.png')
            (root/'labels/train'/f'{name}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
        a.rgbt_root=root;a.rgbt_cache=cache
    elif not a.rgbt_root or not a.rgbt_cache:
        p.error('--rgbt-root and --rgbt-cache are required unless --synthetic is set')
    ds=RGBTDDM(a.rgbt_root,a.rgbt_cache,'train')
    checked=0
    for i,(rgb,depth,_) in enumerate(ds.rows[:20]):
        rel=rgb.relative_to(a.rgbt_root/'images');ir=a.rgbt_root/'image'/rel
        if not ir.is_file():raise FileNotFoundError(f'RGBT IR pair missing: {ir}')
        sample=ds[i];ref,tr=prepare_six(rgb,ir,torch,'cpu')
        direct,ttr=rgb_letterboxed(rgb)
        if tr!=ttr:raise RuntimeError(f'transform mismatch: {rgb} {tr} != {ttr}')
        if not torch.equal(sample['rgb'],ref[0,:3]) or not torch.equal(sample['rgb'],direct):
            raise RuntimeError(f'RGB bitwise mismatch: {rgb}')
        reference_depth=_depth_letterboxed(depth,tr,'cpu')[0]
        if not torch.equal(sample['depth'],reference_depth):
            raise RuntimeError(f'depth bitwise mismatch: {depth}')
        checked+=1
    if checked!=20:raise RuntimeError(f'expected first 20 RGBT train pairs, got {checked}')
    print(f'[DDM-V2-ALIGN] RGB torch.equal=PASS depth torch.equal=PASS pairs={checked}')
    if a.synthetic:
        mroot=a.rgbt_root/'m3fd'
        rows=['id,split,scene_block,rgb']
        for j,(part,block) in enumerate((('train',2),('val',1),('test',0))):
            rel=Path('images')/part/f'{j:05d}.png'
            (mroot/rel).parent.mkdir(parents=True,exist_ok=True)
            (mroot/'depth_cache'/rel).parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.zeros((32,48,3),np.uint8)).save(mroot/rel)
            Image.fromarray(np.zeros((32,48),np.uint16)).save((mroot/'depth_cache'/rel).with_suffix('.png'))
            rows.append(f'{j},{part},{block},{rel.as_posix()}')
        (mroot/'manifest.csv').write_text('\n'.join(rows)+'\n')
        assert len(M3FDDM(mroot,'train'))==1 and len(M3FDDM(mroot,'val'))==1
        (mroot/'manifest.csv').write_text(('\n'.join(rows)+'\n').replace('0,train,2,','0,train,0,'))
        try:M3FDDM(mroot,'train')
        except RuntimeError:pass
        else:raise RuntimeError('M3FD block-0 leakage guard failed')
        print('[DDM-V2-M3FD] train/val selection=PASS block-0 leakage=PASS')
    if temporary:temporary.cleanup()


if __name__=='__main__':main()
