"""Invert paired YHT inferno thermal JPEGs to approximate uint8 relative heat.

This is a lossy LUT inversion only: 255 means hottest relative inferno index,
never an absolute temperature. Every thermal frame is name-paired to RGB first.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import colormaps
from scipy.spatial import cKDTree

def rgb_to_lab(rgb):
    x=np.asarray(rgb,np.float32)/255.;x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    xyz=np.stack([.4124564*x[...,0]+.3575761*x[...,1]+.1804375*x[...,2],.2126729*x[...,0]+.7151522*x[...,1]+.072175*x[...,2],.0193339*x[...,0]+.119192*x[...,1]+.9503041*x[...,2]],-1)/np.array([.95047,1.,1.08883],np.float32)
    f=np.where(xyz>0.008856,xyz**(1/3),7.787*xyz+16/116);return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
def valid(path):
    try:
        with Image.open(path) as im:return im.mode=='L' and im.size[0]>0
    except Exception:return False
def rgb_index(root):
    out={}
    for p in root.rglob('*'):
        if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:out.setdefault(p.stem,[]).append(p)
    return out
def paired_rgb(src,thermal_root,rgb_root,index):
    mapped=rgb_root/src.name.replace('Thermal','RGB')
    if mapped.is_file():return mapped
    rel=src.relative_to(thermal_root);same=(rgb_root/rel)
    if same.is_file():return same
    options=index.get(src.stem,[])
    return options[0] if len(options)==1 else None
def visual(raw,gray,path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(1,3,figsize=(14,4));ax[0].imshow(raw);ax[0].set_title('inferno thermal source');ax[1].imshow(gray,cmap='gray',vmin=0,vmax=255);ax[1].set_title('inferno inverted: 255=hottest (approximate)');lum=np.asarray(raw).mean(2);ax[2].hist(lum.ravel(),bins=64,alpha=.65,label='source brightness');ax[2].hist(gray.ravel(),bins=64,alpha=.65,label='inverted index');ax[2].legend();[a.axis('off') for a in ax[:2]];fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True);p.add_argument('--rgb-root',type=Path,required=True);p.add_argument('--dst',type=Path,required=True);p.add_argument('--ext',default='.jpg');p.add_argument('--visual-out',type=Path);p.add_argument('--visual-n',type=int,default=10);p.add_argument('--max-lab-distance',type=float,default=12.);a=p.parse_args()
    thermal=sorted(x for x in a.src.rglob('*') if x.suffix.lower()==a.ext.lower());idx=rgb_index(a.rgb_root);lut=(colormaps['inferno'](np.linspace(0,1,256))[:,:3]*255).astype(np.uint8);tree=cKDTree(rgb_to_lab(lut).reshape(256,3));created=skipped=0;fail=[];unpaired=[];visuals=[];t=time.perf_counter()
    for src in thermal:
        rgb_pair=paired_rgb(src,a.src,a.rgb_root,idx)
        if rgb_pair is None:unpaired.append(str(src.relative_to(a.src)));continue
        dst=(a.dst/src.relative_to(a.src)).with_suffix('.png')
        if valid(dst):skipped+=1;continue
        try:
            raw=np.asarray(Image.open(src).convert('RGB'));dist,ind=tree.query(rgb_to_lab(raw).reshape(-1,3),workers=-1);gray=ind.reshape(raw.shape[:2]).astype(np.uint8)
            if float(dist.mean())>a.max_lab_distance:fail.append({'file':str(src.relative_to(a.src)),'mean_lab_distance':float(dist.mean())});continue
            dst.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(gray,'L').save(dst);created+=1
            if a.visual_out and len(visuals)<a.visual_n:visual(raw,gray,a.visual_out/f'{len(visuals):03d}_{src.stem}.png');visuals.append(str(src.relative_to(a.src)))
        except Exception as e:fail.append({'file':str(src.relative_to(a.src)),'error':str(e)})
    meta={'source_total':len(thermal),'paired_rgb':len(thermal)-len(unpaired),'unpaired_rgb':unpaired,'created':created,'skipped':skipped,'failed':len(fail),'failures':fail,'seconds':time.perf_counter()-t,'format':'uint8 PNG; 255=hottest inferno LUT index; lossy relative brightness only','visuals':visuals};a.dst.mkdir(parents=True,exist_ok=True);(a.dst/'inferno_invert_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False));print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
