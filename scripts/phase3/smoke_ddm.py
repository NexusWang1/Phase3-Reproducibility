"""DDM unit smoke and optional real frozen-v2.7.2 bit-exact smoke."""
import argparse,os,sys,torch
from pathlib import Path
from ultralytics.nn.modules.ddm_cues import DDM3Cue
from ultralytics.nn.modules.ddm_branch import DDMBranch,DDMScoreFusion
def unit():
 x=torch.rand(2,3,64,64);d=torch.linspace(0,1,64).view(1,1,1,64).expand(2,1,64,64);c=DDM3Cue();l1,l2,l3=c(x,d);b=DDMBranch(32);p=b(l1,l2,l3,torch.rand(2,32,16,16));assert p.shape==(2,1,16,16) and all(torch.isfinite(z).all() for z in (l1,l2,l3,p));n3=sum(z.numel() for z in c.parameters());nb=sum(z.numel() for z in b.parameters());assert n3<=1500000 and nb<=500000;s=torch.rand_like(p);os.environ['PHY_DDM']='0';assert torch.equal(DDMScoreFusion()(s,p,p),s);return n3,nb
def real(weights,device,phase2_package):
 sys.path.insert(0,str(phase2_package))
 from smoke_frp_v2_7 import make
 from ultralytics.nn.modules.ddm_fusion_hook import DDMFusionHook
 frozen,_,_=make(Path(weights),device,True);candidate,_,_=make(Path(weights),device,True);DDMFusionHook().attach(candidate);x=torch.zeros(1,6,640,640,device=device)
 os.environ['PHY_DDM']='0'
 with torch.no_grad():a=frozen(x);b=candidate(x)
 a=a[0] if isinstance(a,tuple) else a;b=b[0] if isinstance(b,tuple) else b;assert torch.equal(a,b),'PHY_DDM=0 is not bit-exact to frozen v2.7.2'
 os.environ['PHY_DDM']='1'
 with torch.no_grad():c=candidate(x)
 c=c[0] if isinstance(c,tuple) else c
 assert torch.equal(a,c),'PHY_DDM=1 purifier/detector changed without box veto'
 aux=candidate._last_frp_aux;assert aux['p_ddm'].shape==candidate._last_t.shape and 'q' not in aux
 return 'real_model_bitcheck=PASS'
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights');p.add_argument('--device',default='cuda:0');p.add_argument('--phase2-package',type=Path,default=Path(__file__).resolve().parent.parent/'FieldGuide_Phase2_FRP_20260916');p.add_argument('--unit-only',action='store_true');a=p.parse_args();n3,nb=unit();extra=real(a.weights,a.device,a.phase2_package) if a.weights and not a.unit_only else 'unit_only=PASS';print(f'[DDM-SMOKE] L3={n3} branch={nb} ddm_off_bitcheck=PASS {extra}')
if __name__=='__main__':main()
