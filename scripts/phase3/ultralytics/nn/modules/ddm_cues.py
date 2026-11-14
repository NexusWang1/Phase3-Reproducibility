"""Pure-torch, batched DDM cues L1/L2/L3."""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def _sobel(x):
    k=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3); gx=F.conv2d(x,k,padding=1);gy=F.conv2d(x,k.transpose(-1,-2),padding=1);return (gx.square()+gy.square()+1e-8).sqrt()
def _gray(rgb): return .2989*rgb[:,0:1]+.587*rgb[:,1:2]+.114*rgb[:,2:3]

class DDM3Cue(nn.Module):
    def __init__(self,w=11,k=8,base=24):
        super().__init__();self.w=w;self.k=k
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,2,1),nn.GroupNorm(8,b),nn.SiLU())
        self.d1=block(3,base);self.d2=block(base,base*2);self.d3=block(base*2,base*4)
        self.u2=nn.Sequential(nn.ConvTranspose2d(base*4,base*2,2,2),nn.GroupNorm(8,base*2),nn.SiLU())
        self.u1=nn.Sequential(nn.ConvTranspose2d(base*2,base,2,2),nn.GroupNorm(8,base),nn.SiLU())
        self.out=nn.Sequential(nn.ConvTranspose2d(base,base,2,2),nn.SiLU(),nn.Conv2d(base,1,1),nn.Sigmoid())
    def forward(self,rgb,depth):
        if depth.ndim==3:depth=depth[:,None]
        g=_gray(rgb);gi=_sobel(g);pad=self.w//2
        mean=lambda z:F.avg_pool2d(z,self.w,1,pad)
        if os.getenv('PHY_DDM_CUE_MODE','depth')=='rgb_only':
            # Frozen-weight inference ablation: RGB texture autocorrelation,
            # no depth-dependent cue or branch input. L2 is explicitly off.
            neighbor=(torch.roll(gi,1,-1)+torch.roll(gi,1,-2))/2
            neighbor=neighbor.clone();neighbor[...,0,:]=gi[...,0,:];neighbor[...,:,0]=gi[...,:,0]
            rho=(mean(gi*neighbor)-mean(gi)*mean(neighbor))/(mean(gi.square()).sub(mean(gi).square()).clamp_min(1e-6).sqrt()*mean(neighbor.square()).sub(mean(neighbor).square()).clamp_min(1e-6).sqrt())
            l1=(mean(gi)*(1-rho.clamp(-1,1))/2).clamp(0,1)
            l2=torch.zeros_like(l1)
            z=torch.cat([g,torch.full_like(g,.5),l1],1)
            l3=self.out(self.u1(self.u2(self.d3(self.d2(self.d1(z))))))
            return l1,l2,l3
        if os.getenv('PHY_DDM_CUE_MODE','depth')!='depth':
            raise ValueError('PHY_DDM_CUE_MODE must be depth or rgb_only')
        gd=_sobel(depth)
        cov=mean(gi*gd)-mean(gi)*mean(gd);rho=cov/(mean(gi.square()).sub(mean(gi).square()).clamp_min(1e-6).sqrt()*mean(gd.square()).sub(mean(gd).square()).clamp_min(1e-6).sqrt())
        l1=mean(gi)*(1-rho.clamp(-1,1))/2
        # differentiable edge co-occurrence, depth strata gate each pixel.
        edge_i=torch.sigmoid((gi-gi.mean((-2,-1),keepdim=True))/gi.std((-2,-1),keepdim=True).clamp_min(1e-6))
        edge_d=torch.sigmoid((gd-gd.mean((-2,-1),keepdim=True))/gd.std((-2,-1),keepdim=True).clamp_min(1e-6))
        q=torch.quantile(depth.flatten(2),torch.linspace(0,1,self.k+1,device=depth.device),dim=2).squeeze(-1).permute(1,0).unsqueeze(-1).unsqueeze(-1)
        e=edge_i*edge_d
        j_idx=torch.searchsorted(q[:,1:-1,0,0].contiguous(),depth.flatten(2).squeeze(1).contiguous(),right=False)
        l2_raw=torch.zeros_like(e)
        for j in range(self.k):
            # Half-open partition: each pixel belongs to exactly one stratum.
            mask=(j_idx==j).view_as(e).to(e.dtype)
            n=mask.sum((-2,-1),keepdim=True)
            mu=(e*mask).sum((-2,-1),keepdim=True)/(n+1e-6)
            var=((e-mu).square()*mask).sum((-2,-1),keepdim=True)/(n+1e-6)
            l2_raw=l2_raw+mask*(e-mu)/torch.sqrt(var+1e-6)
        l2=torch.sigmoid(l2_raw)
        z=torch.cat([g,depth,l1],1)
        l3=self.out(self.u1(self.u2(self.d3(self.d2(self.d1(z))))))
        return l1.clamp(0,1),l2.clamp(0,1),l3
