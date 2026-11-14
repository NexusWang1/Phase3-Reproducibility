"""Frozen pre-L2-repair DDM cue code for the A.1/A.3 regression probe only.

Never import this into v3 training or evaluation. It records the 2026-09-30
implementation including inclusive quantile-bin endpoints.
"""
from __future__ import annotations
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
        g=_gray(rgb);gi=_sobel(g);gd=_sobel(depth);pad=self.w//2
        mean=lambda z:F.avg_pool2d(z,self.w,1,pad)
        cov=mean(gi*gd)-mean(gi)*mean(gd);rho=cov/(mean(gi.square()).sub(mean(gi).square()).clamp_min(1e-6).sqrt()*mean(gd.square()).sub(mean(gd).square()).clamp_min(1e-6).sqrt())
        l1=mean(gi)*(1-rho.clamp(-1,1))/2
        edge_i=torch.sigmoid((gi-gi.mean((-2,-1),keepdim=True))/gi.std((-2,-1),keepdim=True).clamp_min(1e-6))
        edge_d=torch.sigmoid((gd-gd.mean((-2,-1),keepdim=True))/gd.std((-2,-1),keepdim=True).clamp_min(1e-6))
        q=torch.quantile(depth.flatten(2),torch.linspace(0,1,self.k+1,device=depth.device),dim=2).squeeze(-1).permute(1,0).unsqueeze(-1).unsqueeze(-1)
        l2=sum((edge_i*edge_d*((depth>=q[:,j:j+1])&(depth<=q[:,j+1:j+2])).float()) for j in range(self.k))/self.k
        z=torch.cat([g,depth,l1],1)
        l3=self.out(self.u1(self.u2(self.d3(self.d2(self.d1(z))))))
        return l1.clamp(0,1),l2.clamp(0,1),l3
