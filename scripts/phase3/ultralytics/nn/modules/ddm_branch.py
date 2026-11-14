"""Optional DDM score head; no dependency on frozen FRP files."""
from __future__ import annotations
import os,torch
import torch.nn as nn
import torch.nn.functional as F
class DDMBranch(nn.Module):
 def __init__(self,p3_channels,hidden=64):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(p3_channels+3,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU(),nn.Conv2d(hidden,1,1))
 def forward(self,l1,l2,l3,p3):
  cues=torch.cat([l1,l2,l3],1);cues=F.interpolate(cues,p3.shape[-2:],mode='bilinear',align_corners=False);return torch.sigmoid(self.net(torch.cat([cues,p3],1)))
class DDMScoreFusion(nn.Module):
 """New optional external scoring adapter. PHY_DDM=0 returns S unchanged."""
 def forward(self,s,p_ddm,p_rgb):
  if os.getenv('PHY_DDM','0') not in {'1','true','True'}:return s
  w=torch.tensor([float(os.getenv('PHY_DDM_W1','.5')),float(os.getenv('PHY_DDM_W2','.3')),float(os.getenv('PHY_DDM_W3','.2'))],device=s.device,dtype=s.dtype);w=w/w.sum().clamp_min(1e-6)
  return w[0]*s+w[1]*p_ddm+w[2]*p_rgb
