"""Opt-in instance hook: integrates DDM without editing frozen FRP source files.

Tap contract: wraps the *instance* RGB backbone to cache RGB P3, then wraps
the *instance* FRP forward.  It reads raw thermal field ``t`` as S and the
base FRP auxiliary; DDM produces P_ddm from RGB/depth/P3 for the box veto.
The purifier output is never replaced. PHY_DDM=0 returns the original FRP
tuple before any DDM computation, preserving the frozen v2.7.2 path exactly.
"""
from __future__ import annotations
import os, types
import torch
import torch.nn.functional as F
from .ddm_cues import DDM3Cue
from .ddm_branch import DDMBranch

class DDMFusionHook:
 def __init__(self,depth_provider=None,cues=None):
  self.depth_provider=depth_provider;self.cues=cues or DDM3Cue();self.branch=None;self._rgb=self._p3=None;self.fusion_mode='veto_only'
 def attach(self,model,ddm_ckpt=None):
  """Attach to one model instance, optionally loading a frozen DDM checkpoint."""
  if getattr(model,'_ddm_hook',None):return self
  loaded=False;p3_channels=None
  if ddm_ckpt:
   ckpt=torch.load(ddm_ckpt,map_location='cpu',weights_only=False)
   missing={'cues','branch','p3_channels'}-set(ckpt)
   if missing:raise KeyError(f'DDM checkpoint missing keys: {sorted(missing)}')
   self.cues.load_state_dict(ckpt['cues'],strict=True)
   p3_channels=int(ckpt['p3_channels'])
   self.branch=DDMBranch(p3_channels)
   self.branch.load_state_dict(ckpt['branch'],strict=True)
   self.cues.eval().requires_grad_(False);self.branch.eval().requires_grad_(False);loaded=True
  print(f'[DDM-HOOK] ckpt={ddm_ckpt or "None"} p3_channels={p3_channels or "lazy"} loaded={loaded}')
  rgb_backbone=model._rgb_backbone;frp_forward=model.frp.forward
  def rgb_tap(_self,rgb):
   out=rgb_backbone(rgb);self._rgb=rgb;self._p3=out[1];return out
  def frp_tap(_frp,t,i2p,r2):
   pur,aux=frp_forward(t,i2p,r2)
   if os.getenv('PHY_DDM','0') not in {'1','true','True'}:return pur,aux
   if self._rgb is None or self._p3 is None:raise RuntimeError('DDM hook missing RGB P3 tap')
   depth=self.depth_provider(self._rgb) if self.depth_provider else torch.ones_like(t)
   if depth.ndim==3:depth=depth[:,None]
   depth=F.interpolate(depth.float(),self._rgb.shape[-2:],mode='bilinear',align_corners=False)
   l1,l2,l3=self.cues.to(self._rgb.device)(self._rgb.float(),depth)
   if self.branch is None:self.branch=DDMBranch(self._p3.shape[1]).to(self._p3.device)
   elif self.branch.net[0].in_channels-3!=self._p3.shape[1]:raise RuntimeError(f'DDM P3 channels mismatch: checkpoint={self.branch.net[0].in_channels-3}, model={self._p3.shape[1]}')
   self.cues=self.cues.to(self._rgb.device);self.branch=self.branch.to(self._p3.device)
   pddm=F.interpolate(self.branch(l1,l2,l3,self._p3),t.shape[-2:],mode='bilinear',align_corners=False)
   aux=dict(aux,p_ddm=pddm,ddm_enabled=True)
   return pur,aux
  model._rgb_backbone=types.MethodType(rgb_tap,model);model.frp.forward=types.MethodType(frp_tap,model.frp);model._ddm_hook=self
  return self
