"""Output-side camera-coordinate reconstruction and component-flow direction."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

def backproject_field(depth, field, fx=640., fy=640., cx=320., cy=320., scale=None, threshold=.05, topk=None):
    d=np.asarray(depth,np.float32);s=np.asarray(field,np.float32);scale=float(scale or os.getenv('PHY_F3D_SCALE','50'));topk=int(topk or os.getenv('PHY_F3D_TOPK','20000'))
    y,x=np.where(s>threshold);strength=s[y,x]
    if topk>0 and len(strength)>topk:
        keep=np.argpartition(strength,-topk)[-topk:];y,x,strength=y[keep],x[keep],strength[keep]
    z=d[y,x]*scale;return np.c_[(x-cx)*z/fx,(y-cy)*z/fy,z,strength].astype(np.float32)
def box_camera_coordinate(depth, box, fx=640., fy=640., cx=320., cy=320., scale=None):
    d=np.asarray(depth,np.float32);x=(float(box[0])+float(box[2]))/2;y=(float(box[1])+float(box[3]))/2;ix=int(np.clip(round(x),0,d.shape[1]-1));iy=int(np.clip(round(y),0,d.shape[0]-1));z=float(d[iy,ix])*float(scale or os.getenv('PHY_F3D_SCALE','50'));return {'X':(x-cx)*z/fx,'Y':(y-cy)*z/fy,'Z':z,'depth_relative':float(d[iy,ix])}
def save_ply(points,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);p=np.asarray(points,np.float32)
    with path.open('w',encoding='ascii') as f:f.write(f'ply\nformat ascii 1.0\nelement vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n');np.savetxt(f,p,fmt='%.6f')
def field_centroids(field,threshold=.05):
    s=np.asarray(field,np.float32)
    try:
        from scipy.ndimage import label
        marks,n=label(s>threshold,structure=np.ones((3,3),np.uint8))
    except ImportError:
        binary=s>threshold;marks=np.zeros(binary.shape,np.int32);n=0
        for yy,xx in zip(*np.where(binary)):
            if marks[yy,xx]:continue
            n+=1;marks[yy,xx]=n;stack=[(yy,xx)]
            while stack:
                y,x=stack.pop()
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<binary.shape[0] and 0<=nx<binary.shape[1] and binary[ny,nx] and not marks[ny,nx]:marks[ny,nx]=n;stack.append((ny,nx))
    out=[]
    for k in range(1,n+1):
        y,x=np.where(marks==k);w=s[y,x]
        out.append({'id':k,'centroid':[float(np.average(x,weights=w)),float(np.average(y,weights=w))],'power':float(w.sum()),'area':int(len(w)),'bbox':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],'mask':marks==k})
    return out
def _iou(a,b):
    inter=np.logical_and(a['mask'],b['mask']).sum();return float(inter/(np.logical_or(a['mask'],b['mask']).sum() or 1))
def match_components(previous,current,max_dist=50.,min_iou=.01):
    pairs=[];used=set()
    for a in previous:
        choices=[(np.linalg.norm(np.subtract(a['centroid'],b['centroid'])),b) for b in current if b['id'] not in used]
        if choices:
            dist,b=min(choices,key=lambda x:x[0]);iou=_iou(a,b)
            if dist<=max_dist and iou>=min_iou:used.add(b['id']);pairs.append({'previous':a,'current':b,'distance':float(dist),'iou':iou})
    return pairs
def direction(v_flow,prior=(0.,0.),wf=None):
    """Fuse one explicit 2-D centroid displacement vector with FRP prior Vt."""
    wf=float(wf or os.getenv('PHY_DIR_WF','.7'));flow=np.asarray(v_flow,float).reshape(2);prior=np.asarray(prior,float).reshape(2);v=wf*flow+(1-wf)*prior;n=np.linalg.norm(v);v=v/n if n else v
    ang=np.degrees(np.arccos(np.clip(np.dot(flow,prior)/(np.linalg.norm(flow)*np.linalg.norm(prior)),-1,1))) if np.linalg.norm(flow) and np.linalg.norm(prior) else 90.
    return {'vector':v.tolist(),'flow':flow.tolist(),'angle_agreement_deg':float(ang),'confidence_factor':1. if ang<30 else .3 if ang>90 else float(1-(ang-30)*.7/60)}
def estimate_direction_from_fields(previous_field,current_field,prior=(0.,0.),threshold=.05,max_dist=50.,min_iou=.01):
    prev=field_centroids(previous_field,threshold);curr=field_centroids(current_field,threshold);pairs=match_components(prev,curr,max_dist,min_iou)
    if not pairs:return {'direction':direction((0.,0.),prior),'pairs':[],'previous_count':len(prev),'current_count':len(curr)}
    weights=np.asarray([p['current']['power'] for p in pairs]);flow=np.average(np.asarray([np.subtract(p['current']['centroid'],p['previous']['centroid']) for p in pairs]),axis=0,weights=weights)
    return {'direction':direction(flow,prior),'pairs':[{'previous_id':p['previous']['id'],'current_id':p['current']['id'],'distance':p['distance'],'iou':p['iou']} for p in pairs],'previous_count':len(prev),'current_count':len(curr)}

def deflection_confidence(field, vec, history, threshold=.05, kept_classes=None):
    """Calibrated core-field, event-detection, and angular-consistency product."""
    strength=np.asarray(field,dtype=float)
    finite=strength[np.isfinite(strength)]
    top_count=max(1,int(np.ceil(finite.size*.1))) if finite.size else 0
    core=float(np.partition(finite,-top_count)[-top_count:].mean()) if top_count else 0.
    fs=float(os.getenv('PHY_CDIR_FS','.5'));minh=int(os.getenv('PHY_CDIR_MINH','10'))
    if not np.isfinite(fs) or fs<=0 or minh<0:raise ValueError('PHY_CDIR_FS must be positive and PHY_CDIR_MINH nonnegative')
    field_factor=float(np.clip(core/fs,0.,1.))
    gate=1. if kept_classes is not None and any(int(c) in {0,1} for c in kept_classes) else .3
    current=np.asarray(vec,dtype=float).reshape(-1)
    past=[np.asarray(v,dtype=float).reshape(-1) for v in history]
    past=[v/np.linalg.norm(v) for v in past if v.shape==current.shape and np.isfinite(v).all() and np.linalg.norm(v)>0]
    if not np.isfinite(current).all() or np.linalg.norm(current)==0 or not past:
        consistency=.5
    else:
        combined=np.mean(past,axis=0)
        norm=np.linalg.norm(combined)
        if norm==0:
            consistency=.5
        else:
            cosine=np.clip(np.dot(current,combined)/(np.linalg.norm(current)*norm),-1.,1.)
            angle=float(np.degrees(np.arccos(cosine)))
            consistency=1. if angle<30. else .3 if angle>90. else 1.-(angle-30.)*.7/60.
    result=float(np.clip(field_factor*gate*np.clip(consistency,0.,1.),0.,1.))
    return min(result,.3) if len(past)<minh else result
