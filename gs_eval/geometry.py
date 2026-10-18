import numpy as np
from .alignment import backproject,rigid

def valid_reference(render,pose,k,c,explicit_mask=None):
    d=render['depth'];a=render['alpha'];points=backproject(d,pose,k);lo=np.asarray(c['roi']['min']);hi=np.asarray(c['roi']['max']);s=c['surface']
    mask=np.isfinite(d)&(a>=s['alpha_high_min'])&(d>=s['near'])&(d<=s['far'])&np.isfinite(points).all(axis=-1)&(points>=lo).all(axis=-1)&(points<hi).all(axis=-1)
    if explicit_mask is not None:
        if explicit_mask.shape!=mask.shape:raise ValueError('MASK_SHAPE_MISMATCH')
        mask&=explicit_mask.astype(bool)
    return mask,points

def residual_maps(high,low,mask,c):
    r=c['residual'];rgb=np.mean(np.abs(np.clip(low['rgb'],0,1)-np.clip(high['rgb'],0,1)),axis=-1);hd=high['depth'];ld=low['depth'];valid=np.isfinite(ld)&(ld>0)&(low['alpha']>=r['alpha_low_depth_min']);dep=np.full(hd.shape,r['depth_relative_clip'],dtype=float)
    np.divide(np.abs(ld-hd),np.maximum(hd,r['depth_floor']),out=dep,where=valid);dep=np.where(valid,np.minimum(dep,r['depth_relative_clip']),r['depth_relative_clip']);loss=r['rgb_weight']*rgb+r['depth_weight']*dep
    return rgb,dep,loss

def visible_surface(points,pose,k,high,mask,c):
    s=c['surface'];m=rigid(pose);cam=(points-m[:3,3])@m[:3,:3];z=cam[:,2];good=np.isfinite(cam).all(axis=1)&(z>=s['near'])&(z<=s['far']);u=k['fx']*cam[:,0]/np.maximum(z,1e-12)+k['cx'];v=k['fy']*cam[:,1]/np.maximum(z,1e-12)+k['cy']
    # Nearest-pixel, half-up rounding, strict image bounds; fixed for every metric.
    good&=(u>=0)&(u<k['width']-.5)&(v>=0)&(v<k['height']-.5);ii=np.where(good)[0];x=np.floor(u[ii]+.5).astype(int);y=np.floor(v[ii]+.5).astype(int);d=high['depth'][y,x]
    hit=mask[y,x]&(np.abs(z[ii]-d)<=s['visibility_abs_tolerance']+s['visibility_rel_tolerance']*d)&(min(k['fx'],k['fy'])*s['voxel_size']/z[ii]>=s['min_projected_voxel_pixels'])
    return ii[hit]

def union_update(seen,visible,weights=None):
    old=int(seen.sum());seen[visible]=True;coverage=float(seen.mean());den=None if weights is None else float(weights.sum());weighted=None if not den else float(weights[seen].sum()/den);return int(seen.sum())-old,coverage,weighted
