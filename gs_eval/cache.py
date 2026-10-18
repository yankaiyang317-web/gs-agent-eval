import json
from pathlib import Path
import numpy as np
from .storage import atomic_json,atomic_npz,read_json,digest,canonical_hash
from .config import fingerprint,require_verified
from .trajectory import read_poses,unique_calibration
from .alignment import resize,fov_signature
from .geometry import valid_reference,residual_maps
from .scoring import fixed_reliability,evidence_map
from .renderer import EvalRenderer

def cache_identity(c):
    fp,meta=fingerprint(c);return Path(c['cache_root'])/fp,fp,meta

def explicit_mask(pose,c,k):
    name=c['valid_masks'].get(str(pose.get('id',pose.get('frame'))))
    if name is None:return None
    from PIL import Image
    return np.array(Image.open(name).convert('L').resize((k['width'],k['height']),Image.Resampling.NEAREST))>0

def save_render(path,render):atomic_npz(path,**render)
def read_render(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}

def render_bank(renderer,poses,kind,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);renderer.load(kind)
    for i,p in enumerate(poses):
        out,k=renderer.render(p,kind);save_render(folder/f'{i:05d}.npz',out)
    return folder

def group_mean(keys,values):
    unique,inv=np.unique(keys,axis=0,return_inverse=True);out=np.zeros((len(unique),)+values.shape[1:],dtype=np.float64);np.add.at(out,inv,values);count=np.bincount(inv);out/=count.reshape((-1,)+(1,)*(out.ndim-1));return unique,out,inv

def prepare_reference(c):
    require_verified(c);root,fp,meta=cache_identity(c)
    if (root/'reference_manifest.json').exists():
        verify_cache(root,'reference_manifest.json');return root
    poses=read_poses(c['reference_poses']);renderer=EvalRenderer(c);render_bank(renderer,poses,'high',root/'reference_high');samples=[];perview=[];allpoints=[];allkeys=[];stride=c['surface']['reference_pixel_stride'];lo=np.array(c['roi']['min']);vox=c['surface']['voxel_size']
    for i,p in enumerate(poses):
        high=read_render(root/'reference_high'/f'{i:05d}.npz');k=resize(p['intrinsics'],c['evaluation_camera']['width'],c['evaluation_camera']['height']);mask,points=valid_reference(high,p['camera_to_world'],k,c,explicit_mask(p,c,k));sample=np.zeros(mask.shape,bool);sample[::stride,::stride]=True;valid=mask&sample;xyz=points[valid];keys=np.floor((xyz-lo)/vox).astype(np.int64);uk,means,inv=group_mean(keys,xyz);samples.append((valid,keys,uk));allkeys.append(uk);allpoints.append(means)
    if not any(len(k) for k in allkeys):raise ValueError('EMPTY_REFERENCE_DOMAIN')
    keys,points,globalinv=group_mean(np.concatenate(allkeys),np.concatenate(allpoints));keymap={tuple(k):i for i,k in enumerate(keys)};values=[[] for _ in keys];invalid=0
    render_bank(renderer,poses,'low',root/'reference_low')
    for i,p in enumerate(poses):
        high=read_render(root/'reference_high'/f'{i:05d}.npz');low=read_render(root/'reference_low'/f'{i:05d}.npz');k=resize(p['intrinsics'],c['evaluation_camera']['width'],c['evaluation_camera']['height']);mask,_=valid_reference(high,p['camera_to_world'],k,c,explicit_mask(p,c,k));_,_,loss=residual_maps(high,low,mask,c);valid,ks,uk=samples[i];invalid+=not mask.any()
        if len(ks):
            kmean,means,_=group_mean(ks,loss[valid]);
            for key,v in zip(kmean,means):values[keymap[tuple(key)]].append(float(v))
    errors=np.array([np.median(v) if v else np.nan for v in values]);weights=np.clip((errors-c['defect']['loss_low'])/(c['defect']['loss_high']-c['defect']['loss_low']),0,1)
    if not np.isfinite(weights).all():raise ValueError('INVALID_REFERENCE_DEFECT_WEIGHTS')
    atomic_npz(root/'reference_surface.npz',keys=keys,points=points,observations=np.array([len(v) for v in values]));atomic_npz(root/'defect_weights.npz',weights=weights,loss=errors)
    atomic_json(root/'reference_manifest.json',{'fingerprint':fp,'metadata':meta,'reference_voxels':len(keys),'reference_views':len(poses),'invalid_reference_frames':invalid,'reference_domain':c.get('reference_domain',{'reachability':'provided_pose_bank_unverified_reachability'}),'files':{n:digest(root/n) for n in ['reference_surface.npz','defect_weights.npz']},'reference_sampling':'nearest pixel, stride fixed, voxel point = mean of per-view means','status':'prepared'});print('reference prepared',len(keys),flush=True);return root

def verify_cache(root,name):
    m=read_json(root/name)
    for n,h in m['files'].items():
        if not (root/n).is_file() or digest(root/n)!=h:raise ValueError('CACHE_CORRUPT_OR_INCOMPLETE: '+str(root/n))
    return m

def prepare_reliability(c):
    require_verified(c);root=prepare_reference(c)
    if (root/'reliability_manifest.json').exists():verify_cache(root,'reliability_manifest.json');return root
    if not c['reliability']['enabled']:raise ValueError('RELIABILITY_DISABLED')
    original=read_poses(c['calibration_poses']);poses,duplicates=unique_calibration(original);renderer=EvalRenderer(c);render_bank(renderer,poses,'high',root/'calibration_high');renderer.load('low');deltas=[];support=np.zeros(renderer.scene.count,dtype=np.float64);stats=[]
    for i,p in enumerate(poses):
        high=read_render(root/'calibration_high'/f'{i:05d}.npz');low,k=renderer.render(p,'low');mask,_=valid_reference(high,p['camera_to_world'],k,c,explicit_mask(p,c,k));rgb,depth,loss=residual_maps(high,low,mask,c);q=evidence_map(loss,mask,c);grad=renderer.contribution(p,[q,mask.astype(float)]);deltas.append(grad[:,0].astype(np.float64)*c['reliability']['lambda_evidence']);support+=grad[:,1].astype(np.float64);stats.append({'id':p['id'],'rgb':float(rgb[mask].mean()) if mask.any() else None,'relative_depth':float(depth[mask].mean()) if mask.any() else None})
    p,g,raw,diagnostics=fixed_reliability(deltas,support,c);atomic_npz(root/'reliability.npz',p=p,g=g,raw_logodds=raw,support=support);diagnostics.update({'duplicate_calibration_poses_removed':duplicates,'calibration_views':len(poses),'calibration_residuals':stats,'gaussians':len(p)});atomic_json(root/'reliability_manifest.json',{'fingerprint':canonical_hash([root.name,digest(root/'reliability.npz')]),'reference_fingerprint':root.name,'diagnostics':diagnostics,'files':{'reliability.npz':digest(root/'reliability.npz')},'adaptation':'paper inspired adaptation, not full 2D GauSS-MI, not calibrated correctness probabilities'});print('reliability prepared',diagnostics,flush=True);return root
