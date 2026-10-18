from pathlib import Path
import copy
import numpy as np
from .storage import read_json,canonical_hash,digest
from .alignment import similarity

DEFAULTS={
 'evaluation_camera':{'width':640,'height':480,'resize_policy':'preserve_fov'},
 'budget':{'mode':'recorded_frames','frames':81,'start_frame':0,'allow_partial':False},
 'surface':{'reference_pixel_stride':2,'alpha_high_min':.9,'visibility_abs_tolerance':.1,'visibility_rel_tolerance':.01,'min_projected_voxel_pixels':1.},
 'residual':{'rgb_weight':1.,'depth_weight':1.,'depth_relative_clip':1.,'depth_floor':.1,'alpha_low_depth_min':.7,'loss_epsilon':1e-4},
 'reliability':{'enabled':True,'method':'frozen_scalar_logodds','lambda_loss':10.,'lambda_evidence':.01,'logodds_clip':8.},
 'defect':{'per_view_aggregation':'mean','across_view_aggregation':'median','loss_low':.02,'loss_high':.20},
 'reconstruction_validation':{'enabled':False},'seed':0,
 'trajectory_checks':{'translation_jump':2.,'rotation_jump_deg':120.}}

def load_config(path,require_assets=True):
    path=Path(path).resolve();raw=read_json(path);c=copy.deepcopy(DEFAULTS)
    for k,v in raw.items():
        if isinstance(v,dict) and isinstance(c.get(k),dict):c[k].update(v)
        else:c[k]=v
    for k in ['schema_version','scene_id','low_scene_manifest','high_ply','high_to_low_similarity','roi','calibration_poses','reference_poses','cache_root']:
        if k not in c:raise ValueError('MISSING_CONFIG: '+k)
    if c['schema_version']!=1:raise ValueError('UNSUPPORTED_SCHEMA')
    similarity(c['high_to_low_similarity'])
    for k in ['near','far','voxel_size']:
        if k not in raw.get('surface',{}):raise ValueError('MISSING_EXPLICIT_SURFACE: '+k)
    lo=np.array(c['roi']['min']);hi=np.array(c['roi']['max'])
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>=hi):raise ValueError('INVALID_ROI')
    s=c['surface'];r=c['residual'];rel=c['reliability'];b=c['budget'];cam=c['evaluation_camera']
    if not 0<s['near']<s['far'] or s['voxel_size']<=0 or not 0<s['alpha_high_min']<=1 or int(s['reference_pixel_stride'])!=s['reference_pixel_stride'] or s['reference_pixel_stride']<1:raise ValueError('INVALID_SURFACE')
    if min(s['visibility_abs_tolerance'],s['visibility_rel_tolerance'],s['min_projected_voxel_pixels'])<0:raise ValueError('INVALID_VISIBILITY')
    if min(r['depth_floor'],r['depth_relative_clip'],r['loss_epsilon'],rel['lambda_loss'],rel['logodds_clip'])<=0 or min(r['rgb_weight'],r['depth_weight'],rel['lambda_evidence'])<0 or not 0<=r['alpha_low_depth_min']<=1:raise ValueError('INVALID_RESIDUAL_OR_RELIABILITY')
    if b['mode']!='recorded_frames' or b['frames']!=81 or b['start_frame']<0 or int(b['start_frame'])!=b['start_frame']:raise ValueError('V1_REQUIRES_81_RECORDED_FRAMES')
    if cam['resize_policy']!='preserve_fov' or min(cam['width'],cam['height'])<=0 or int(cam['width'])!=cam['width'] or int(cam['height'])!=cam['height']:raise ValueError('INVALID_EVALUATION_CAMERA')
    if c['defect']['loss_high']<=c['defect']['loss_low'] or c['defect']['per_view_aggregation']!='mean' or c['defect']['across_view_aggregation']!='median':raise ValueError('INVALID_DEFECT')
    if rel['method']!='frozen_scalar_logodds':raise ValueError('UNSUPPORTED_RELIABILITY')
    for sec in [s,r,rel,cam,c['trajectory_checks']]:
        if any(isinstance(v,(float,int)) and not np.isfinite(v) for v in sec.values()):raise ValueError('NONFINITE_CONFIG')
    for key in ['low_scene_manifest','high_ply','calibration_poses','reference_poses','cache_root']:
        c[key]=str((path.parent/c[key]).resolve())
        if require_assets and key!='cache_root' and not Path(c[key]).is_file():raise ValueError('MISSING_ASSET: '+c[key])
    if require_assets:
        m=read_json(c['low_scene_manifest']);c['_low_ply']=str((Path(c['low_scene_manifest']).parent/m['ply']).resolve())
        if not Path(c['_low_ply']).is_file():raise ValueError('MISSING_LOW_PLY: '+c['_low_ply'])
    else:c['_low_ply']=None
    masks=c.get('valid_masks',{})
    c['valid_masks']={str(k):str((path.parent/v).resolve()) for k,v in masks.items()}
    if require_assets:
        for p in c['valid_masks'].values():
            if not Path(p).is_file():raise ValueError('MISSING_MASK: '+p)
    c['_config_path']=str(path);return c

def fingerprint(c):
    import importlib.metadata,platform
    from . import FORMULA_VERSION
    files={k:digest(c[k]) for k in ['high_ply','_low_ply','calibration_poses','reference_poses']}
    files['masks']={k:digest(v) for k,v in c['valid_masks'].items()}
    files['low_scene_manifest']=digest(c['low_scene_manifest'])
    scene=read_json(c['low_scene_manifest'])
    if scene.get('collision_mesh'):
        collision=(Path(c['low_scene_manifest']).parent/scene['collision_mesh']).resolve();files['collision_mesh']=digest(collision)
    code={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))};code['vendored_runtime']={p.name:digest(p) for p in sorted((Path(__file__).parent/'_vendor').glob('*.py'))};versions={'python':platform.python_version()}
    for k in ['numpy','torch','gsplat']:
        try:versions[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:versions[k]='unavailable'
    from .renderer import environment_identity
    versions.update(environment_identity())
    protocol={k:v for k,v in c.items() if not k.startswith('_') and k not in ['cache_root','reconstruction_validation']}
    value={'files':files,'config':protocol,'formula':FORMULA_VERSION,'code':code,'versions':versions,'contribution_backend':'gsplat_ND_feature_gradient_v1'}
    return canonical_hash(value),value

def require_verified(c):
    if c['high_to_low_similarity'].get('verified') is not True:raise ValueError('ALIGNMENT_UNVERIFIED: run validate, inspect overlays, then explicitly mark verified')

