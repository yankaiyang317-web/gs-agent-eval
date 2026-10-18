from pathlib import Path
import time
import numpy as np
from .storage import digest,read_json,atomic_json,canonical_hash
from .config import require_verified
from .trajectory import select_trajectory,trajectory_diagnostics,read_poses
from .alignment import resize,fov_signature
from .cache import cache_identity,verify_cache,read_render,render_bank,explicit_mask
from .renderer import EvalRenderer
from .geometry import valid_reference,residual_maps,visible_surface,union_update
from .scoring import view_utility
from .reporting import report,image_pair,heatmap

def mean_valid(values):
    v=[x for x in values if x is not None];return float(np.mean(v)) if v else None

def rank_corr(x,y):
    pairs=[(a,b) for a,b in zip(x,y) if a is not None and b is not None]
    if len(pairs)<3:return None
    def rank(z):
        _,inv,counts=np.unique(z,return_inverse=True,return_counts=True);means=np.cumsum(counts)-(counts+1)/2;return means[inv]
    a,b=np.array(pairs).T;a,b=rank(a),rank(b)
    if np.std(a)<1e-10 or np.std(b)<1e-10:return None
    return float(np.corrcoef(a,b)[0,1])

def evaluate(c,run_dir,out,allow_partial=False,coverage_only=False,skip_prediction=False,method_id=None):
    require_verified(c);start=time.perf_counter();out=Path(out)
    if (out/'summary.json').exists():raise ValueError('OUTPUT_ALREADY_EXISTS: choose a new output directory')
    selected,partial,runmeta=select_trajectory(run_dir,c,allow_partial);diag=trajectory_diagnostics(selected,c);root,fp,meta=cache_identity(c)
    if not (root/'reference_manifest.json').exists():raise ValueError('REFERENCE_CACHE_MISSING: run prepare-reference')
    rm=verify_cache(root,'reference_manifest.json');prediction=not (coverage_only or skip_prediction);rel=None;gain=None;relhash=None
    if prediction:
        if not c['reliability']['enabled']:raise ValueError('RELIABILITY_DISABLED: use --skip-prediction or --coverage-only')
        if not (root/'reliability_manifest.json').exists():raise ValueError('RELIABILITY_CACHE_MISSING: run prepare-reliability')
        rel=verify_cache(root,'reliability_manifest.json');relhash=digest(root/'reliability.npz')
        with np.load(root/'reliability.npz') as a:gain=a['g'].copy()
    # Cache integrity and one common FOV are checked before any scores are published.
    bank=read_poses(c['reference_poses']);sig=fov_signature(bank[0]['intrinsics'])
    if any(not np.allclose(sig,fov_signature(r['intrinsics']),atol=1e-6) for r in selected+bank):raise ValueError('PROTOCOL_FOV_MISMATCH')
    with np.load(root/'reference_surface.npz') as a:points=a['points'].copy()
    weights=None
    if not coverage_only:
        with np.load(root/'defect_weights.npz') as a:weights=a['weights'].copy()
    out.mkdir(parents=True,exist_ok=True);renderer=EvalRenderer(c);render_bank(renderer,selected,'high',out/'diagnostics/high_frames')
    if not coverage_only:render_bank(renderer,selected,'low',out/'diagnostics/low_frames')
    seen=np.zeros(len(points),bool);rows=[]
    if prediction:renderer.load('low')
    for i,p in enumerate(selected):
        high=read_render(out/'diagnostics/high_frames'/f'{i:05d}.npz');k=resize(p['intrinsics'],c['evaluation_camera']['width'],c['evaluation_camera']['height']);mask,_=valid_reference(high,p['camera_to_world'],k,c,explicit_mask(p,c,k));visible=visible_surface(points,p['camera_to_world'],k,high,mask,c);new,coverage,defect=union_update(seen,visible,weights);rgbval=depthval=None;raw=norm=control=None;lowvalid=None
        if not coverage_only:
            low=read_render(out/'diagnostics/low_frames'/f'{i:05d}.npz');rgb,dep,loss=residual_maps(high,low,mask,c);rgbval=float(rgb[mask].mean()) if mask.any() else None;depthval=float(dep[mask].mean()) if mask.any() else None;lowvalid=float((np.isfinite(low['depth'])&(low['depth']>0)&(low['alpha']>=c['residual']['alpha_low_depth_min'])).mean())
        if prediction:
            feature=renderer.feature(p,np.column_stack([gain,np.full_like(gain,np.log(2))]));raw,norm=view_utility(feature[...,0],mask);_,control=view_utility(feature[...,1],mask)
        pos=np.array(p['camera_to_world'])[:3,3];rows.append({'frame':p['frame'],'source_frame':p.get('source_frame'),'x':float(pos[0]),'y':float(pos[1]),'z':float(pos[2]),'I_raw':raw,'I_normalized':norm,'I_constant_p05':control,'valid_reference_fraction':float(mask.mean()),'valid_low_fraction':lowvalid,'new_surface_voxels':new,'cumulative_surface_coverage':coverage,'cumulative_defect_coverage':defect,'gt_rgb_residual':rgbval,'gt_relative_depth_residual':depthval,'jump_flag':p['frame'] in diag['jump_frames']})
    if prediction and digest(root/'reliability.npz')!=relhash:raise RuntimeError('FROZEN_RELIABILITY_MODIFIED')
    validframes=sum(r['gt_rgb_residual'] is not None for r in rows);origin=runmeta.get('trajectory_origin','existing_recorded_run');reach=c.get('reference_domain',{}).get('reachability','provided_pose_bank_unverified_reachability');unvalidated='synthetic' in origin or reach=='provided_pose_bank_unverified_reachability'
    status='partial' if partial else ('unvalidated' if unvalidated else 'complete');bstatus='coverage_only' if coverage_only else ('prediction_skipped' if not prediction else rel['diagnostics']['reliability_status']);final=rows[-1];summary={'schema_version':1,'status':status,'scene_id':c['scene_id'],'method_id':method_id or runmeta.get('method_id',Path(run_dir).name),'budget_mode':'recorded_frames','requested_frames':81,'evaluated_frames':len(rows),'metric_families':{'A_surface_coverage':{'primary':{'surface_coverage_at_81':None if partial else final['cumulative_surface_coverage']},'supporting':{'surface_coverage_auc_81':None if partial else float(np.mean([r['cumulative_surface_coverage'] for r in rows]))},'at_evaluated_frames':final['cumulative_surface_coverage']},'B_view_value':{'primary':{'frozen_gaussmi_mean':mean_valid([r['I_normalized'] for r in rows])},'observed_reference':{'gt_rgb_residual_mean':mean_valid([r['gt_rgb_residual'] for r in rows]),'gt_relative_depth_residual_mean':mean_valid([r['gt_relative_depth_residual'] for r in rows]),'valid_reference_frames':validframes,'invalid_reference_fraction':sum(r['valid_reference_fraction']==0 for r in rows)/len(rows)},'supporting':{'defect_weighted_coverage_at_81':None if partial else final['cumulative_defect_coverage']},'defect_status':'not_computed' if weights is None else ('no_weighted_defect_in_reference' if weights.sum()==0 else 'available'),'interpretation':'prediction_and_reference_residuals_not_realized_repair_gain'}},'comparability':{'continuous_navigation_comparable':diag['continuous_navigation_comparable'],'reference_fingerprint':fp,'reliability_fingerprint':rel['fingerprint'] if rel else None,'protocol_fingerprint':fp,'fov_signature':runmeta['fov_signature'],'start_camera_to_world':selected[0]['camera_to_world']},'diagnostics':{'reference_voxels':len(points),'invalid_reference_frames':sum(r['valid_reference_fraction']==0 for r in rows),'reliability_status':bstatus,'trajectory_origin':origin,'spearman_prediction_rgb':rank_corr([r['I_normalized'] for r in rows],[r['gt_rgb_residual'] for r in rows]),'spearman_prediction_depth':rank_corr([r['I_normalized'] for r in rows],[r['gt_relative_depth_residual'] for r in rows])},'trajectory_diagnostics':diag,'reference_domain':c.get('reference_domain',{'reachability':reach}),'inputs':{'low_sha256':meta['files']['_low_ply'],'high_sha256':meta['files']['high_ply'],'trajectory_sha256':runmeta['trajectory_sha256'],'config':c['_config_path'],'method_metadata':runmeta},'formula':'frozen_gaussmi_3d_v1','elapsed_s':time.perf_counter()-start}
    # Deterministic representative diagnostics: no cherry-picking the scored trajectory.
    examples=set(range(0,len(rows),20))|{len(rows)-1}
    if prediction:
        utility=np.array([r['I_normalized'] for r in rows]);examples|={int(utility.argmin()),int(utility.argmax())}
    for i in sorted(examples):
        high=read_render(out/'diagnostics/high_frames'/f'{i:05d}.npz');k=resize(selected[i]['intrinsics'],c['evaluation_camera']['width'],c['evaluation_camera']['height']);mask,_=valid_reference(high,selected[i]['camera_to_world'],k,c,explicit_mask(selected[i],c,k))
        if not coverage_only:
            low=read_render(out/'diagnostics/low_frames'/f'{i:05d}.npz');image_pair(out/'diagnostics/example_rgb_pairs'/f'{i:05d}.png',high,low);rgb,dep,_=residual_maps(high,low,mask,c);heatmap(out/'diagnostics/residual_maps'/f'{i:05d}_rgb.png',rgb,mask,'GT RGB residual');heatmap(out/'diagnostics/residual_maps'/f'{i:05d}_depth.png',dep,mask,'GT relative depth residual')
        if prediction:heatmap(out/'diagnostics/gain_maps'/f'{i:05d}.png',renderer.feature(selected[i],gain)[...,0],mask,'Frozen gain (unclipped)')
    summary['diagnostics']['example_indices']=sorted(examples);summary['elapsed_s']=time.perf_counter()-start;report(out,summary,rows,selected);print('evaluated',summary['method_id'],summary['metric_families'],flush=True);return summary

