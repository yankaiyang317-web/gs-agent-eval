"""Reusable CLI: frozen benchmark state + recorded trajectory; no Agent imports."""
import argparse,copy,json,os,platform
from pathlib import Path
import numpy as np
from gs_eval.storage import read_json,digest,canonical_hash,atomic_json
from gs_eval.trajectory import read_poses,trajectory_diagnostics
from gs_eval.alignment import fov_signature,resize
from gs_eval.geometry import valid_reference,residual_maps,visible_surface,union_update
from gs_eval.reporting import report
from . import __version__

def load_benchmark(path,asset_root=None):
    path=Path(path).resolve();b=read_json(path)
    if b.get('schema_version')!=1:raise ValueError('UNSUPPORTED_BENCHMARK_SCHEMA')
    if b['budget']['frames']!=81:raise ValueError('FROZEN_BUDGET_MUST_BE_81')
    frozen=copy.deepcopy(b);frozen.pop('locations',None)
    for spec in frozen['assets'].values():spec.pop('path',None)
    b['_fingerprint']=canonical_hash(frozen);locations={key:os.path.expandvars(value) for key,value in b['locations'].items()}
    if asset_root:locations['asset_root']=str(Path(asset_root).resolve())
    for key,value in locations.items():
        if not Path(value).is_absolute():locations[key]=str((path.parent/value).resolve())
    resolved={}
    for name,spec in b['assets'].items():
        value=os.path.expandvars(spec['path'])
        for key,prefix in locations.items():value=value.replace('${'+key+'}',prefix)
        p=Path(value)
        if not p.is_absolute():p=path.parent/p
        p=p.resolve()
        if not p.is_file():raise ValueError('MISSING_BENCHMARK_ASSET: '+str(p))
        if digest(p)!=spec['sha256']:raise ValueError('FROZEN_ASSET_HASH_MISMATCH: '+name)
        resolved[name]=str(p)
    b['_assets']=resolved;b['_path']=str(path);return b

def select_run(run_dir,b,clip_index=0,allow_partial=False):
    if clip_index<0:raise ValueError('NEGATIVE_CLIP_INDEX')
    run=Path(run_dir);rows=read_poses(run/'trajectory.jsonl');ids=[r.get('frame') for r in rows]
    if any(not isinstance(x,int) for x in ids) or any(y<=x for x,y in zip(ids,ids[1:])):raise ValueError('INVALID_RECORDED_FRAME_ORDER')
    start=81*clip_index;selected=[r for r in rows if start<=r['frame']<start+81]
    if not selected or [r['frame'] for r in selected]!=list(range(start,start+len(selected))):raise ValueError('NONCONTIGUOUS_OR_MISSING_CLIP')
    partial=len(selected)!=81
    if partial and not allow_partial:raise ValueError('INSUFFICIENT_FRAMES')
    expected=np.array(b['camera_fov_signature'])
    if any(not np.allclose(fov_signature(r['intrinsics']),expected,atol=1e-6) for r in selected):raise ValueError('BENCHMARK_FOV_MISMATCH')
    campaign=read_json(run/'campaign.json') if (run/'campaign.json').exists() else {}
    meta=read_json(run/'run_metadata.json') if (run/'run_metadata.json').exists() else {}
    scene=campaign.get('scene') or meta.get('scene');source='metadata'
    if not scene:scene=run.parent.name;source='parent_directory_label'
    if scene!=b['scene_name']:raise ValueError('RUN_SCENE_MISMATCH: '+str(scene))
    if campaign.get('world_up') is not None and not np.allclose(campaign['world_up'],b['coordinates']['world_up'],atol=1e-6):raise ValueError('WORLD_UP_MISMATCH')
    environment=meta.get('environment',{})
    for key in ['collision_sha256','camera_radius','camera_body_height','collision_step']:
        if key in environment and environment[key]!=b['navigation'][key]:raise ValueError('NAVIGATION_PROTOCOL_MISMATCH: '+key)
    return selected,partial,{'trajectory_sha256':digest(run/'trajectory.jsonl'),'scene_identity_source':source,'campaign':campaign,'run_metadata':meta,'navigation_protocol_verified_from_metadata':all(key in environment for key in ['collision_sha256','camera_radius','camera_body_height','collision_step'])}

def score(benchmark,run_dir,out,clip_index=0,allow_partial=False,method_id=None,agent_version=None):
    from gs_eval.renderer import EvalRenderer,environment_identity
    from gs_eval.cache import save_render,read_render
    out=Path(out)
    if out.exists():raise ValueError('OUTPUT_ALREADY_EXISTS: '+str(out))
    poses,partial,meta=select_run(run_dir,benchmark,clip_index,allow_partial);assets=benchmark['_assets'];c=copy.deepcopy(benchmark['scoring']);c.update(high_ply=assets['high_ply'],_low_ply=assets['low_ply'])
    with np.load(assets['reference_surface']) as a:points=a['points'].copy()
    with np.load(assets['defect_weights']) as a:weights=a['weights'].copy()
    with np.load(assets['reliability']) as a:g=a['g'].copy();support=a['support'].copy()
    if len(points)!=len(weights) or not np.isfinite(g).all():raise ValueError('INVALID_FROZEN_STATE')
    out.mkdir(parents=True);renderer=EvalRenderer(c)
    for kind in ['high','low']:
        for i,p in enumerate(poses):
            render,k=renderer.render(p,kind);save_render(out/'diagnostics'/kind/f'{i:05d}.npz',render)
    scene=renderer.load('low')
    if scene.count!=len(g):raise ValueError('RELIABILITY_GAUSSIAN_ORDER_OR_COUNT_MISMATCH')
    features=np.column_stack([g,np.full_like(g,np.log(2)),(support<=1e-10).astype(float)]);seen=np.zeros(len(points),bool);rows=[];diag=trajectory_diagnostics(poses,c)
    for i,p in enumerate(poses):
        high=read_render(out/'diagnostics/high'/f'{i:05d}.npz');low=read_render(out/'diagnostics/low'/f'{i:05d}.npz');k=resize(p['intrinsics'],c['evaluation_camera']['width'],c['evaluation_camera']['height']);mask,_=valid_reference(high,p['camera_to_world'],k,c)
        ids=visible_surface(points,p['camera_to_world'],k,high,mask,c);new,coverage,defect=union_update(seen,ids,weights);rgb,depth,_=residual_maps(high,low,mask,c);feat=renderer.feature(p,features);raw=float(feat[...,0][mask].sum(dtype=np.float64));utility=raw/mask.size;position=np.array(p['camera_to_world'])[:3,3]
        rows.append({'frame':p['frame'],'campaign_frame':p.get('campaign_frame',p['frame']),'source_frame':p.get('source_frame'),'x':float(position[0]),'y':float(position[1]),'z':float(position[2]),'I_raw':raw,'I_normalized':utility,'I_constant_p05':float(feat[...,1][mask].sum(dtype=np.float64)/mask.size),'unknown_state_render_mass':float(feat[...,2][mask].sum(dtype=np.float64)/mask.size),'valid_reference_fraction':float(mask.mean()),'new_surface_voxels':new,'cumulative_surface_coverage':coverage,'cumulative_defect_coverage':defect,'gt_rgb_residual':float(rgb[mask].mean()) if mask.any() else None,'gt_relative_depth_residual':float(depth[mask].mean()) if mask.any() else None,'jump_flag':p['frame'] in diag['jump_frames']})
    mean=lambda key:float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None
    final=rows[-1];fp=benchmark['_fingerprint'];version=agent_version or meta['run_metadata'].get('agent_version','unspecified')
    summary={'schema_version':1,'status':'partial' if partial else 'complete','scene_id':benchmark['scene_name'],'method_id':method_id or Path(run_dir).name,'evaluated_frames':len(rows),'requested_frames':81,'budget_mode':'recorded_frames','metric_families':{'A_surface_coverage':{'primary':{'surface_coverage_at_81':None if partial else final['cumulative_surface_coverage']},'supporting':{'surface_coverage_auc_81':None if partial else mean('cumulative_surface_coverage')},'at_evaluated_frames':final['cumulative_surface_coverage']},'B_view_value':{'primary':{'frozen_gaussmi_mean':mean('I_normalized')},'observed_reference':{'gt_rgb_residual_mean':mean('gt_rgb_residual'),'gt_relative_depth_residual_mean':mean('gt_relative_depth_residual'),'valid_reference_frames':sum(r['gt_rgb_residual'] is not None for r in rows)},'supporting':{'defect_weighted_coverage_at_81':None if partial else final['cumulative_defect_coverage']}}},'comparability':{'continuous_navigation_comparable':diag['continuous_navigation_comparable'],'protocol_fingerprint':fp,'reference_fingerprint':benchmark['assets']['reference_surface']['sha256'],'reliability_fingerprint':benchmark['assets']['reliability']['sha256'],'start_camera_to_world':poses[0]['camera_to_world'],'evaluator_version':__version__,'implementation_fingerprint':implementation_fingerprint()},'reference_domain':benchmark['reference_domain'],'trajectory_diagnostics':diag,'diagnostics':{'reliability_status':'full_input_evidence_state_scientifically_unvalidated','trajectory_origin':meta['run_metadata'].get('trajectory_origin','recorded_agent_run'),'reference_voxels':len(points),'budget_start_frame':81*clip_index,'example_indices':[]},'provenance':{'benchmark_id':benchmark['benchmark_id'],'benchmark_file':benchmark['_path'],'agent_version':version,'method_id':method_id,'inputs':meta,'environment':environment_identity(),'asset_hashes':{key:value['sha256'] for key,value in benchmark['assets'].items()}},'interpretation':'Frozen v1 prototype: A coverage and B weak-region view utility separately; not realized reconstruction gain; no combined score'}
    report(out,summary,rows,poses)
    (out/'HOW_TO_READ.md').write_text('本报告的 B 使用完整128张已知训练输入的 RGBD+深度法线固定状态；不是旧24视角状态，也没有重新校准。A/B分开解释。分数不是概率或实际修复量。若导航元数据不足，navigation_protocol_verified_from_metadata=false，仅说明未核实，并不表示环境不一致。科学局限见冻结基准profile。\n',encoding='utf-8')
    print(json.dumps({'out':str(out),'benchmark_id':benchmark['benchmark_id'],'A':summary['metric_families']['A_surface_coverage'],'B':summary['metric_families']['B_view_value'],'navigation_metadata_verified':meta['navigation_protocol_verified_from_metadata']},ensure_ascii=False),flush=True)
    return summary

def implementation_fingerprint():
    import gs_eval
    paths=[*Path(gs_eval.__file__).parent.rglob('*.py'),*Path(__file__).parent.rglob('*.py')]
    return canonical_hash({str(p.relative_to(Path(__file__).parent.parent)):digest(p) for p in sorted(paths)})

def main(argv=None):
    parser=argparse.ArgumentParser(description='Frozen offline GSAgent benchmark; no Agent runtime dependency.');sub=parser.add_subparsers(dest='command',required=True)
    for name in ['verify','score']:
        p=sub.add_parser(name);p.add_argument('--benchmark',required=True);p.add_argument('--asset-root')
        if name=='score':p.add_argument('--run-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--clip-index',type=int,default=0);p.add_argument('--allow-partial',action='store_true');p.add_argument('--method-id');p.add_argument('--agent-version')
    p=sub.add_parser('compare');p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='compare':
            from gs_eval.cli import compare
            rows=[read_json(x) for x in a.inputs]
            if len({x['comparability'].get('implementation_fingerprint') for x in rows})!=1:raise ValueError('INCOMPARABLE_EVALUATOR_IMPLEMENTATIONS')
            compare(a.inputs,a.out);return
        b=load_benchmark(a.benchmark,a.asset_root)
        if a.command=='verify':print(json.dumps({'benchmark_id':b['benchmark_id'],'fingerprint':b['_fingerprint'],'assets_verified':len(b['_assets']),'evaluator':__version__}));return
        score(b,a.run_dir,a.out,a.clip_index,a.allow_partial,a.method_id,a.agent_version)
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as e:parser.exit(2,'ERROR: '+str(e)+'\n')

if __name__=='__main__':main()
