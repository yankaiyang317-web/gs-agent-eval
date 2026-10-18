import argparse,json,sys,time
from pathlib import Path
import numpy as np
from .config import load_config,require_verified
from .storage import atomic_json,read_json,write_jsonl,digest,canonical_hash

def compare(inputs,out):
    from .reporting import csv_rows
    rows=[read_json(p) for p in inputs]
    if not rows:raise ValueError('NO_COMPARISON_INPUTS')
    protocols={(r['scene_id'],r['comparability']['protocol_fingerprint'],r['comparability']['reference_fingerprint'],r['comparability']['reliability_fingerprint'],r['evaluated_frames']) for r in rows}
    if len(protocols)!=1:raise ValueError('INCOMPARABLE_FINGERPRINTS_OR_BUDGETS: compare within one scene/protocol first; cross-scene macro aggregation must be explicit')
    groups={}
    for r in rows:
        key=(r['method_id'],r['comparability']['continuous_navigation_comparable'],r['status'],canonical_hash(r['comparability'].get('start_camera_to_world')));groups.setdefault(key,[]).append(r)
    result=[]
    paths={'surface_coverage_at_81':('A_surface_coverage','primary'),'surface_coverage_auc_81':('A_surface_coverage','supporting'),'frozen_gaussmi_mean':('B_view_value','primary'),'gt_rgb_residual_mean':('B_view_value','observed_reference'),'gt_relative_depth_residual_mean':('B_view_value','observed_reference'),'defect_weighted_coverage_at_81':('B_view_value','supporting')}
    for (method,continuous,status,start_signature),runs in groups.items():
        entry={'method_id':method,'start_signature':start_signature,'continuous_navigation_comparable':continuous,'status':status,'run_count':len(runs),'A_surface_coverage':{},'B_view_value':{}}
        for name,(family,section) in paths.items():
            v=[r['metric_families'][family][section][name] for r in runs];v=[x for x in v if x is not None];entry[family][name]={'mean':float(np.mean(v)) if v else None,'std':float(np.std(v,ddof=1)) if len(v)>1 else None,'valid_runs':len(v)}
        result.append(entry)
    out=Path(out);atomic_json(out/'comparison.json',{'groups':result,'ranking':None,'single_run_note':'single-run groups have no variance estimate','aggregation':'within_scene_only; no A/B combined metric'})
    for family in ['A_surface_coverage','B_view_value']:
        csv_rows(out/(family+'.csv'),[{'method':r['method_id'],'status':r['status'],'continuous':r['continuous_navigation_comparable'],'runs':r['run_count'],**{k:v['mean'] for k,v in r[family].items()}} for r in result])
    (out/'report.md').write_text('# 离线评价对比\n\nA 探索覆盖与 B 视角价值分开见两个 CSV，不产生统一排名。每项报告均值、样本标准差和有效 run 数；单次运行不估计方差。不同行走连续性或状态分组。\n',encoding='utf-8');return result

def validate(c,out):
    from .renderer import EvalRenderer,environment_identity
    from .trajectory import read_poses
    from .cache import explicit_mask
    from .geometry import valid_reference
    from .reporting import image_pair,heatmap
    from .alignment import resize
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ref=read_poses(c['reference_poses']);cal=read_poses(c['calibration_poses']);r=EvalRenderer(c);examples=[]
    # Alignment diagnostics only: scores are never published while unverified.
    chosen=ref[:min(3,len(ref))];high=[]
    for p in chosen:high.append(r.render(p,'high')[0])
    for i,p in enumerate(chosen):
        low,k=r.render(p,'low');image_pair(out/f'alignment_{i:02d}.png',high[i],low);mask,_=valid_reference(high[i],p['camera_to_world'],k,c,explicit_mask(p,c,k));heatmap(out/f'depth_difference_{i:02d}.png',np.abs(high[i]['depth']-low['depth']),mask,'Expected depth difference, low units');examples.append({'pose_id':p['id'],'valid_high_fraction':float(mask.mean())})
    result={'status':'alignment_verified_by_operator' if c['high_to_low_similarity'].get('verified') else 'alignment_unverified_diagnostics_only','reference_poses':len(ref),'calibration_poses':len(cal),'versions':environment_identity(),'examples':examples,'contribution_backend':'must_run_scripts/smoke/test_eval_gpu.py','missing_assets':[]};atomic_json(out/'validation.json',result);return result

def export_reference(c,run_dir,out,partial=False):
    from .trajectory import select_trajectory
    from .renderer import EvalRenderer
    from PIL import Image
    require_verified(c);poses,ispartial,meta=select_trajectory(run_dir,c,partial);out=Path(out);(out/'images').mkdir(parents=True,exist_ok=True);r=EvalRenderer(c);records=[]
    for i,p in enumerate(poses):
        high,k=r.render(p,'high');name=f'images/{i:05d}.png';Image.fromarray(np.round(high['rgb']*255).astype('uint8')).save(out/name);records.append({**p,'intrinsics':k,'reference_image':name})
    write_jsonl(out/'poses.jsonl',records);atomic_json(out/'export.json',{'frames':len(records),'partial':ispartial,'pseudo_gt':c['high_ply'],'trajectory_sha256':meta['trajectory_sha256'],'training_submitted':False});return out

def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline evaluation only: separate A coverage and B fixed-snapshot utility/residuals.');sub=parser.add_subparsers(dest='command',required=True)
    for name in ['validate','prepare-reference','prepare-reliability','evaluate','export-reference-frames','validate-reconstruction']:
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name in ['evaluate','validate','export-reference-frames','validate-reconstruction']:p.add_argument('--out',required=True)
        if name in ['evaluate','export-reference-frames','validate-reconstruction']:p.add_argument('--run-dir',required=True);p.add_argument('--allow-partial',action='store_true')
        if name=='evaluate':p.add_argument('--coverage-only',action='store_true');p.add_argument('--skip-prediction',action='store_true');p.add_argument('--method-id')
    p=sub.add_parser('compare');p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('export-poses');p.add_argument('--images-txt',required=True);p.add_argument('--cameras-txt',required=True);p.add_argument('--selection',required=True);p.add_argument('--world-to-low-similarity',required=True);p.add_argument('--out',required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='compare':compare(a.inputs,a.out);return
        if a.command=='export-poses':
            from .trajectory import export_colmap
            export_colmap(a.images_txt,a.cameras_txt,a.selection,a.world_to_low_similarity,a.out);return
        c=load_config(a.config)
        if a.command=='validate':print(json.dumps(validate(c,a.out),ensure_ascii=False));return
        if a.command=='prepare-reference':
            from .cache import prepare_reference
            prepare_reference(c);return
        if a.command=='prepare-reliability':
            from .cache import prepare_reliability
            prepare_reliability(c);return
        if a.command=='evaluate':
            from .evaluation import evaluate
            evaluate(c,a.run_dir,a.out,a.allow_partial,a.coverage_only,a.skip_prediction,a.method_id);return
        if a.command=='export-reference-frames':export_reference(c,a.run_dir,a.out,a.allow_partial);return
        if a.command=='validate-reconstruction':
            # Explicit boundary: no arbitrary trainer commands and no hidden optimization.
            enabled=c['reconstruction_validation']['enabled'];atomic_json(Path(a.out)/'validation.json',{'status':'skipped','reason':'reconstruction_validation_disabled' if not enabled else 'external_trainer_adapter_not_configured','training_tasks_submitted':0,'input_models_modified':False});return
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as e:
        if a.command=='validate':atomic_json(Path(a.out)/'validation.json',{'status':'failed','error':str(e),'missing_assets_or_config':[str(e)]})
        parser.exit(2,'ERROR: '+str(e)+'\n')

