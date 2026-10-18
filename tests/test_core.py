import copy,json,subprocess,sys,ast
from pathlib import Path
import numpy as np
import pytest
from gs_eval.config import DEFAULTS,load_config,require_verified,fingerprint
from gs_eval.alignment import camera_for_high,rigid,resize,backproject,fov_signature
from gs_eval.trajectory import select_trajectory,trajectory_diagnostics,unique_calibration,export_colmap
from gs_eval.storage import atomic_json,atomic_npz,read_json,digest,write_jsonl,canonical_hash
from gs_eval.geometry import valid_reference,residual_maps,visible_surface,union_update
from gs_eval.scoring import fixed_reliability,evidence_map,view_utility
from gs_eval.cache import verify_cache

@pytest.fixture
def config(tmp_path):
    c=copy.deepcopy(DEFAULTS);c.update({'schema_version':1,'scene_id':'unit','low_scene_manifest':str(tmp_path/'scene.json'),'high_ply':str(tmp_path/'high.ply'),'calibration_poses':str(tmp_path/'cal.jsonl'),'reference_poses':str(tmp_path/'ref.jsonl'),'cache_root':str(tmp_path/'cache'),'roi':{'min':[-3,-3,0],'max':[3,3,6]},'high_to_low_similarity':{'scale':1.,'rotation':np.eye(3).tolist(),'translation':[0,0,0],'verified':True}});c['surface'].update({'near':.1,'far':5.,'voxel_size':.2,'min_projected_voxel_pixels':.1});c['evaluation_camera'].update({'width':8,'height':8})
    (tmp_path/'low.ply').write_text('PLY-low');(tmp_path/'high.ply').write_text('PLY-high');atomic_json(tmp_path/'scene.json',{'ply':'low.ply'});write_jsonl(tmp_path/'cal.jsonl',[pose(0)]);write_jsonl(tmp_path/'ref.jsonl',[pose(0)]);atomic_json(tmp_path/'config.json',c);return load_config(tmp_path/'config.json')

def pose(frame=0,position=(0,0,0)):
    m=np.eye(4);m[:3,3]=position;return {'frame':frame,'id':str(frame),'camera_to_world':m.tolist(),'intrinsics':{'width':8,'height':8,'fx':4.,'fy':4.,'cx':4.,'cy':4.},'event':{'action':'move','executed_distance':10}}

def render(depth=2.,alpha=1.,rgb=.5):return {'depth':np.full((8,8),depth),'alpha':np.full((8,8),alpha),'rgb':np.full((8,8,3),rgb)}

def test_identity_does_not_verify(config):
    config['high_to_low_similarity']['verified']=False
    with pytest.raises(ValueError,match='ALIGNMENT_UNVERIFIED'):require_verified(config)

@pytest.mark.parametrize('field',['near','far','voxel_size'])
def test_required_explicit_surface(config,field,tmp_path):
    c=read_json(config['_config_path']);del c['surface'][field];atomic_json(tmp_path/'bad.json',c)
    with pytest.raises(ValueError,match='MISSING_EXPLICIT'):load_config(tmp_path/'bad.json')

@pytest.mark.parametrize('r',[np.diag([-1,1,1]),np.diag([2,1,1])])
def test_reject_reflection_scale_in_rotation(r):
    m=np.eye(4);m[:3,:3]=r
    with pytest.raises(ValueError):rigid(m)

def test_similarity_camera_and_depth():
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]]);s=2.;t=np.array([1,2,3]);h=np.eye(4);h[:3,3]=[.2,.3,.4];low=np.eye(4);low[:3,:3]=R@h[:3,:3];low[:3,3]=s*R@h[:3,3]+t;sim={'scale':s,'rotation':R.tolist(),'translation':t.tolist()};assert np.allclose(camera_for_high(low,sim),h);assert np.allclose((np.array([0,0,2])*s),[0,0,4])

def test_resize_and_backproject():
    k=pose()['intrinsics'];big=resize(k,16,24);assert np.allclose(fov_signature(k),fov_signature(big));p=backproject(np.full((8,8),2.),np.eye(4),k);assert np.allclose(p[4,4],[0,0,2])

def test_occlusion_not_just_frustum(config):
    high=render();mask=np.ones((8,8),bool);ids=visible_surface(np.array([[0,0,2],[0,0,3],[0,0,-1]]),np.eye(4),pose()['intrinsics'],high,mask,config);assert ids.tolist()==[0]

@pytest.mark.parametrize('change',[{'near':3.},{'far':1.},{'min_projected_voxel_pixels':100.}])
def test_visibility_bounds_and_scale(config,change):
    config['surface'].update(change);assert len(visible_surface(np.array([[0,0,2]]),np.eye(4),pose()['intrinsics'],render(),np.ones((8,8),bool),config))==0

def test_c2w_not_w2c(config):
    m=np.eye(4);m[0,3]=2.;point=np.array([[2,0,2.]])
    assert len(visible_surface(point,m,pose()['intrinsics'],render(),np.ones((8,8),bool),config))==1
    assert len(visible_surface(point,np.linalg.inv(m),pose()['intrinsics'],render(),np.ones((8,8),bool),config))==0

def test_roi_and_masks(config):
    mask,p=valid_reference(render(),np.eye(4),pose()['intrinsics'],config);assert mask.any();mask2,_=valid_reference(render(),np.eye(4),pose()['intrinsics'],config,np.zeros((8,8),bool));assert not mask2.any();config['roi']={'min':[10,10,10],'max':[11,11,11]};assert not valid_reference(render(),np.eye(4),pose()['intrinsics'],config)[0].any()

def test_low_missing_is_penalized(config):
    high=render();low=render(depth=np.nan,alpha=0,rgb=0);mask,_=valid_reference(high,np.eye(4),pose()['intrinsics'],config);rgb,dep,L=residual_maps(high,low,mask,config);assert np.all(dep[mask]==1);assert np.allclose(rgb[mask],.5);assert np.allclose(L[mask],1.5)

def test_dedup_coverage_auc_and_weights():
    seen=np.zeros(3,bool);w=np.array([1.,2.,0.]);curves=[]
    for ids in [[0],[0],[1]]:
        new,c,b=union_update(seen,ids,w);curves.append(c);assert 0<=c<=1 and 0<=b<=1
    assert np.allclose(curves,[1/3,1/3,2/3]);assert np.isclose(np.mean(curves),4/9);assert union_update(seen,[1],w)[0]==0
    assert union_update(seen,[2],np.zeros(3))[2] is None

def test_unobserved_reliability_and_finite_extremes(config):
    p,g,raw,stats=fixed_reliability([np.array([100.,-100.,2.])],np.array([1,1,0]),config);assert p[2]==.5;assert np.isfinite(g).all();assert stats['positive_clip_fraction']==1/3;assert stats['negative_clip_fraction']==1/3
    q=evidence_map(np.zeros((2,2)),np.ones((2,2),bool),config);assert np.isfinite(q).all()

def test_order_independence(config):
    ds=[np.array([1.,-2.]),np.array([3.,5.]),np.array([-4.,9.])];assert np.allclose(fixed_reliability(ds,np.ones(2),config)[0],fixed_reliability(ds[::-1],np.ones(2),config)[0])

def test_unique_calibration_sorted():
    a,b=pose(1),pose(2);b['camera_to_world']=a['camera_to_world'];out,dupes=unique_calibration([b,a]);assert dupes==1 and out[0]['id']=='1'

def test_utility_unclipped_and_invalid_frame():
    image=np.full((2,2),3.);mask=np.ones((2,2),bool);assert view_utility(image,mask)==(12.,3.);assert view_utility(image,np.zeros_like(mask))==(0.,0.)

@pytest.mark.parametrize('n',[80,81,100])
def test_budget_no_resampling(config,tmp_path,n):
    write_jsonl(tmp_path/'trajectory.jsonl',[pose(i) for i in range(n)])
    if n<81:
        with pytest.raises(ValueError,match='INSUFFICIENT_FRAMES'):select_trajectory(tmp_path,config)
    rows,partial,meta=select_trajectory(tmp_path,config,allow_partial=True);assert len(rows)==min(n,81);assert partial==(n<81);assert [r['frame'] for r in rows]==list(range(min(n,81)))

def test_budget_includes_duplicates_intermediate_frames(config,tmp_path):
    rows=[pose(i) for i in range(81)];rows[1]['event']['action']='move_intermediate';write_jsonl(tmp_path/'trajectory.jsonl',rows);s,_,_=select_trajectory(tmp_path,config);assert len(s)==81;assert trajectory_diagnostics(s,config)['duplicate_pose_ratio']==80/81;assert trajectory_diagnostics(s,config)['sampled_path_length']==0

@pytest.mark.parametrize('action',['gs_set_pose','restore_checkpoint'])
def test_jump_and_teleport(config,action):
    rows=[pose(0),pose(1,(3,0,0))];rows[1]['event']['action']=action;d=trajectory_diagnostics(rows,config);assert not d['continuous_navigation_comparable'];assert d['sampled_path_length']==3

def test_cache_integrity_and_invalidation(config,tmp_path):
    a,_=fingerprint(config);(tmp_path/'high.ply').write_text('changed');b,_=fingerprint(config);assert a!=b;config['surface']['voxel_size']=.3;c,_=fingerprint(config);assert c!=b
    atomic_npz(tmp_path/'data.npz',a=np.arange(4));atomic_json(tmp_path/'m.json',{'files':{'data.npz':digest(tmp_path/'data.npz')}});verify_cache(tmp_path,'m.json');(tmp_path/'data.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='CACHE_CORRUPT'):verify_cache(tmp_path,'m.json')

def test_standalone_has_no_agent_dependency():
    for package in ['gs_eval','gs_agent_eval']:
        for path in Path(package).rglob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom):assert not str(n.module).startswith(('gs_env','gs_mcp'))
                elif isinstance(n,ast.Import):assert not any(a.name.startswith(('gs_env','gs_mcp')) for a in n.names)
    code="import sys; sys.modules['torch']=None; sys.modules['gsplat']=None; sys.modules['gs_env']=None; sys.modules['gs_mcp']=None; import gs_agent_eval.cli,gs_eval.config,gs_eval.geometry,gs_eval.trajectory,gs_eval.scoring; print('standalone CPU ok')"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_export_colmap_all_and_distortion(tmp_path):
    cameras=tmp_path/'cameras.txt';images=tmp_path/'images.txt';sim=tmp_path/'sim.json';cameras.write_text('1 PINHOLE 8 8 4 4 4 4\n');images.write_text('# images\n1 1 0 0 0 0 0 0 1 first.png\n\n2 1 0 0 0 1 0 0 1 second.png\n\n');atomic_json(sim,{'scale':1,'rotation':np.eye(3).tolist(),'translation':[0,0,0],'verified':True});out=tmp_path/'out.jsonl';export_colmap(images,cameras,'all',sim,out);rows=[json.loads(l) for l in out.read_text().splitlines()];assert len(rows)==2;assert rows[1]['camera_to_world'][0][3]==-1
    cameras.write_text('1 OPENCV 8 8 4 4 4 4 0 0 0 0\n')
    with pytest.raises(ValueError,match='UNDISTORT'):export_colmap(images,cameras,'all',sim,out)

def test_optional_trainer_never_runs_by_default(config,tmp_path):
    from gs_eval.cli import main
    before=digest(config['_low_ply']);main(['validate-reconstruction','--config',config['_config_path'],'--run-dir',str(tmp_path),'--out',str(tmp_path/'optional')]);result=read_json(tmp_path/'optional/validation.json');assert result['status']=='skipped';assert result['training_tasks_submitted']==0;assert digest(config['_low_ply'])==before

def test_compare_rejects_different_fingerprints(tmp_path):
    from gs_eval.cli import compare
    for i in range(2):atomic_json(tmp_path/f'{i}.json',{'scene_id':'s','comparability':{'protocol_fingerprint':str(i),'reference_fingerprint':'a','reliability_fingerprint':'b'},'evaluated_frames':81})
    with pytest.raises(ValueError,match='INCOMPARABLE'):compare([tmp_path/'0.json',tmp_path/'1.json'],tmp_path/'comparison')


def test_manifest_change_invalidates_cache(config,tmp_path):
    a,_=fingerprint(config);atomic_json(tmp_path/'scene.json',{'ply':'low.ply','initial_pose':{'position':[1,0,0]}});b,_=fingerprint(config);assert a!=b

def test_reject_existing_run_recorded_in_different_scene(config,tmp_path):
    atomic_json(tmp_path/'scene.json',{'name':'low','ply':'low.ply'});atomic_json(tmp_path/'campaign.json',{'scene':'high'});write_jsonl(tmp_path/'trajectory.jsonl',[pose(i) for i in range(81)])
    with pytest.raises(ValueError,match='RUN_SCENE_MISMATCH'):select_trajectory(tmp_path,config)
