import copy,json
from pathlib import Path
import numpy as np
import pytest
from gs_agent_eval.cli import load_benchmark,select_run
from gs_eval.storage import atomic_json,write_jsonl,digest

@pytest.fixture
def bench(tmp_path):
    f=tmp_path/'data.bin';f.write_bytes(b'fixed')
    b={'schema_version':1,'benchmark_id':'unit','scene_name':'unit_scene','budget':{'frames':81},'locations':{'asset_root':str(tmp_path)},'assets':{'x':{'path':'${asset_root}/data.bin','sha256':digest(f)}},'camera_fov_signature':[.5,.5,.5,.5],'coordinates':{'world_up':[0,1,0]},'navigation':{'collision_sha256':'mesh','camera_radius':.25,'camera_body_height':.9,'collision_step':.02}}
    atomic_json(tmp_path/'b.json',b);return load_benchmark(tmp_path/'b.json')

def rows(n):
    return [{'frame':i,'camera_to_world':np.eye(4).tolist(),'intrinsics':{'width':8,'height':8,'fx':4,'fy':4,'cx':4,'cy':4}} for i in range(n)]

def run(tmp_path,n=243):
    folder=tmp_path/'run';write_jsonl(folder/'trajectory.jsonl',rows(n));atomic_json(folder/'campaign.json',{'scene':'unit_scene','world_up':[0,1,0]});return folder

def test_relocation_preserves_protocol_identity(bench,tmp_path):
    other=tmp_path/'elsewhere';other.mkdir();(other/'data.bin').write_bytes(b'fixed')
    assert load_benchmark(tmp_path/'b.json',other)['_fingerprint']==bench['_fingerprint']

def test_state_tampering_is_rejected(bench,tmp_path):
    (tmp_path/'data.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='HASH_MISMATCH'):load_benchmark(tmp_path/'b.json')

def test_frame_budget_changes_protocol(bench,tmp_path):
    b=json.loads((tmp_path/'b.json').read_text());b['budget']['frames']=82;atomic_json(tmp_path/'bad.json',b)
    with pytest.raises(ValueError,match='BUDGET'):load_benchmark(tmp_path/'bad.json')

def test_exact_clip_segments_no_padding(bench,tmp_path):
    folder=run(tmp_path);poses,partial,meta=select_run(folder,bench,2);assert not partial and [r['frame'] for r in poses]==list(range(162,243));assert not meta['navigation_protocol_verified_from_metadata']

def test_partial_has_no_fake_81_frames(bench,tmp_path):
    folder=run(tmp_path,80)
    with pytest.raises(ValueError,match='INSUFFICIENT'):select_run(folder,bench)
    poses,partial,_=select_run(folder,bench,allow_partial=True);assert len(poses)==80 and partial

def test_wrong_scene_and_fov_rejected(bench,tmp_path):
    folder=run(tmp_path);atomic_json(folder/'campaign.json',{'scene':'other'})
    with pytest.raises(ValueError,match='SCENE'):select_run(folder,bench)
    atomic_json(folder/'campaign.json',{'scene':'unit_scene'});a=rows(81);a[0]['intrinsics']['fx']=8;write_jsonl(folder/'trajectory.jsonl',a)
    with pytest.raises(ValueError,match='FOV'):select_run(folder,bench)

def test_navigation_metadata_checked_when_present(bench,tmp_path):
    folder=run(tmp_path);atomic_json(folder/'run_metadata.json',{'environment':bench['navigation']});assert select_run(folder,bench)[2]['navigation_protocol_verified_from_metadata']
    atomic_json(folder/'run_metadata.json',{'environment':{**bench['navigation'],'camera_radius':.1}})
    with pytest.raises(ValueError,match='NAVIGATION_PROTOCOL'):select_run(folder,bench)
