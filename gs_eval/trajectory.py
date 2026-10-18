import json
from pathlib import Path
import numpy as np
from .alignment import rigid,intrinsics,fov_signature
from .storage import write_jsonl,read_json,canonical_hash,digest

def read_poses(path):
    rows=[];ids=set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            a=json.loads(line);a['camera_to_world']=rigid(a['camera_to_world']).tolist();a['intrinsics']=intrinsics(a['intrinsics']);a.setdefault('id',str(a.get('frame',len(rows))))
            if str(a['id']) in ids:raise ValueError('DUPLICATE_POSE_ID: '+str(a['id']))
            ids.add(str(a['id']));rows.append(a)
    if not rows:raise ValueError('EMPTY_POSE_FILE: '+str(path))
    return rows

def unique_calibration(rows):
    out=[];seen=set()
    for r in sorted(rows,key=lambda x:str(x['id'])):
        h=canonical_hash([r['camera_to_world'],r['intrinsics']])
        if h not in seen:out.append(r);seen.add(h)
    return out,len(rows)-len(out)

def select_trajectory(run_dir,c,allow_partial=False):
    path=Path(run_dir)/'trajectory.jsonl';rows=read_poses(path)
    ids=[r.get('frame') for r in rows]
    if any(not isinstance(i,int) for i in ids) or any(b<=a for a,b in zip(ids,ids[1:])):raise ValueError('INVALID_RECORDED_FRAME_ORDER')
    start=c['budget']['start_frame'];selected=[r for r in rows if start<=r['frame']<start+81]
    if not selected or [r['frame'] for r in selected]!=list(range(start,start+len(selected))):raise ValueError('NONCONTIGUOUS_RECORDED_FRAMES')
    partial=len(selected)<81
    if partial and not (allow_partial or c['budget']['allow_partial']):raise ValueError(f'INSUFFICIENT_FRAMES: {len(selected)}/81')
    sig=fov_signature(selected[0]['intrinsics'])
    if any(not np.allclose(sig,fov_signature(r['intrinsics']),atol=1e-6) for r in selected):raise ValueError('MIXED_FOV')
    meta=read_json(Path(run_dir)/'run_metadata.json') if (Path(run_dir)/'run_metadata.json').exists() else {}
    campaign=read_json(Path(run_dir)/'campaign.json') if (Path(run_dir)/'campaign.json').exists() else {}
    scene=read_json(c['low_scene_manifest'])
    if campaign.get('scene') and scene.get('name') and campaign['scene']!=scene['name'] and 'synthetic' not in meta.get('trajectory_origin',''):
        raise ValueError('RUN_SCENE_MISMATCH: campaign was not recorded in configured low scene')
    meta['campaign_metadata']={k:campaign.get(k) for k in ['scene','exploration_objective','video_segment_frames','video_fps']}
    return selected,partial,{'trajectory_sha256':digest(path),'fov_signature':sig.tolist(),**meta}

def trajectory_diagnostics(rows,c):
    positions=np.array([r['camera_to_world'] for r in rows])[:,:3,3];length=0.;angle=0.;jumps=[];seen=set();duplicates=0;events=[]
    for i,r in enumerate(rows):
        h=canonical_hash([r['camera_to_world'],r['intrinsics']]);duplicates+=h in seen;seen.add(h);jump=False;event=r.get('event',{});action=str(event.get('action',event.get('tool','')))
        if 'set_pose' in action or 'restore_checkpoint' in action:jump=True;events.append({'frame':r['frame'],'action':action})
        if i:
            d=float(np.linalg.norm(positions[i]-positions[i-1]));ra=np.array(rows[i-1]['camera_to_world'])[:3,:3];rb=np.array(r['camera_to_world'])[:3,:3];a=float(np.degrees(np.arccos(np.clip((np.trace(ra.T@rb)-1)/2,-1,1))));length+=d;angle+=a;jump|=d>c['trajectory_checks']['translation_jump'] or a>c['trajectory_checks']['rotation_jump_deg']
        if jump:jumps.append(r['frame'])
    return {'sampled_path_length':length,'total_rotation_deg':angle,'duplicate_pose_ratio':duplicates/len(rows),'jump_frames':jumps,'teleport_events':events,'continuous_navigation_comparable':not jumps,'collision_action_count':None}

def export_colmap(images_path,cameras_path,selection_path,sim_path,out):
    from .alignment import similarity
    sim=read_json(sim_path)
    if sim.get('verified') is not True:raise ValueError('COLMAP_ALIGNMENT_UNVERIFIED')
    s,r,t=similarity(sim);cameras={}
    for line in Path(cameras_path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=line.split();cid=int(p[0]);model=p[1];w,h=map(int,p[2:4]);v=list(map(float,p[4:]))
        if model=='PINHOLE' and len(v)==4:fx,fy,cx,cy=v
        elif model=='SIMPLE_PINHOLE' and len(v)==3:fx,cx,cy=v;fy=fx
        else:raise ValueError('UNDISTORT_PINHOLE_REQUIRED: '+model)
        cameras[cid]={'fx':fx,'fy':fy,'cx':cx,'cy':cy,'width':w,'height':h}
    selected=None if selection_path=='all' else set(read_json(selection_path));lines=Path(images_path).read_text().splitlines();outrows=[];i=0
    from ._vendor.transforms import quaternion_to_rotation_matrix
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip() or line.startswith('#'):continue
        p=line.split(maxsplit=9)
        if len(p)!=10:raise ValueError('INVALID_COLMAP_IMAGE_LINE')
        i+=1 # the following observations line may be empty
        name=p[9]
        if selected is not None and name not in selected:continue
        q=np.array(list(map(float,p[1:5])));q/=np.linalg.norm(q);rcw=quaternion_to_rotation_matrix(q);tcw=np.array(list(map(float,p[5:8])));m=np.eye(4);m[:3,:3]=r@rcw.T;m[:3,3]=s*(r@(-rcw.T@tcw))+t
        outrows.append({'id':p[0],'camera_to_world':m.tolist(),'intrinsics':cameras[int(p[8])],'source_image':name})
    if selected is not None and {r['source_image'] for r in outrows}!=selected:raise ValueError('COLMAP_SELECTION_MISSING_IMAGES')
    if not outrows:raise ValueError('EMPTY_COLMAP_SELECTION')
    write_jsonl(out,outrows)

