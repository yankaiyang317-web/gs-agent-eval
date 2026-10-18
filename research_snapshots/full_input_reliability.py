"""Full actual training-input calibration; independent of legacy frozen pilot caches."""
from pathlib import Path
import json,csv,time,copy
import numpy as np
from PIL import Image,ImageDraw
from gs_eval.storage import atomic_json,atomic_npz,read_json,digest,canonical_hash
from gs_eval.config import load_config
from gs_eval.renderer import EvalRenderer
from gs_eval.cache import read_render
from gs_eval.trajectory import read_poses
from gs_eval.scoring import fixed_reliability,view_utility
from gs_eval.geometry import valid_reference
from gs_eval.alignment import resize

def depth_normals(depth,valid,k,relative_edge=.05):
    """Camera-space depth normals, oriented toward camera, 5-point valid stencil."""
    d=np.asarray(depth,dtype=float);h,w=d.shape;y,x=np.mgrid[:h,:w]
    xyz=np.stack([(x-k['cx'])*d/k['fx'],(y-k['cy'])*d/k['fy'],d],axis=-1)
    n=np.zeros((h,w,3),dtype=float);ok=np.zeros((h,w),bool)
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1];cross=np.cross(dx,dy);length=np.linalg.norm(cross,axis=-1)
    good=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&(length>1e-10)
    for v in [d[1:-1,2:],d[1:-1,:-2],d[2:,1:-1],d[:-2,1:-1]]:good&=np.abs(v-d[1:-1,1:-1])<=relative_edge*np.maximum(d[1:-1,1:-1],.1)
    cross/=np.maximum(length[...,None],1e-12);cross*=np.where(np.sum(cross*xyz[1:-1,1:-1],axis=-1)>0,-1.,1.)[...,None]
    n[1:-1,1:-1]=np.where(good[...,None],cross,0);ok[1:-1,1:-1]=good
    return n.astype('float32'),ok

def self_check():
    k={'fx':20,'fy':20,'cx':8,'cy':6};d=np.full((13,17),2.);valid=np.ones(d.shape,bool)
    n,m=depth_normals(d,valid,k);assert m.sum()==11*15 and np.max(np.abs(n[m]-[0,0,-1]))<1e-6
    n2,m2=depth_normals(d*3,valid,k);assert np.array_equal(m,m2) and np.allclose(n,n2)
    valid[6,8]=False;_,m=depth_normals(d,valid,k);assert not m[6,8] and not m[6,7] and not m[5,8]
    d[:,9:]=4;_,m=depth_normals(d,np.ones(d.shape,bool),k);assert not m[6,8] and not m[6,9]

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    self_check();root=Path('/app');assets=Path('/machine25_data1/gs-agent/outputs/jiudian_synthetic_20261015');out=root/'outputs/evaluation/full_input_reliability_20261016';out.mkdir(parents=True,exist_ok=True)
    oldc=load_config(root/'eval_configs/jiudian_pilot_v1.json');oldcache=Path(read_json(root/'outputs/evaluation/pilot_validation.json')['reference_cache'])
    original={str(p):digest(p) for p in [Path(oldc['_low_ply']),Path(oldc['high_ply']),oldcache/'reliability.npz',oldcache/'reference_surface.npz',oldcache/'defect_weights.npz']}
    train=read_json(assets/'training.json');meta=read_json(assets/'transforms.json');frames={f['file_path']:f for f in meta['frames']};names=train['train_frames'];assert len(set(names))==len(names) and not set(names)&set(train['held_out_frames'])
    c=copy.deepcopy(oldc);c['evaluation_camera'].update(width=480,height=360)
    k0={'width':meta['w'],'height':meta['h'],'fx':meta['fl_x'],'fy':meta['fl_y'],'cx':meta['cx'],'cy':meta['cy']};k=resize(k0,480,360)
    inputs=[]
    for name in names:
        f=frames[name];rgb=assets/name;depth=assets/'depth'/Path(name).name.replace('.png','.npz')
        if not rgb.is_file() or not depth.is_file():raise FileNotFoundError(name)
        inputs.append({'id':name,'camera_to_world':f['transform_matrix'],'intrinsics':k0,'rgb':str(rgb),'rgb_sha256':digest(rgb),'depth':str(depth),'depth_sha256':digest(depth)})
    posesig=[canonical_hash([p['camera_to_world'],p['intrinsics']]) for p in inputs];assert len(set(posesig))==len(inputs),'Duplicate input poses need explicit treatment'
    protocol={'method':'full_training_inputs_scalar_logodds_rgbdn_v1','scene':'jiudian_degraded','calibration_frames':len(inputs),'excluded_held_out_frames':len(train['held_out_frames']),'source':'actual synthetic RGB and depth used for the original degraded-model reconstruction; no newly sampled calibration cameras','normal_source':'derived from saved expected depth with camera-facing five-point cross-product stencil; no archived normal files exist','normal_edge_relative_threshold':.05,'calibration_resolution':[480,360],'score_resolution':[320,240],'rgb_resize':'original 1920x1440 PNG downsampled Lanczos, preserving FOV','depth_definition':'saved accumulated D divided by saved alpha at native 480x360','reference_valid':'saved alpha>=0.9; finite expected depth 0.1..12; all training views, no reachability/ROI filtering','low_depth_valid':'alpha>=0.7; finite positive depth','loss':'mean RGB L1 + clipped relative depth error + 0.15*(1-dot(depth normals)); normal missing on low but valid on reference receives penalty 1','normal_weight':.15,'lambda_loss':10.,'lambda_evidence':.01,'logodds_clip':8.,'primary_state':'RGBDN, unchanged evidence strength and loss threshold; full-input RGBD ablation saved separately','state_scope':'per-Gaussian scalar; no front/back states or online updates; probabilities are evidence-model states, not calibrated correctness frequencies','score':'legacy fixed-snapshot masked feature rendering and per-frame mean, using one fixed state for all trajectories','unobserved':'support<=1e-10 remains p=.5 and is flagged separately','code_sha256':digest(Path(__file__)),'source_metadata':{str(assets/n):digest(assets/n) for n in ['training.json','transforms.json','defects.json']},'inputs':inputs,'versions':{'renderer':'gsplat N-D feature gradient','normal_tests':'plane orientation, scale invariance, invalid stencil, discontinuity passed'}}
    protocol['fingerprint']=canonical_hash(protocol);atomic_json(out/'protocol.json',protocol)
    renderer=EvalRenderer(c);scene=renderer.load('low');N=scene.count
    accum=np.zeros((N,7),dtype=np.float64);viewstats=[];start=time.perf_counter()
    for i,p in enumerate(inputs):
        rgb=np.array(Image.open(p['rgb']).convert('RGB').resize((480,360),Image.Resampling.LANCZOS),dtype='float32')/255
        with np.load(p['depth']) as a:d=a['depth'].astype(float)/np.maximum(a['alpha'],1e-6);alpha=a['alpha'].copy()
        assert d.shape==(360,480)
        mask=np.isfinite(d)&(d>=.1)&(d<=12)&(alpha>=.9)
        low,_=renderer.render(p,'low');validlow=np.isfinite(low['depth'])&(low['depth']>0)&(low['alpha']>=.7)
        rn,rvalid=depth_normals(d,mask,k);ln,lvalid=depth_normals(low['depth'],validlow,k)
        rgb_error=np.mean(np.abs(rgb-low['rgb']),axis=-1);depth_error=np.ones(d.shape);np.divide(np.abs(low['depth']-d),np.maximum(d,.1),out=depth_error,where=validlow);depth_error=np.where(validlow,np.clip(depth_error,0,1),1.)
        normal_error=np.where(lvalid,1-np.clip(np.sum(rn*ln,axis=-1),-1,1),1.);normal_error=np.where(rvalid,normal_error,0.)
        loss=rgb_error+depth_error;lossn=loss+.15*normal_error
        q=-np.log(np.maximum(10*loss,1e-4))*mask;qn=-np.log(np.maximum(10*lossn,1e-4))*mask
        maps=[q,qn,mask.astype(float),(mask&rvalid).astype(float),mask*rgb_error,mask*depth_error,mask*normal_error]
        grad=renderer.contribution(p,maps);assert np.isfinite(grad).all();accum+=grad
        # Feature linearity / per-Gaussian attribution: contribution sums equal alpha-integrated maps.
        if i in (0,len(inputs)-1):
            for j in [0,1,2]:assert np.isclose(float(grad[:,j].sum()),float((low['alpha']*maps[j]).sum()),rtol=5e-4,atol=.1),(i,j)
        viewstats.append({'input':p['id'],'valid_reference_fraction':float(mask.mean()),'normal_reference_fraction':float(rvalid.mean()),'rgb_error':float(rgb_error[mask].mean()) if mask.any() else None,'relative_depth_error':float(depth_error[mask].mean()) if mask.any() else None,'normal_error':float(normal_error[rvalid].mean()) if rvalid.any() else None,'reference_missing_low_fraction':float((mask&~validlow).sum()/max(mask.sum(),1))})
        if i in [0,len(inputs)//2,len(inputs)-1]:
            canvas=Image.new('RGB',(480*4,385));draw=ImageDraw.Draw(canvas)
            for j,img in enumerate([rgb,low['rgb'],(rn+1)/2,(ln+1)/2]):canvas.paste(Image.fromarray(np.uint8(np.clip(img,0,1)*255)),(j*480,0))
            draw.text((8,365),f'{p["id"]}: input RGB | low RGB | input depth-normal | low depth-normal',fill='white');canvas.save(out/f'input_check_{i:03d}.jpg')
        if (i+1)%16==0:print('CALIBRATION',i+1,len(inputs),round(time.perf_counter()-start,1),flush=True)
    states={};diagnostics={}
    for name,col in [('RGBD',0),('RGBDN',1)]:
        raw=.01*accum[:,col];p,g,_,stats=fixed_reliability([raw],accum[:,2],c);stats.update({'near_half_fraction':float(np.mean(np.abs(p-.5)<=.05)),'low_reliability_fraction':float(np.mean(p<.25)),'high_reliability_fraction':float(np.mean(p>.75)),'weak_support_below_one_fraction':float(np.mean(accum[:,2]<1))})
        atomic_npz(out/f'reliability_{name}.npz',p=p,g=g,raw_logodds=raw,support=accum[:,2],normal_support=accum[:,3],rgb_error_sum=accum[:,4],depth_error_sum=accum[:,5],normal_error_sum=accum[:,6]);states[name]=(p,g);diagnostics[name]=stats
    atomic_json(out/'calibration_summary.json',{'fingerprint':protocol['fingerprint'],'calibration_frames':len(inputs),'gaussians':N,'diagnostics':diagnostics,'views':viewstats,'elapsed_s':time.perf_counter()-start,'normal_self_checks_passed':True,'gaussian_attribution_mass_checks_passed':True})
    # Score exactly the previously recorded poses at unchanged evaluation resolution.
    renderer.c=oldc;nav=root/'outputs/evaluation/three_clip_run_20261016';scores=[];perframe=[]
    with np.load(oldcache/'reliability.npz') as a:oldg=a['g'].copy()
    allg=np.column_stack([oldg,states['RGBD'][1],states['RGBDN'][1],np.full(N,np.log(2)),(accum[:,2]<=1e-10).astype(float)])
    for clip in range(3):
        poses=read_poses(nav/'inputs'/f'clip_{clip:03d}'/'trajectory.jsonl');records=[]
        for j,p in enumerate(poses):
            high=read_render(nav/f'clip_{clip:03d}'/'diagnostics/high_frames'/f'{j:05d}.npz');kk=resize(p['intrinsics'],320,240);mask,_=valid_reference(high,p['camera_to_world'],kk,oldc);feat=renderer.feature(p,allg)
            row={'clip':clip+1,'campaign_frame':clip*81+j,'old_B':view_utility(feat[...,0],mask)[1],'full_input_RGBD_B':view_utility(feat[...,1],mask)[1],'full_input_RGBDN_B':view_utility(feat[...,2],mask)[1],'constant_p05_B':view_utility(feat[...,3],mask)[1],'unknown_state_render_mass':view_utility(feat[...,4],mask)[1]};records.append(row);perframe.append(row)
        old=read_json(nav/f'clip_{clip:03d}'/'summary.json');expected=old['metric_families']['B_view_value']['primary']['frozen_gaussmi_mean'];assert np.isclose(np.mean([r['old_B'] for r in records]),expected,atol=1e-6)
        scores.append({'clip':clip+1,'frames':81,**{key:float(np.mean([r[key] for r in records])) for key in records[0] if key not in ('clip','campaign_frame')},'A_surface_coverage_unchanged':old['metric_families']['A_surface_coverage']['primary']['surface_coverage_at_81']})
        print('SCORED',scores[-1],flush=True)
    with (out/'per_frame_scores.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=perframe[0]);w.writeheader();w.writerows(perframe)
    assert all(digest(p)==h for p,h in original.items())
    base=read_json(root/'BASELINE_SNAPSHOT.json');assert all(digest(Path(base['source'])/k)==h for k,h in base['files'].items())
    from gs_eval.evaluation import rank_corr
    comparison={'old_constant_spearman':rank_corr([r['old_B'] for r in perframe],[r['constant_p05_B'] for r in perframe]),'full_RGBDN_constant_spearman':rank_corr([r['full_input_RGBDN_B'] for r in perframe],[r['constant_p05_B'] for r in perframe]),'RGBD_RGBDN_spearman':rank_corr([r['full_input_RGBD_B'] for r in perframe],[r['full_input_RGBDN_B'] for r in perframe])}
    atomic_json(out/'scores.json',{'fingerprint':protocol['fingerprint'],'primary':'full_input_RGBDN_B','scores':scores,'diagnostics':comparison,'source_assets_and_legacy_cache_unchanged':True,'original_code_files_unchanged':len(base['files']),'state_hashes':{n:digest(out/f'reliability_{n}.npz') for n in states},'interpretation':'Actual full training-input evidence, fixed-snapshot per-frame utility; not realized repair gain. New calibration protocol, old numeric scores shown as diagnostics only.'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    with np.load(oldcache/'reliability.npz') as a:oldp=a['p'].copy()
    for label,p in [('old 24 views',oldp),('full inputs RGBD',states['RGBD'][0]),('full inputs RGBD+normal',states['RGBDN'][0])]:axes[0].hist(p,bins=np.linspace(0,1,51),histtype='step',label=label)
    axes[0].set(xlabel='Reliability state p',ylabel='Gaussian count (log)',yscale='log',title='Same model; different calibration evidence');axes[0].legend()
    for key,label in [('old_B','old'),('full_input_RGBD_B','full RGBD'),('full_input_RGBDN_B','full RGBD+normal')]:axes[1].plot([r['campaign_frame'] for r in perframe],[r[key] for r in perframe],label=label)
    axes[1].set(xlabel='Campaign frame',ylabel='Frozen view utility',title='Same 243 recorded poses');axes[1].legend();fig.tight_layout();fig.savefig(out/'reliability_and_scores.png',dpi=160)
    lines=['# jiudian_degraded 完整重建输入可靠性与重评分','',f'使用 training.json 中全部 {len(inputs)} 张实际训练 RGB 和对应深度、内外参；排除 {len(train["held_out_frames"])} 张留出图。没有按碰撞可达性筛掉训练输入。', '',
    '输入是之前从高质量 PLY 合成的重建数据，不是原始实拍照片。没有归档 normal map；参考和 low 的法线均从各自有效期望深度计算，屏蔽深度跳变及无效邻域。', '',
    '按 Gaussian 的实际 alpha 合成贡献回传 RGB、深度和法线误差，逐视角累积 log-odds，再转换为 p。没有将整图平均误差赋给所有 Gaussian。没有证据的 Gaussian 单独标记且保留 p=.5。', '',
    '主结果使用完整输入 RGBD+normal。另保存完整输入 RGBD 消融。证据强度 .01、loss 倍率 10、clip 8 均保持旧值，没有根据待评分轨迹调参；normal 项权重 .15。状态建立分辨率为原始深度的 480×360，评分仍为 320×240。', '',
    '|状态|无证据比例|p 在 .45–.55 的比例|p<.25|p>.75|', '|---|---:|---:|---:|---:|']
    for name,stats in diagnostics.items():lines.append(f'|完整输入 {name}|{stats["no_evidence_fraction"]:.2%}|{stats["near_half_fraction"]:.2%}|{stats["low_reliability_fraction"]:.2%}|{stats["high_reliability_fraction"]:.2%}|')
    lines+=['','|片段|旧 B，24视角|完整 RGBD B|完整 RGBD+normal B（主项）|A，保持不变|','|---|---:|---:|---:|---:|']
    for s in scores:lines.append(f'|{s["clip"]}|{s["old_B"]:.6f}|{s["full_input_RGBD_B"]:.6f}|{s["full_input_RGBDN_B"]:.6f}|{s["A_surface_coverage_unchanged"]:.2%}|')
    lines+=['',f'与常量 p=.5 对照的逐帧 Spearman：旧状态 {comparison["old_constant_spearman"]:.4f}，完整 RGBD+normal 状态 {comparison["full_RGBDN_constant_spearman"]:.4f}。相邻帧相关，此处仅作描述性诊断。', '',
    '新旧 B 数值来自不同校准协议，分数上升不代表轨迹改善。p 仍是指定证据模型下的可靠性状态，不是经外部标定的真实正确概率；仍保留普通3DGS单标量、固定状态、无正反面方向权重的限制。纯缺失、没有low Gaussian贡献的区域不能分配到一个不存在的Gaussian，仍需参考缺陷覆盖补充。', '',
    '这次没有重新训练或修改场景；所有原 PLY、旧缓存和原项目98个基线文件哈希未变。state_hashes 和全部输入文件 SHA256 见 scores.json / protocol.json。', '',
    '校验：平面法线方向、尺度不变性、无效深度邻域、深度不连续屏蔽均通过；首尾输入的逐Gaussian归因总量与alpha积分一致；旧状态回放得分与历史报告一致。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('FULL_INPUT_COMPLETE',json.dumps({'diagnostics':diagnostics,'scores':scores,'comparison':comparison}),flush=True)

if __name__=='__main__':main()
