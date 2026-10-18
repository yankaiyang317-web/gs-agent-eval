"""GT RGB-D augmentation pilot: add geometry, replay old views, fixed held-out tests."""
from pathlib import Path
import json,time,csv,math,gc,copy
import numpy as np
import torch
from plyfile import PlyData,PlyElement
from PIL import Image,ImageDraw
from gsplat import rasterization
from gs_env.rendering.ply_loader import load_gaussian_ply
from gs_eval.storage import atomic_json,atomic_npz,read_json,digest,canonical_hash
from gs_eval.config import load_config
from gs_eval.renderer import EvalRenderer
from gs_eval.cache import read_render
from gs_eval.trajectory import read_poses
from gs_eval.geometry import valid_reference
from gs_eval.alignment import resize

def main():
    root=Path('/app');out=root/'outputs/evaluation/gt_augmentation_validation_20261016';out.mkdir(parents=True,exist_ok=True);nav=root/'outputs/evaluation/three_clip_run_20261016';assets=Path('/machine25_data1/gs-agent/outputs/jiudian_synthetic_20261015');c=load_config(root/'eval_configs/jiudian_pilot_v1.json');train=read_json(assets/'training.json');meta=read_json(assets/'transforms.json')
    srcfiles={str(p):digest(p) for p in [Path(c['_low_ply']),Path(c['high_ply']),assets/'training.json',assets/'transforms.json']};fullstate=root/'outputs/evaluation/full_input_reliability_20261016';statehash=digest(fullstate/'reliability_RGBDN.npz')
    k0={'width':meta['w'],'height':meta['h'],'fx':meta['fl_x'],'fy':meta['fl_y'],'cx':meta['cx'],'cy':meta['cy']};k=resize(k0,320,240)
    frames={f['file_path']:{'id':f['file_path'],'camera_to_world':f['transform_matrix'],'intrinsics':k0} for f in meta['frames']}
    oldposes=[frames[n] for n in train['train_frames']];allnew=[];groups={}
    for i in range(3):
        ps=read_poses(nav/'inputs'/f'clip_{i:03d}'/'trajectory.jsonl');allnew+=ps;groups[f'clip_{i+1}']=[(p,read_render(nav/f'clip_{i:03d}'/'diagnostics/high_frames'/f'{j:05d}.npz')) for j,p in enumerate(ps)]
    groups['repeat_control']=[groups['clip_1'][0]]*81
    heldposes=[]
    for name in train['held_out_frames']:
        p=frames[name];m=np.array(p['camera_to_world']);overlap=False
        for q in allnew:
            n=np.array(q['camera_to_world']);dist=np.linalg.norm(m[:3,3]-n[:3,3]);angle=np.degrees(np.arccos(np.clip((np.trace(m[:3,:3].T@n[:3,:3])-1)/2,-1,1)))
            if dist<.4 and angle<20:overlap=True;break
        if not overlap:heldposes.append(p)
    assert len(heldposes)>=8
    settings={'method':'GT_RGBD_seed_insertion_and_replay_pilot_v1','scene':'jiudian_degraded','steps':1000,'seeds':[253,719],'resolution':[320,240],'old_frames':len(oldposes),'new_slots':81,'groups':['replay_only',*groups],'old_frame_replay_probability':.5,'replay_only_sampling':'100% old frames','old_seed_max':20000,'new_seed_max':20000,'seed_voxel':.04,'seed_selection':'valid high alpha>.9, depth .35..12, ROI, stride3, low alpha<.7 OR relative depth error>.08 OR RGB error>.08; unique voxel; fixed random subsample; new seeds exclude shared old-seed voxels','geometry_update':'Same initial low model plus common old-input RGB-D seeds in every group; add group-specific new-input seeds. Optimize all Gaussian parameters; no later split/prune.','loss':'RGB L1 + .2 relative depth L1 (clipped1) + .05 alpha deficit, on valid GT pixels for depth and alpha; RGB full image','learning_rates':{'means':.00005,'scales':.001,'quats':.0005,'opacity':.01,'colors':.004},'heldout_ids':[p['id'] for p in heldposes],'heldout_scope':'Original 32 held-out frames, excluded from original low training and full-input reliability calibration; remove near augmentation views (distance<.4 and rotation<20deg), same heldout set for all groups','new_observation_source':'high PLY pseudo-GT RGB-D at recorded poses; oracle RGB-D augmentation, not RGB-only/video enhancement','old_observation_source':'existing 128 training camera poses; high PLY re-render at320 for consistent RGB/expected-depth targets. These poses already had RGB-D reconstruction inputs.','scores':'Use previously frozen full-input RGBDN B; no B-guided frame selection or changes','interpretation':'Validate groups of81 views, not individual marginal frame utility. Same compute, two seeds, one scene; relative to both untouched low and replay-only. No claim of full 2D GauSS-MI replication or real-world GT.' ,'code_sha256':digest(Path(__file__)),'source_hashes':srcfiles,'reliability_sha256':statehash,'early_stopping':False,'best_checkpoint_selection':False}
    settings['fingerprint']=canonical_hash(settings);atomic_json(out/'protocol.json',settings)
    renderer=EvalRenderer(c);old=[];held=[]
    for i,p in enumerate(oldposes):
        gt,_=renderer.render(p,'high');old.append((p,gt))
    for p in heldposes:
        gt,_=renderer.render(p,'high');mask,_=valid_reference(gt,p['camera_to_world'],k,c)
        if mask.sum()>=500:held.append((p,gt,mask))
    assert len(held)>=8
    print('DATA',len(old),'old',len(held),'heldout',flush=True)
    renderer.load('low')
    def seeds(data,limit,exclude=None):
        points=[];colors=[];scales=[]
        for p,gt in data:
            low,_=renderer.render(p,'low');d=gt['depth'];y,x=np.mgrid[1:240:3,1:320:3];z=d[y,x];rgb=gt['rgb'][y,x]
            valid=(gt['alpha'][y,x]>.9)&np.isfinite(z)&(z>.35)&(z<12)
            dz=np.abs(low['depth'][y,x]-z)/np.maximum(z,.1);color=np.mean(np.abs(low['rgb'][y,x]-rgb),axis=-1);valid&=(low['alpha'][y,x]<.7)|(dz>.08)|(color>.08)
            xyz=np.stack([(x-k['cx'])*z/k['fx'],(y-k['cy'])*z/k['fy'],z],-1);m=np.array(p['camera_to_world']);xyz=xyz@m[:3,:3].T+m[:3,3];valid&=(xyz>=np.array(c['roi']['min'])).all(-1)&(xyz<np.array(c['roi']['max'])).all(-1)
            points.append(xyz[valid]);colors.append(rgb[valid]);scales.append(np.clip(z[valid]*3/k['fx']*.55,.008,.06))
        pts=np.concatenate(points).astype('float32');col=np.concatenate(colors).astype('float32');sz=np.concatenate(scales).astype('float32');keys=np.floor(pts/.04).astype('int64');_,ix=np.unique(keys,axis=0,return_index=True)
        if exclude is not None:ix=np.array([i for i in ix if tuple(keys[i]) not in exclude],dtype=int)
        rng=np.random.default_rng(253)
        if len(ix)>limit:ix=rng.choice(ix,limit,replace=False)
        return pts[ix],col[ix],sz[ix]
    oldseed=seeds(old,20000);exclude={tuple(x) for x in np.floor(oldseed[0]/.04).astype(int)};newseed={g:seeds(data,20000,exclude) for g,data in groups.items()}
    atomic_npz(out/'common_old_seeds.npz',points=oldseed[0],colors=oldseed[1],scales=oldseed[2])
    for name,a in newseed.items():atomic_npz(out/f'{name}_new_seeds.npz',points=a[0],colors=a[1],scales=a[2])
    scene=renderer.scene;degree=scene.sh_degree
    initial={name:v.detach().clone() for name,v in {'means':scene.means,'quats':scene.quats,'scales':scene.scales.log(),'opacity':torch.logit(scene.opacities.clamp(1e-6,1-1e-6)),'colors':scene.colors}.items()}
    def camera(p):
        kk=resize(p['intrinsics'],320,240);return torch.tensor(np.linalg.inv(p['camera_to_world']),device='cuda',dtype=torch.float32)[None],torch.tensor([[[kk['fx'],0,kk['cx']],[0,kk['fy'],kk['cy']],[0,0,1]]],device='cuda',dtype=torch.float32)
    heldcam=[camera(p) for p,g,m in held]
    def render(params,v,K):
        pred,al,_=rasterization(params['means'],params['quats'],params['scales'].exp(),params['opacity'].sigmoid(),params['colors'],v,K,320,240,packed=True,sh_degree=degree,render_mode='RGB+ED',near_plane=.1,far_plane=12.);return pred[0],al[0,...,0]
    weakmasks=[];baselineimages=[]
    with torch.no_grad():
        for (p,g,m),(v,K) in zip(held,heldcam):
            pred,al=render(initial,v,K);a=pred.cpu().numpy();alpha=al.cpu().numpy();dep=np.abs(a[...,3]-g['depth'])/np.maximum(g['depth'],.1);bad=(alpha<.7)|(dep>.1)|(np.mean(np.abs(np.clip(a[...,:3],0,1)-g['rgb']),axis=-1)>.08);weakmasks.append(m&bad);baselineimages.append(np.clip(a[...,:3],0,1))
    def evaluate(params):
        counts=np.zeros(2);rgbse=np.zeros(2);rgbl1=np.zeros(2);dep=np.zeros(2);recovered=np.zeros(2);images=[]
        with torch.no_grad():
            for i,((p,g,m),(v,K)) in enumerate(zip(held,heldcam)):
                pred,al=render(params,v,K);a=pred.cpu().numpy();alpha=al.cpu().numpy();err=(np.clip(a[...,:3],0,1)-g['rgb']);de=np.clip(np.abs(a[...,3]-g['depth'])/np.maximum(g['depth'],.1),0,1);valid=(alpha>=.7)&np.isfinite(a[...,3])&(a[...,3]>0);de[~valid]=1
                for j,mask in enumerate([m,weakmasks[i]]):
                    counts[j]+=mask.sum();rgbse[j]+=(err[mask]**2).sum();rgbl1[j]+=np.abs(err[mask]).sum();dep[j]+=de[mask].sum();recovered[j]+=(mask&valid&(de<=.1)).sum()
                if i in (0,len(held)//2,len(held)-1):images.append(np.clip(a[...,:3],0,1))
        return {label:{'pixels':int(counts[j]),'psnr_db':float(-10*np.log10(max(rgbse[j]/max(counts[j]*3,1),1e-12))),'rgb_l1':float(rgbl1[j]/max(counts[j]*3,1)),'relative_depth_error':float(dep[j]/max(counts[j],1)),'accurate_depth_fraction':float(recovered[j]/max(counts[j],1))} for j,label in enumerate(['all','baseline_weak'])},images
    baseline,_=evaluate(initial);atomic_json(out/'untouched_low_baseline.json',baseline)
    def dataset(data):
        out=[]
        for p,g in data:
            v,K=camera(p);mask=np.isfinite(g['depth'])&(g['depth']>.1)&(g['depth']<12)&(g['alpha']>=.9)
            out.append((v,K,torch.tensor(g['rgb'],device='cuda'),torch.tensor(g['depth'],device='cuda'),torch.tensor(mask,device='cuda')))
        return out
    replay=dataset(old);records=[]
    for name in ['replay_only',*groups]:
        new=dataset(groups[name]) if name!='replay_only' else [];extra=oldseed if name=='replay_only' else tuple(np.concatenate([a,b]) for a,b in zip(oldseed,newseed[name]))
        pts,col,sz=extra;N=len(pts);newcolors=np.zeros((N,initial['colors'].shape[1],3),dtype='float32');newcolors[:,0]=(col-.5)/.28209479177387814
        additions={'means':pts,'quats':np.tile([1.,0,0,0],(N,1)).astype('float32'),'scales':np.repeat(np.log(sz[:,None]),3,1),'opacity':np.full(N,math.log(.6/.4),dtype='float32'),'colors':newcolors}
        for seed in [253,719]:
            dest=out/f'{name}_seed{seed}';dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():records.append(read_json(dest/'result.json'));continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed);params=torch.nn.ParameterDict({key:torch.nn.Parameter(torch.cat([value.clone(),torch.tensor(additions[key],device='cuda')],0)) for key,value in initial.items()});opt=torch.optim.Adam([{'params':[params[key]],'lr':lr} for key,lr in settings['learning_rates'].items()],eps=1e-15);start=time.perf_counter();torch.cuda.reset_peak_memory_stats();losslog=[]
            for step in range(settings['steps']):
                ds=new if new and rng.random()<.5 else replay;v,K,rgb,d,mask=ds[int(rng.integers(len(ds)))];pred,al=render(params,v,K)
                rgb_loss=(pred[...,:3]-rgb).abs().mean();depth_loss=((pred[...,3]-d).abs()/d.clamp_min(.1)).clamp_max(1)[mask].mean();alpha_loss=(1-al[mask]).mean();loss=rgb_loss+.2*depth_loss+.05*alpha_loss
                assert torch.isfinite(loss);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
                with torch.no_grad():params['scales'].clamp_(-9,0);params['opacity'].clamp_(-10,10)
                if (step+1)%250==0:
                    losslog.append({'step':step+1,'rgb':float(rgb_loss),'depth':float(depth_loss),'alpha':float(alpha_loss)});print('TRAIN',name,seed,losslog[-1],flush=True)
            after,afterimages=evaluate(params);result={'group':name,'seed':seed,'steps':settings['steps'],'old_seed_gaussians':len(oldseed[0]),'new_seed_gaussians':0 if name=='replay_only' else len(newseed[name][0]),'total_gaussians':len(params['means']),'heldout_views':len(held),'after':after,'delta_vs_untouched_low':{area:{'psnr_db':after[area]['psnr_db']-baseline[area]['psnr_db'],'rgb_l1_reduction':baseline[area]['rgb_l1']-after[area]['rgb_l1'],'depth_error_reduction':baseline[area]['relative_depth_error']-after[area]['relative_depth_error'],'accurate_depth_fraction_gain':after[area]['accurate_depth_fraction']-baseline[area]['accurate_depth_fraction']} for area in baseline},'elapsed_s':time.perf_counter()-start,'peak_memory_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'loss_log':losslog}
            source=PlyData.read(c['_low_ply'])['vertex'].data;vertex=np.zeros(len(params['means']),dtype=source.dtype);vertex[:len(source)]=source;vals={key:value.detach().cpu().numpy() for key,value in params.items()}
            for j,key in enumerate(['x','y','z']):vertex[key]=vals['means'][:,j]
            for j in range(3):vertex[f'scale_{j}']=vals['scales'][:,j];vertex[f'f_dc_{j}']=vals['colors'][:,0,j]
            for j in range(4):vertex[f'rot_{j}']=vals['quats'][:,j]
            rest=vals['colors'][:,1:,:].transpose(0,2,1).reshape(len(vertex),-1)
            for j in range(rest.shape[1]):vertex[f'f_rest_{j}']=rest[:,j]
            vertex['opacity']=vals['opacity'];PlyData([PlyElement.describe(vertex,'vertex')]).write(str(dest/'augmented.ply'))
            sheet=Image.new('RGB',(960,3*260));draw=ImageDraw.Draw(sheet)
            for j,idx in enumerate([0,len(held)//2,len(held)-1]):
                for col,img in enumerate([held[idx][1]['rgb'],baselineimages[idx],afterimages[j]]):sheet.paste(Image.fromarray(np.uint8(np.clip(img,0,1)*255)),(col*320,j*260+20))
                draw.text((5,j*260+3),f'{held[idx][0]["id"]}: pseudo-GT | original low | augmented',fill='white')
            sheet.save(dest/'heldout_examples.jpg');atomic_json(dest/'result.json',result);records.append(result);del params,opt;gc.collect();torch.cuda.empty_cache();print('RESULT',name,seed,result['delta_vs_untouched_low'],flush=True)
    assert all(digest(p)==h for p,h in srcfiles.items()) and digest(fullstate/'reliability_RGBDN.npz')==statehash
    Bscores=read_json(fullstate/'scores.json');B={f'clip_{r["clip"]}':r['full_input_RGBDN_B'] for r in Bscores['scores']}
    with (fullstate/'per_frame_scores.csv').open() as f:B['repeat_control']=float(next(csv.DictReader(f))['full_input_RGBDN_B'])
    rows=[]
    for name in ['replay_only',*groups]:
        rs=[r for r in records if r['group']==name];row={'group':name,'B':B.get(name),'new_seed_gaussians':rs[0]['new_seed_gaussians']}
        for area in ['all','baseline_weak']:
            for metric in ['psnr_db','rgb_l1_reduction','depth_error_reduction','accurate_depth_fraction_gain']:
                values=[r['delta_vs_untouched_low'][area][metric] for r in rs];row[f'{area}_{metric}_mean']=float(np.mean(values));row[f'{area}_{metric}_std']=float(np.std(values,ddof=1))
                if name!='replay_only':
                    controls={r['seed']:r for r in records if r['group']=='replay_only'};paired=[r['delta_vs_untouched_low'][area][metric]-controls[r['seed']]['delta_vs_untouched_low'][area][metric] for r in rs];row[f'{area}_{metric}_net_vs_replay_mean']=float(np.mean(paired))
        rows.append(row)
    base=read_json(root/'BASELINE_SNAPSHOT.json');assert all(digest(Path(base['source'])/key)==h for key,h in base['files'].items())
    atomic_json(out/'all_results.json',records);atomic_json(out/'summary.json',{'fingerprint':settings['fingerprint'],'heldout_views':len(held),'baseline':baseline,'groups':rows,'source_models_and_frozen_reliability_unchanged':True,'original_code_files_unchanged':len(base['files']),'conclusion_scope':'GT RGB-D group-level supplementation pilot on one scene, two seeds; no video enhancement and no proof of B generalization; net vs replay isolates benefit beyond more optimization of existing observations.'})
    print('GT_AUGMENTATION_COMPLETE',json.dumps({'heldout_views':len(held),'groups':rows}),flush=True)

if __name__=='__main__':main()
