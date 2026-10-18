"""Isolated RGB-only fixed-topology fine-tuning pilot. No source PLY overwritten."""
from pathlib import Path
import json,time,csv,math,gc
import numpy as np
import torch
from plyfile import PlyData,PlyElement
from gsplat import rasterization
from gs_env.rendering.ply_loader import load_gaussian_ply
from gs_eval.config import load_config
from gs_eval.storage import read_json,atomic_json,digest
from gs_eval.cache import read_render,explicit_mask
from gs_eval.geometry import valid_reference
from gs_eval.alignment import resize
from gs_eval.trajectory import read_poses
from gs_eval.evaluation import rank_corr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/app');src=root/'outputs/evaluation/three_clip_run_20261016';out=root/'outputs/evaluation/frame_value_validation_20261016';out.mkdir(parents=True,exist_ok=True)
c=load_config(root/'eval_configs/jiudian_pilot_v1.json');cache=Path(read_json(root/'outputs/evaluation/pilot_validation.json')['reference_cache'])
hashes={str(p):digest(p) for p in [Path(c['_low_ply']),Path(c['high_ply']),cache/'reliability.npz',cache/'reference_surface.npz']}
allposes=[];groups={};perframe=[]
for i in range(3):
    poses=read_poses(src/'inputs'/f'clip_{i:03d}'/'trajectory.jsonl');allposes+=poses
    data=[read_render(src/f'clip_{i:03d}'/'diagnostics/high_frames'/f'{j:05d}.npz') for j in range(81)]
    groups[f'clip_{i+1}']=(poses,data)
    perframe+=list(csv.DictReader((src/f'clip_{i:03d}'/'per_frame.csv').open()))
groups['repeat_control']=([allposes[0]]*81,[groups['clip_1'][1][0]]*81)
ref=read_poses(c['reference_poses']);held=[]
for i,p in enumerate(ref):
    m=np.array(p['camera_to_world']);overlap=False
    for q in allposes:
        n=np.array(q['camera_to_world']);dist=np.linalg.norm(m[:3,3]-n[:3,3]);angle=np.degrees(np.arccos(np.clip((np.trace(m[:3,:3].T@n[:3,:3])-1)/2,-1,1)))
        if dist<.4 and angle<20:overlap=True;break
    if not overlap:
        gt=read_render(cache/'reference_high'/f'{i:05d}.npz');k=resize(p['intrinsics'],320,240);mask,_=valid_reference(gt,p['camera_to_world'],k,c,explicit_mask(p,c,k))
        if mask.sum()>=500:held.append((p,gt,mask))
assert len(held)>=20
def camera(p):
    k=resize(p['intrinsics'],320,240)
    return torch.tensor(np.linalg.inv(p['camera_to_world']),device='cuda',dtype=torch.float32)[None],torch.tensor([[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]],device='cuda',dtype=torch.float32)
heldcam=[camera(p) for p,g,m in held]
scene=load_gaussian_ply(c['_low_ply'],'cuda');degree=scene.sh_degree
initial={k:v.detach().clone() for k,v in {'means':scene.means,'quats':scene.quats,'scales':scene.scales.log(),'opacity':torch.logit(scene.opacities.clamp(1e-6,1-1e-6)),'colors':scene.colors}.items()}
def render(params,v,k,depth=False):
    rgb,alpha,_=rasterization(params['means'],params['quats'],params['scales'].exp(),params['opacity'].sigmoid(),params['colors'],v,k,320,240,packed=True,sh_degree=degree,render_mode='RGB+ED' if depth else 'RGB',near_plane=.1,far_plane=12.)
    return rgb[0],alpha[0,...,0]
def metrics(params):
    sse=0.;l1=0.;dep=0.;count=0;details=[];images=[]
    with torch.no_grad():
        for j,((p,gt,mask),(v,k)) in enumerate(zip(held,heldcam)):
            pred,alpha=render(params,v,k,True);a=pred.cpu().numpy();al=alpha.cpu().numpy();err=(np.clip(a[...,:3],0,1)-gt['rgb'])[mask];se=float((err**2).sum());ab=float(np.abs(err).sum());n=int(mask.sum());sse+=se;l1+=ab;count+=n
            bad=(~np.isfinite(a[...,3]))|(a[...,3]<=0)|(al<.7);d=np.minimum(np.abs(a[...,3]-gt['depth'])/np.maximum(gt['depth'],.1),1);d[bad]=1;de=float(d[mask].sum());dep+=de
            details.append({'id':p['id'],'pixels':n,'rgb_mse':se/(n*3),'relative_depth':de/n})
            if j in (0,len(held)//2,len(held)-1):images.append(np.clip(a[...,:3],0,1))
    return {'psnr_db':float(-10*np.log10(max(sse/(count*3),1e-12))),'rgb_l1':l1/(count*3),'relative_depth_error':dep/count,'valid_pixels':count,'views':details},images
baseline,baseimages=metrics(initial);atomic_json(out/'baseline.json',baseline)
settings={'steps':400,'seeds':[253,719],'resolution':[320,240],'training_frames_per_group':81,'groups':list(groups),'heldout_pose_ids':[p['id'] for p,g,m in held],'heldout_exclusion':{'distance_below':.4,'rotation_deg_below':20},'loss':'full-frame RGB L1 only; no GT depth supervision','optimizer':'Adam','learning_rates':{'means':.00003,'scales':.0005,'quats':.0002,'opacity':.005,'colors':.003},'topology':'fixed low-scene Gaussian count; no densification/pruning; fine-tuning pilot, not a full reconstruction benchmark','heldout_scope':'Held out from this fine-tuning only, not guaranteed unseen by original synthetic low-model training or reliability calibration. Same heldout set for all groups.','data':'High-quality pseudo-GT RGB at the recorded camera poses. Low-quality video RGB is not new ground truth.','budget':'Equal optimizer steps and 81 frame slots; repeated poses intentionally retained. No replay data. No early stopping or best-checkpoint selection.'}
atomic_json(out/'protocol.json',settings);results=[]
for name,(poses,data) in groups.items():
    cams=[camera(p) for p in poses];images=[torch.tensor(a['rgb'],device='cuda',dtype=torch.float32) for a in data]
    for seed in settings['seeds']:
        dest=out/f'{name}_seed{seed}';dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():results.append(read_json(dest/'result.json'));continue
        torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.clone()) for k,v in initial.items()});opt=torch.optim.Adam([{'params':[params[k]],'lr':lr} for k,lr in settings['learning_rates'].items()],eps=1e-15)
        losses=[]
        for step in range(settings['steps']):
            ix=int(rng.integers(81));v,k=cams[ix];pred,_=render(params,v,k);loss=(pred[...,:3]-images[ix]).abs().mean();loss.backward();opt.step();opt.zero_grad(set_to_none=True)
            with torch.no_grad():params['scales'].clamp_(math.log(.003),math.log(.25));params['opacity'].clamp_(-10,10)
            if (step+1)%100==0:
                losses.append({'step':step+1,'rgb_l1_training_sample':float(loss)});print(name,seed,step+1,float(loss),flush=True)
        after,afterimages=metrics(params)
        result={'group':name,'seed':seed,'steps':settings['steps'],'elapsed_s':time.perf_counter()-start,'peak_memory_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'before_psnr_db':baseline['psnr_db'],'after_psnr_db':after['psnr_db'],'delta_psnr_db':after['psnr_db']-baseline['psnr_db'],'delta_rgb_l1':baseline['rgb_l1']-after['rgb_l1'],'delta_depth_error':baseline['relative_depth_error']-after['relative_depth_error'],'after':after,'loss_samples':losses}
        # Preserve all original properties and only replace fitted Gaussian fields.
        ply=PlyData.read(c['_low_ply']);a=ply['vertex'].data.copy();vals={k:v.detach().cpu().numpy() for k,v in params.items()}
        for j,k in enumerate(['x','y','z']):a[k]=vals['means'][:,j]
        for j in range(3):a[f'scale_{j}']=vals['scales'][:,j];a[f'f_dc_{j}']=vals['colors'][:,0,j]
        for j in range(4):a[f'rot_{j}']=vals['quats'][:,j]
        rest=vals['colors'][:,1:,:].transpose(0,2,1).reshape(len(a),-1)
        for j in range(rest.shape[1]):a[f'f_rest_{j}']=rest[:,j]
        a['opacity']=vals['opacity'];PlyData([PlyElement.describe(a,'vertex')]).write(str(dest/'refined.ply'))
        from PIL import Image,ImageDraw
        canvas=Image.new('RGB',(960,3*260));draw=ImageDraw.Draw(canvas)
        for j,idx in enumerate((0,len(held)//2,len(held)-1)):
            for col,img in enumerate((held[idx][1]['rgb'],baseimages[j],afterimages[j])):canvas.paste(Image.fromarray(np.uint8(np.clip(img,0,1)*255)),(col*320,j*260+20))
            draw.text((5,j*260+3),f'{held[idx][0]["id"]}: reference | baseline low | refined',fill='white')
        canvas.save(dest/'heldout_examples.jpg');atomic_json(dest/'result.json',result);results.append(result)
        del params,opt;gc.collect();torch.cuda.empty_cache();print('RESULT',name,seed,result['delta_psnr_db'],result['delta_depth_error'],flush=True)
assert all(digest(p)==h for p,h in hashes.items())
values=np.array([float(r['I_normalized']) for r in perframe]);constant=np.array([float(r['I_constant_p05']) for r in perframe])
stats={'frame_spearman_B_constant_p05':rank_corr(values,constant),'frame_mean_abs_difference_B_constant_p05':float(np.mean(np.abs(values-constant))),'warning':'Within-trajectory adjacent frames are correlated; frame correlation is descriptive only, not a significance test.'}
table=[]
for name in groups:
    r=[x for x in results if x['group']==name];mean=lambda k:float(np.mean([x[k] for x in r]))
    b=float(perframe[0]['I_normalized']) if name=='repeat_control' else read_json(src/f'clip_{int(name[-1])-1:03d}'/'summary.json')['metric_families']['B_view_value']['primary']['frozen_gaussmi_mean']
    table.append({'group':name,'B':b,'delta_psnr_db_mean':mean('delta_psnr_db'),'delta_psnr_db_std':float(np.std([x['delta_psnr_db'] for x in r],ddof=1)),'delta_depth_error_mean':mean('delta_depth_error')})
stats['four_group_spearman_B_delta_psnr']=rank_corr([r['B'] for r in table],[r['delta_psnr_db_mean'] for r in table]);atomic_json(out/'validation_summary.json',{'protocol':settings,'baseline':baseline,'groups':table,'diagnostics':stats,'source_models_and_caches_unchanged':True,'conclusion_scope':'Small single-scene synthetic RGB-only fixed-topology pilot. Cannot establish general metric validity or measure hole restoration. No significance or broad ranking claim.'})
atomic_json(out/'all_results.json',results)
fig,ax=plt.subplots(figsize=(7,4.5))
for r in table:ax.errorbar(r['B'],r['delta_psnr_db_mean'],yerr=r['delta_psnr_db_std'],fmt='o');ax.annotate(r['group'],(r['B'],r['delta_psnr_db_mean']),xytext=(5,5),textcoords='offset points')
ax.axhline(0,color='grey',linestyle='--');ax.set(xlabel='Frozen B score (mean per frame)',ylabel='Held-out PSNR improvement (dB)',title='One scene; RGB fine-tuning; mean +/- SD over 2 seeds');fig.tight_layout();fig.savefig(out/'B_vs_realized_gain.png',dpi=160);plt.close(fig)
# Right-handed display coordinates: Xd=Xw, Yd=-Zw, Zd=Yw.
pos=np.array([p['camera_to_world'] for p in allposes])[:,:3,3];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for i in range(3):
    q=pos[i*81:(i+1)*81];axes[0].plot(q[:,0],-q[:,2],label=f'Clip {i+1}');axes[1].plot(np.arange(i*81,(i+1)*81),q[:,1],label=f'Clip {i+1}')
axes[0].scatter(pos[0,0],-pos[0,2],c='black',label='Start');axes[0].scatter(pos[-1,0],-pos[-1,2],c='black',marker='x',label='Finish')
axes[0].set(xlabel='Plan X = world X',ylabel='Plan Y = -world Z',title='Ground-plane trajectory (display axes)');axes[0].axis('equal');axes[0].legend()
axes[1].set(xlabel='Campaign frame',ylabel='Height = world Y (scene units)',title='Height is shown separately');axes[1].legend();fig.tight_layout();fig.savefig(out/'trajectory_plan_xy_and_height.png',dpi=160)
print('VALIDATION_COMPLETE',json.dumps({'groups':table,'diagnostics':stats}),flush=True)
