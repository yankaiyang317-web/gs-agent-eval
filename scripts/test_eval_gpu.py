"""Installed gsplat GPU validation against explicit projected alpha compositing."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from gs_eval.storage import atomic_json,digest

def run(out):
 import torch,gsplat
 from gsplat import rasterization
 if not torch.cuda.is_available():raise RuntimeError('GPU_SMOKE_NOT_EXECUTED: CUDA unavailable')
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter();d='cuda';means=torch.tensor([[0.,0.,2.],[0.,0.,3.]],device=d);scales=torch.full((2,3),.1,device=d);quats=torch.tensor([[1.,0,0,0],[1.,0,0,0]],device=d);op=torch.tensor([.5,.6],device=d);v=torch.eye(4,device=d)[None];K=torch.tensor([[[20.,0,8.],[0,20.,8.],[0,0,1.]]],device=d);before=[t.clone() for t in [means,scales,quats,op]]
 def raster(features,mode='RGB'):
  return rasterization(means,quats,scales,op,features,v,K,16,16,sh_degree=None,render_mode=mode,packed=True,eps2d=.3)
 y,x=np.indices((16,16));a=[]
 for z,opacity in [(2,.5),(3,.6)]:
  var=(20*.1/z)**2+.3;aa=opacity*np.exp(-.5*((x+.5-8)**2+(y+.5-8)**2)/var);aa=np.where(aa>=1/255,aa,0);a.append(aa)
 weights=np.stack([a[0],(1-a[0])*a[1]],-1);explicit_alpha=weights.sum(-1);ones=torch.ones((2,1),device=d);render,alpha,_=raster(ones);err_alpha=float(np.max(np.abs(render[0,...,0].cpu().numpy()-explicit_alpha)));assert err_alpha<2e-5,err_alpha;assert torch.allclose(render[...,0],alpha[...,0],atol=2e-6)
 q=np.random.default_rng(0).normal(size=(16,16)).astype('float32');f=torch.zeros((2,1),device=d,requires_grad=True);im,_,_=raster(f);J=(im[0,...,0]*torch.tensor(q,device=d)).sum();grad=torch.autograd.grad(J,f)[0].cpu().numpy()[:,0];expected=(weights*q[...,None]).sum((0,1));err_grad=float(np.max(np.abs(grad-expected)));assert err_grad<3e-5,(grad,expected)
 finite=[]
 for i in range(2):
  plus=torch.zeros((2,1),device=d);minus=plus.clone();plus[i]=.001;minus[i]=-.001;rp=raster(plus)[0][0,...,0];rn=raster(minus)[0][0,...,0];finite.append(float(((rp-rn)*torch.tensor(q,device=d)).sum()/.002))
 assert np.allclose(grad,finite,atol=3e-5)
 gain=torch.tensor([[2.],[3.]],device=d);im,_,_=raster(gain);err_gain=float(np.max(np.abs(im[0,...,0].cpu().numpy()-(weights*np.array([2.,3.])).sum(-1))));assert err_gain<3e-5;assert im.max()>1,'gain incorrectly clipped'
 rgb=torch.tensor([[.2,.3,.4],[.4,.1,.2]],device=d);ed,al,_=raster(rgb,'RGB+ED');acc,_,_=raster(rgb,'RGB+D');valid=al[...,0]>.01;err_depth=float(torch.max(torch.abs(ed[...,3][valid]-acc[...,3][valid]/al[...,0][valid])));assert err_depth<2e-5
 assert all(torch.equal(a,b) for a,b in zip(before,[means,scales,quats,op]));torch.cuda.synchronize()
 result={'status':'passed','torch':torch.__version__,'gsplat':gsplat.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'explicit_alpha_max_error':err_alpha,'contribution_gradient_max_error':err_grad,'finite_difference':finite,'gain_max_error':err_gain,'gain_max':float(im.max()),'expected_depth_max_error':err_depth,'geometry_unchanged':True,'elapsed_s':time.perf_counter()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'tolerances':{'alpha':2e-5,'gradient':3e-5,'gain':3e-5,'depth':2e-5}};atomic_json(out,result);print(json.dumps(result));return result
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'outputs/evaluation/gpu_smoke.json')
