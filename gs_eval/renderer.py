"""GPU dependencies are loaded only by the offline renderer, never navigation."""
import gc
import numpy as np
from .alignment import camera_for_high,resize

def environment_identity():
    try:
        import torch
        return {'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}
    except ImportError:return {'cuda_runtime':None,'gpu':'unavailable'}

class EvalRenderer:
    def __init__(self,c,device='cuda'):
        import torch
        import gsplat
        self.torch=torch;self.raster=gsplat.rasterization;self.c=c;self.device=device;self.scene=None;self.kind=None
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA_UNAVAILABLE')
    def load(self,kind):
        if self.kind!=kind:
            from ._vendor.ply_loader import load_gaussian_ply
            self.scene=None;gc.collect()
            if self.device=='cuda':self.torch.cuda.empty_cache()
            self.scene=load_gaussian_ply(self.c['high_ply'] if kind=='high' else self.c['_low_ply'],self.device);self.kind=kind
        return self.scene
    def arguments(self,pose,kind):
        t=self.torch;c=self.c;k=resize(pose['intrinsics'],c['evaluation_camera']['width'],c['evaluation_camera']['height']);m=np.array(pose['camera_to_world']);scale=float(c['high_to_low_similarity']['scale']) if kind=='high' else 1.
        if kind=='high':m=camera_for_high(m,c['high_to_low_similarity'])
        view=t.tensor(np.linalg.inv(m),device=self.device,dtype=t.float32)[None];K=t.tensor([[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]],device=self.device,dtype=t.float32)
        return dict(viewmats=view,Ks=K,width=k['width'],height=k['height'],near_plane=c['surface']['near']/scale,far_plane=c['surface']['far']/scale,packed=True,backgrounds=None),scale,k
    def render(self,pose,kind):
        s=self.load(kind);kw,scale,k=self.arguments(pose,kind)
        with self.torch.no_grad():out,alpha,_=self.raster(s.means,s.quats,s.scales,s.opacities,s.colors,sh_degree=s.sh_degree,render_mode='RGB+ED',**kw)
        a=out[0].detach().cpu().numpy();return {'rgb':np.clip(a[...,:3],0,1),'depth':a[...,3]*scale,'alpha':alpha[0,...,0].detach().cpu().numpy()},k
    def feature(self,pose,features):
        s=self.load('low');t=self.torch;kw,_,_=self.arguments(pose,'low');f=t.as_tensor(features,device=self.device,dtype=t.float32).reshape(s.count,-1)
        try:
            with t.no_grad():out,_,_=self.raster(s.means.detach(),s.quats.detach(),s.scales.detach(),s.opacities.detach(),f,sh_degree=None,render_mode='RGB',**kw)
        except (RuntimeError,NotImplementedError) as e:raise RuntimeError('CONTRIBUTION_BACKEND_UNAVAILABLE: '+str(e)) from e
        return out[0].detach().cpu().numpy()
    def contribution(self,pose,pixel_maps):
        s=self.load('low');t=self.torch;kw,_,_=self.arguments(pose,'low');q=np.stack(pixel_maps,-1);f=t.zeros((s.count,len(pixel_maps)),device=self.device,requires_grad=True)
        try:
            out,_,_=self.raster(s.means.detach(),s.quats.detach(),s.scales.detach(),s.opacities.detach(),f,sh_degree=None,render_mode='RGB',**kw);loss=(out[0]*t.tensor(q,device=self.device,dtype=t.float32)).sum();grad=t.autograd.grad(loss,f)[0]
        except (RuntimeError,NotImplementedError) as e:raise RuntimeError('CONTRIBUTION_BACKEND_UNAVAILABLE: '+str(e)) from e
        return grad.detach().cpu().numpy()
