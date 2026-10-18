import numpy as np

def rigid(matrix):
    m=np.asarray(matrix,dtype=float)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-7):raise ValueError('INVALID_C2W')
    rotation(m[:3,:3]);return m

def rotation(r):
    r=np.asarray(r,dtype=float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):raise ValueError('INVALID_ROTATION')
    return r

def similarity(value):
    s=float(value['scale']);r=rotation(value['rotation']);t=np.asarray(value['translation'],dtype=float)
    if not np.isfinite(s) or s<=0 or t.shape!=(3,) or not np.isfinite(t).all():raise ValueError('INVALID_SIMILARITY')
    return s,r,t

def camera_for_high(c2w,sim):
    s,r,t=similarity(sim);c=rigid(c2w);o=np.eye(4);o[:3,:3]=r.T@c[:3,:3];o[:3,3]=r.T@(c[:3,3]-t)/s;return o

def intrinsics(k):
    k={x:float(k[x]) for x in ['fx','fy','cx','cy','width','height']}
    if not np.isfinite(list(k.values())).all() or min(k['fx'],k['fy'],k['width'],k['height'])<=0 or int(k['width'])!=k['width'] or int(k['height'])!=k['height']:raise ValueError('INVALID_INTRINSICS')
    k['width']=int(k['width']);k['height']=int(k['height']);return k

def resize(k,width,height):
    k=intrinsics(k);sx=width/k['width'];sy=height/k['height'];return {**k,'width':width,'height':height,'fx':k['fx']*sx,'cx':k['cx']*sx,'fy':k['fy']*sy,'cy':k['cy']*sy}

def fov_signature(k):
    k=intrinsics(k);return np.array([k['fx']/k['width'],k['fy']/k['height'],k['cx']/k['width'],k['cy']/k['height']])

def backproject(depth,c2w,k):
    y,x=np.indices(depth.shape);z=depth;cam=np.stack([(x-k['cx'])*z/k['fx'],(y-k['cy'])*z/k['fy'],z],axis=-1);c=rigid(c2w);return cam@c[:3,:3].T+c[:3,3]
