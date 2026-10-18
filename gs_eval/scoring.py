import numpy as np

def fixed_reliability(deltas,support,c):
    raw=np.sum(np.asarray(deltas,dtype=np.float64),axis=0);limit=c['reliability']['logodds_clip'];ell=np.clip(raw,-limit,limit);ell=np.where(support>1e-10,ell,0);p=1/(1+np.exp(-ell));g=np.logaddexp(0,-ell)
    qs=[0,.01,.1,.5,.9,.99,1];stats={'p_quantiles':np.quantile(p,qs).tolist(),'g_quantiles':np.quantile(g,qs).tolist(),'support_quantiles':np.quantile(support,qs).tolist(),'no_evidence_fraction':float(np.mean(support<=1e-10)),'positive_clip_fraction':float(np.mean(raw>=limit)),'negative_clip_fraction':float(np.mean(raw<=-limit))}
    # Adaptation remains scientifically unvalidated even when numerical tests pass.
    stats['reliability_status']='degenerate_or_unvalidated';stats['near_constant']=bool(np.std(p)<1e-4);return p,g,raw,stats

def evidence_map(loss,mask,c):
    r=c['reliability'];return mask*(-np.log(np.maximum(r['lambda_loss']*loss,c['residual']['loss_epsilon'])))

def view_utility(gain,mask):
    raw=float(np.sum(gain[mask],dtype=np.float64));return raw,raw/mask.size
