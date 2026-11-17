"""Portable orchestration for the frozen v5 mechanism variants.

Model classes and paper core are unchanged except the model module's core-file
path. This is a new runner, not a byte-identical copy of formal run_study.py.
"""
import hashlib
import numpy as np
from portable_model_variants import make_model

def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def invariants(model):
    return dict(H_ref=digest(model.H_ref),r_v=digest(model.r_v),endpoints=digest(model._endpoints),
        endpoint_order=digest(model.endpoint_order),effective_u=model._reference_strength,
        coefficients=[model.cp,model.cn,model.vcp,model.vcn],private_domain=model._private_requested,
        data_graph_fingerprint=model._fingerprint)

def fit_variant(method,value,X,L,k,H0,Hv0,P,N,seed):
    if method=='full-v5':method='full'
    boot=make_model(method,value,seed,L,0,0.)
    boot.fit_predict(X,P,N,None,k,H_init=H0,Hv_init=Hv0,update_weights=True)
    expected=invariants(boot)
    common=make_model('full',value,seed,L,0,0.)
    common.fit_predict(X,P,N,None,k,H_init=H0,Hv_init=Hv0,update_weights=True)
    assert expected==invariants(common)
    assert np.array_equal(boot.H_cons,common.H_cons) and np.array_equal(boot.a,common.a)
    assert all(np.array_equal(a,b) for a,b in zip(boot.U+boot.Hv,common.U+common.Hv))
    first=make_model(method,value,seed,L,200,0.)
    _,_,q1=first.fit_predict(X,P,N,None,k,H_init=H0,Hv_init=Hv0,update_weights=True)
    model=make_model(method,value,seed,L,1000,.1)
    z,H,q2=model.fit_predict(X,P,N,None,k,state_init=first.get_state(),update_weights=True)
    continuation=None;last=q2
    if not q2['converged']:
        resumed=make_model(method,value,seed,L,4000,.1)
        z,H,continuation=resumed.fit_predict(X,P,N,None,k,state_init=model.get_state(),update_weights=True)
        assert abs(continuation['J_hist'][0]-q2['J_hist'][-1])<=1e-9*max(1.,abs(q2['J_hist'][-1]))
        model=resumed;last=continuation
    assert invariants(model)==expected
    for stage in [q1,q2,continuation]:
        if stage:
            hist=np.asarray(stage['J_hist']);assert np.isfinite(hist).all()
            assert np.all(np.diff(hist)<=1e-9*np.maximum(1,np.abs(hist[:-1])))
    q=dict(status='ok',method=method,parameter=value,converged=last['converged'],kkt=last['kkt']['maximum'],
           stage1=q1,stage2=q2,continuation=continuation,invariants=expected,final_a=model.a.tolist(),
           runner_scope='Portable controlled variant; exact formal study requires frozen inputs, initial states, graphs and full-lambda coupled branch.')
    return z,H,q,model.get_state()
