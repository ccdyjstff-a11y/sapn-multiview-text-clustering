"""Executable proposed factory. Root must freeze the study scope before real fits."""
from pathlib import Path
import hashlib,json
import numpy as np
from allocation_core import SAPNText

ROOT=Path(__file__).resolve().parent
ETAS={'full':.5,'consensus-only':0.,'views-only':1.}
CONFIG=dict(mu=.001,lambda_c=.2,lambda_r='auto',lambda_g=0.,gamma=2.,
  negative_loss='overlap',private_simplex='endpoints',geometry_weight=.1,
  reference_strength='coverage',supervision_weighting='fixed',reliability='none',
  endpoint_interval=5,tol_J=1e-5,tol_kkt=1e-3)

def make_model(method,value,seed,graph,max_iter,lambda_o):
    if method not in ETAS:raise ValueError(method)
    provenance=json.loads((ROOT/'CORE_DERIVATION.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'allocation_core.py').read_bytes()).hexdigest()==provenance['derived_sha256']
    return SAPNText(**dict(CONFIG,label_to_views=ETAS[method],lambda_p=value,lambda_n=value,
      seed=seed,geometry_graph=graph,max_iter=max_iter,lambda_o=lambda_o))

def digest(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def common_invariants(m):
    # Coefficients intentionally differ with eta and MUST NOT enter this equality.
    return dict(H_ref=digest(m.H_ref),r_v=digest(m.r_v),endpoints=digest(m._endpoints),
      endpoint_order=digest(m.endpoint_order),effective_u=m._reference_strength,
      private_domain=m._private_requested,private_simplex=m.private_simplex,
      data_graph_fingerprint=m._fingerprint)

def assert_allocation(m,value,eta):
    assert m._private_requested=='endpoints' and m.supervision_weighting=='fixed'
    if len(m._PA)+len(m._NA):assert m.private_simplex
    b=m.V**(1-m.gamma);target=b*value/m.n
    totals=[m.cp+sum(m.a[v]**m.gamma*m._view_coefficients(v)[0] for v in range(m.V)),
            m.cn+sum(m.a[v]**m.gamma*m._view_coefficients(v)[1] for v in range(m.V))]
    assert np.allclose(totals,target,atol=1e-13,rtol=1e-12)
    assert np.isclose(m.cp,(1-eta)*target) and np.isclose(m.cn,(1-eta)*target)
    for h in m.Hv:
        assert np.all(h>=-1e-12)
        if len(m._endpoints):assert np.max(abs(h[:,m._endpoints].sum(axis=0)-1))<1e-12
    return dict(eta=eta,consensus_coefficient=m.cp,individual_view_coefficient=b*eta*value/(m.n*m.V),
                total_positive_coefficient=totals[0],total_negative_coefficient=totals[1],target=target)

def fit_allocation(method,value,X,L,k,H0,Hv0,P,N,seed):
    """Same original two-stage plus continuation rule; accepts no category labels."""
    boot=make_model(method,value,seed,L,0,0.)
    boot.fit_predict(X,P,N,None,k,H_init=H0,Hv_init=Hv0,update_weights=True)
    expected=common_invariants(boot)
    common=make_model('full',value,seed,L,0,0.)
    common.fit_predict(X,P,N,None,k,H_init=H0,Hv_init=Hv0,update_weights=True)
    assert expected==common_invariants(common)
    assert all(np.array_equal(a,b) for a,b in zip(boot.U+boot.Hv+[boot.H_cons,boot.a],common.U+common.Hv+[common.H_cons,common.a]))
    assert_allocation(boot,value,ETAS[method])
    first=make_model(method,value,seed,L,200,0.)
    _,_,q1=first.fit_predict(X,P,N,None,k,H_init=H0,Hv_init=Hv0,update_weights=True)
    model=make_model(method,value,seed,L,1000,.1)
    z,H,q2=model.fit_predict(X,P,N,None,k,state_init=first.get_state(),update_weights=True)
    last=q2;continuation=None
    if not q2['converged']:
        resumed=make_model(method,value,seed,L,4000,.1)
        z,H,continuation=resumed.fit_predict(X,P,N,None,k,state_init=model.get_state(),update_weights=True)
        assert abs(continuation['J_hist'][0]-q2['J_hist'][-1])<=1e-9*max(1,abs(q2['J_hist'][-1]))
        model=resumed;last=continuation
    assert common_invariants(model)==expected
    allocation=assert_allocation(model,value,ETAS[method])
    for q in [q1,q2,continuation]:
        if q is not None:
            hist=np.asarray(q['J_hist'])
            assert np.isfinite(hist).all() and np.all(np.diff(hist)<=1e-9*np.maximum(1,abs(hist[:-1])))
    info=dict(method=method,parameter=value,eta=ETAS[method],status='ok',converged=last['converged'],
       kkt=last['kkt']['maximum'],stage1=q1,stage2=q2,continuation=continuation,
       common_invariants=expected,allocation=allocation,final_a=model.a.tolist())
    return z,H,info,model.get_state()
