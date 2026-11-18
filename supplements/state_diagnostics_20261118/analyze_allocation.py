"""Replay all 500 logical allocation diagnostics from 386 minimal saved states."""
from pathlib import Path
import argparse
import json
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from resource_common import read_json,verify_resource,write_csv,sha256

CORPORA=['3Sources','BBC','Reuters','Cora','CiteSeer']
METHODS=['full-CV','consensus-only-CV','views-only-CV','consensus-only-at-full-lambda','views-only-at-full-lambda']


def edge_loss(Z,P,N,lam,n):
    return lam/n*(float(np.square(Z[:,P[:,0]]-Z[:,P[:,1]]).sum())+float((Z[:,N[:,0]]*Z[:,N[:,1]]).sum()))


def matrix(P,N,lam,n):
    row=np.r_[P[:,0],P[:,1],P[:,0],P[:,1],N[:,0],N[:,1]]
    col=np.r_[P[:,0],P[:,1],P[:,1],P[:,0],N[:,1],N[:,0]]
    val=lam/n*np.r_[np.ones(2*len(P)),-np.ones(2*len(P)),.5*np.ones(2*len(N))]
    return sparse.csr_matrix((val,(row,col)),shape=(n,n))


def analyze(root,record):
    with np.load(root/record['state_file'],allow_pickle=False) as z:
        H=z['H'];V=record['V'];n=record['n']
        Hv=[z[f'Hv{v}'] for v in range(V)];P=z['P'].astype(int).reshape(-1,2);N=z['N'].astype(int).reshape(-1,2)
    eta=record['eta'];lam=record['lambda_pair'];b=1/V
    assert record['b']==b and H.shape==(record['k'],n)
    assert len(P)+len(N)==record['budget']
    assert all(h.shape==H.shape for h in Hv)
    endpoints=np.unique(np.r_[P.ravel(),N.ravel()]);m=len(endpoints)
    M=matrix(P,N,lam,n);HM=(M@H.T).T
    direct=b*((1-eta)*edge_loss(H,P,N,lam,n)+eta/V*sum(edge_loss(h,P,N,lam,n) for h in Hv))
    baseline=b*edge_loss(H,P,N,lam,n)
    linear=quadratic=mean_norm=mean_norm2=0.;view_losses=[]
    for h in Hv:
        delta=h-H;ld=2*float((HM*delta).sum());qd=float((delta*(M@delta.T).T).sum())
        linear+=b/V*ld;quadratic+=b/V*qd
        norm=float(np.linalg.norm(delta[:,endpoints]));mean_norm+=norm/V;mean_norm2+=norm**2/V
        view_losses.append(edge_loss(h,P,N,lam,n))
    rms=float(np.sqrt(mean_norm2/m));normbound=float(abs(M).sum(axis=1).max())
    bound=b*eta*normbound*(2*np.linalg.norm(H[:,endpoints])*mean_norm+mean_norm2)
    error=abs(direct-(baseline+eta*(linear+quadratic)))
    matrix_error=abs(edge_loss(H,P,N,lam,n)-float((HM*H).sum()))
    assert error<1e-10*max(1,abs(direct),abs(baseline),abs(eta*linear),abs(eta*quadratic))
    assert matrix_error<1e-10*max(1,edge_loss(H,P,N,lam,n))
    assert abs(direct-baseline)<=bound+1e-10
    assert max(np.max(abs(h[:,endpoints].sum(0)-1)) for h in Hv)<1e-10
    return dict(n=n,k=H.shape[0],V=V,eta=eta,lambda_pair=lam,b=b,endpoints=m,endpoint_rms=rms,
        baseline_bD_H=baseline,allocation_direct=direct,linear_slope=linear,quadratic_slope=quadratic,
        total_slope=linear+quadratic,linear_actual=eta*linear,quadratic_actual=eta*quadratic,
        allocation_minus_baseline=direct-baseline,abs_difference_bound=float(bound),identity_abs_error=error,
        matrix_edge_abs_error=matrix_error,mean_D_Hv=float(np.mean(view_losses)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    assert root!=out and root not in out.parents,'Write results outside the sealed resource'
    verify_resource(root);plan=read_json(root/'ALLOCATION_RUNS.json')
    assert plan['queried_pair_weights']==1
    expected={(d,s,q,m) for d in CORPORA for s in range(504,514) for q in [160,400] for m in METHODS}
    assert len(plan['records'])==500
    assert {(r['dataset'],r['seed'],r['budget'],r['method']) for r in plan['records']}==expected
    out.mkdir(parents=True,exist_ok=False)
    rows=[];cache={}
    for r in plan['records']:
        state=r['state_file']
        if state not in cache:cache[state]=analyze(root,r)
        else:
            for key in ['n','k','V','eta','lambda_pair','b']:assert cache[state][key]==r[key]
        rows.append(dict(dataset=r['dataset'],seed=r['seed'],budget=r['budget'],method=r['method'],**cache[state]))
    assert len(cache)==386
    summaries=[]
    metrics=['endpoint_rms','baseline_bD_H','allocation_direct','linear_slope','quadratic_slope','total_slope','linear_actual','quadratic_actual','allocation_minus_baseline']
    for ds in CORPORA:
        for q in [160,400]:
            for method in METHODS:
                group=[r for r in rows if (r['dataset'],r['budget'],r['method'])==(ds,q,method)]
                assert len(group)==10
                row=dict(dataset=ds,budget=q,method=method,count=10)
                for name in metrics:
                    values=np.array([r[name] for r in group])
                    row.update({name+'_mean':float(values.mean()),name+'_sd':float(values.std(ddof=1)),name+'_min':float(values.min()),name+'_max':float(values.max())})
                row['quadratic_slope_negative']=sum(r['quadratic_slope'] < -1e-12 for r in group)
                row['total_slope_negative']=sum(r['total_slope'] < -1e-12 for r in group)
                summaries.append(row)
    write_csv(out/'allocation_decomposition_500.csv',rows)
    write_csv(out/'allocation_decomposition_summary_50.csv',summaries)
    qa={'status':'PASS','logical_runs':len(rows),'unique_states':len(cache),'summary_groups':len(summaries),
        'max_identity_abs_error':max(r['identity_abs_error'] for r in rows),
        'max_matrix_edge_abs_error':max(r['matrix_edge_abs_error'] for r in rows),
        'manifest_sha256':sha256(root/'MANIFEST.json'),'no_fit':True,'no_external_features_or_classes_read':True,
        'reused_states_are_not_new_observations':True}
    (out/'QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(qa),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
