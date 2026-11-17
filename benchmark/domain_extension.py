"""Declared deterministic graph-domain extensions, not author-code identity.

No jitter, row deletion, random replacement or adjustable parameter is added.
The inherited 1e-10 row-feasibility check determines whether legacy floating
weights already satisfy the old interface; feasible legacy rows are unchanged.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN))
import mvchss_reimplementation as mv
import mvchss_sparse_graph_backend_v1 as sparse_mv


def centered_hypergraph(x, p=5):
    x = sparse.csr_matrix(x, dtype=float) if sparse.issparse(x) else np.asarray(x, dtype=float)
    if x.ndim!=2 or not np.isfinite(x.data if sparse.issparse(x) else x).all():
        raise ValueError('Finite features required')
    n = x.shape[0]
    if not isinstance(p,(int,np.integer)) or not 0<=p<n:
        raise ValueError('Require0<=p<n')
    norms = np.asarray(x.multiply(x).sum(1)).ravel() if sparse.issparse(x) else np.sum(x*x,axis=1)
    gram = x@x.T; gram = gram.toarray() if sparse.issparse(gram) else gram
    d2 = np.maximum(norms[:,None]+norms[None,:]-2*gram,0)
    dist = np.sqrt(d2); sigma = float(dist.mean())
    order = np.argsort(dist,axis=1,kind='stable')
    members = order[:,:p+1].copy(); repaired = []
    for i in range(n):
        if i not in members[i]:
            # Retain the original row if its selected hyperedge already
            # contains the center; otherwise force center + p other objects.
            members[i] = np.r_[i,order[i][order[i]!=i][:p]]
            repaired.append(i)
    if sigma>0:
        weights = np.exp(-d2[np.arange(n)[:,None],members]/sigma**2).sum(1)
    else:
        assert np.max(d2)==0
        # Fixed identical objects have exp(-0/sigma^2)=1 for every sigma>0.
        weights = np.full(n,p+1,dtype=float)
    h = sparse.csr_matrix((np.ones(n*(p+1)),
        (np.repeat(np.arange(n),p+1),members.ravel())),shape=(n,n))
    s = h.T@h.multiply((weights/(p+1))[:,None]); s = ((s+s.T)*.5).tocsr()
    assert np.min(np.asarray(s.sum(1)))>0
    return s,dict(center_insertions=len(repaired),center_insertion_indices=repaired,
                  all_identical_bandwidth_limit=sigma==0,sigma=sigma,
                  minimum_degree=float(np.min(np.asarray(s.sum(1)))))


def mvchss_graphs(views, positive, negative, *, p=5, mu=.3, seed=0):
    if not views or any(x.shape[0]!=views[0].shape[0] for x in views):
        raise ValueError('Aligned views required')
    z = mv.constraint_matrix(views[0].shape[0],positive,negative)
    rng = np.random.default_rng(seed); graphs=[]; reports=[]
    for x in views:
        hypergraph,report = centered_hypergraph(x,p)
        graphs.append(sparse_mv.author_propagation(hypergraph,z,mu,rng))
        reports.append(report)
    return graphs,reports


def das_indices(views,count):
    if not views or not isinstance(count,(int,np.integer)) or not 1<=count<=views[0].shape[0]:
        raise ValueError('Aligned views and valid anchor count required')
    n=views[0].shape[0]; score=np.zeros(n)
    for x in views:
        x=sparse.csr_matrix(x,dtype=float) if sparse.issparse(x) else np.asarray(x,float)
        entries=x.data if sparse.issparse(x) else x
        if x.ndim!=2 or x.shape[0]!=n or not np.isfinite(entries).all() or (entries.size and entries.min()<0):
            raise ValueError('Finite aligned nonnegative views required')
        score+=np.asarray(x.sum(1)).ravel()
    selected=[]; available=np.ones(n,dtype=bool); zero_rounds=[]; tied_rounds=[]
    for step in range(count):
        remaining=np.flatnonzero(available); values=score[remaining]
        maximum=float(values.max())
        if np.all(values==values[0]): tied_rounds.append(step+1)
        index=int(remaining[np.argmax(values)])  # first unselected index in a tie
        selected.append(index);available[index]=False
        if maximum>0:
            # Same grouping, sparse sums and logistic arithmetic as frozen DAS.
            score=score/maximum;score*=1-score
        else:
            assert maximum==0 and np.min(score)>=0
            zero_rounds.append(step+1)
    return np.asarray(selected,dtype=int),dict(zero_score_fill_rounds=zero_rounds,
        all_remaining_tied_rounds=tied_rounds,distinct_anchors=len(set(selected)))


def row_weights(row,neighbors):
    """SFMC Eq32 simplex QP; Eq33 when positive adaptive bandwidth exists."""
    row=np.asarray(row,float)
    if row.ndim!=1 or not np.isfinite(row).all() or row.min()<0 or not 1<=neighbors<len(row):
        raise ValueError('Finite nonnegative distances and K+1 anchors required')
    order=np.argsort(row,kind='stable'); di=row[order[:neighbors+1]]
    denominator=neighbors*di[-1]-sum(di[:-1])
    legacy=np.zeros_like(row)
    if denominator>0:
        legacy[order[:neighbors+1]]=(di[-1]-di)/(denominator+np.finfo(float).eps)
        if np.isfinite(legacy).all() and legacy.min()>=0 and np.isclose(legacy.sum(),1,atol=1e-10,rtol=0):
            return legacy,dict(path='unchanged_legacy_feasible',gamma=float(np.sum(np.maximum(di[-1]-row,0))/2))
    gaps=np.maximum(di[-1]-row,0); delta=float(gaps.sum())
    if delta>0:
        result=gaps/delta
        path='positive_bandwidth_stable_simplex_normalization'
    else:
        minimum=row==row.min();result=minimum.astype(float)/np.sum(minimum)
        path='zero_bandwidth_uniform_minimizers'
    assert result.min()>=0 and np.isclose(result.sum(),1,atol=1e-10,rtol=0)
    return result,dict(path=path,gamma=delta/2)


def anchor_graph(x,indices,neighbors):
    x=sparse.csr_matrix(x,dtype=float) if sparse.issparse(x) else np.asarray(x,float)
    indices=np.asarray(indices,dtype=int);anchor=x[indices]
    if indices.ndim!=1 or not len(indices) or len(np.unique(indices))!=len(indices) or indices.min()<0 or indices.max()>=x.shape[0]:
        raise ValueError('Invalid anchor indices')
    norms=np.asarray(x.multiply(x).sum(1)).ravel() if sparse.issparse(x) else np.sum(x*x,1)
    gram=x@anchor.T;gram=gram.toarray() if sparse.issparse(gram) else gram
    distance=np.maximum(norms[:,None]+norms[indices][None,:]-2*gram,0)
    graph=np.zeros_like(distance);counts={};maximum_residual=0.
    for i,row in enumerate(distance):
        graph[i],report=row_weights(row,neighbors)
        counts[report['path']]=counts.get(report['path'],0)+1
        # Diagnostic simplex-QP first-order conditions at gamma=delta/2.
        gradient=row+2*report['gamma']*graph[i]; active=graph[i]>0
        dual=float(np.mean(gradient[active]));scale=max(1.,float(abs(gradient).max()))
        residual=max(float(np.max(abs(gradient[active]-dual))),
                     float(np.max(np.maximum(dual-gradient[~active],0),initial=0))) / scale
        maximum_residual=max(maximum_residual,residual)
    return graph,dict(row_paths=counts,maximum_scaled_QP_KKT_residual=maximum_residual,
                      maximum_row_mass_error=float(np.max(abs(graph.sum(1)-1))))


def pcbg_graphs(views,anchor_count,neighbors):
    normalized=[normalize(x,norm='l2',axis=1) for x in views]
    indices,das=das_indices(normalized,anchor_count);graphs=[];reports=[]
    for x in normalized:
        graph,report=anchor_graph(x,indices,neighbors);graphs.append(graph);reports.append(report)
    return graphs,indices,dict(das=das,views=reports)
