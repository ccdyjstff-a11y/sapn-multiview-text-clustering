"""Sparse incidence and propagation backend for the frozen AUTHOR graph recipe.

Keeps all 30 random-initialized propagation passes, normalization and outputs.
Only intermediate hypergraph products use sparsity. This is not a new model.
Numerical equivalence must be checked before any larger benchmark use.
"""
import numpy as np
from scipy import sparse
import mvchss_reimplementation as base


def author_hypergraph(x, p=5):
    x = sparse.csr_matrix(x, dtype=float) if sparse.issparse(x) else np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x.data if sparse.issparse(x) else x).all():
        raise ValueError('Finite two-dimensional features required')
    n = x.shape[0]
    if not isinstance(p, (int, np.integer)) or not 0 <= p < n:
        raise ValueError('Require 0 <= p < number of samples')
    norms = np.asarray(x.multiply(x).sum(1)).ravel() if sparse.issparse(x) else np.sum(x*x, axis=1)
    gram = x@x.T
    gram = gram.toarray() if sparse.issparse(gram) else gram
    d2 = np.maximum(norms[:, None]+norms[None, :]-2*gram, 0)
    dist = np.sqrt(d2)
    sigma = float(dist.mean())
    if sigma <= 0:
        raise ValueError('Author bandwidth undefined for identical samples')
    members = np.argsort(dist, axis=1, kind='stable')[:, :p+1]
    weights = np.exp(-d2[np.arange(n)[:, None], members]/sigma**2).sum(1)
    h = sparse.csr_matrix((np.ones(n*(p+1)),
                          (np.repeat(np.arange(n), p+1), members.ravel())), shape=(n, n))
    s = h.T@h.multiply((weights/(p+1))[:, None])
    return ((s+s.T)*.5).tocsr()


def author_propagation(s, z, mu, rng, steps=30):
    s = sparse.csr_matrix(s, dtype=float)
    z = np.asarray(z, dtype=float)
    if s.shape != z.shape or s.shape[0] != s.shape[1]:
        raise ValueError('Square S and Z of equal shape required')
    if not 0 <= mu < 1 or steps < 1:
        raise ValueError('Require 0 <= mu < 1 and positive passes')
    degree = np.asarray(s.sum(1)).ravel()
    if np.any(degree <= 0):
        raise ValueError('Positive degrees required by author normalization')
    inverse = 1/np.sqrt(degree)
    t = s.multiply(inverse[:, None]).multiply(inverse[None, :]).tocsr()
    fv, fh = 2*rng.random(s.shape)-1, 2*rng.random(s.shape)-1
    for _ in range(steps):
        fv = mu*(t@fv)+(1-mu)*z
    for _ in range(steps):
        fh = mu*(t.T@fh.T).T+(1-mu)*fv
    dense = s.toarray()
    diagonal = s.diagonal()[:, None]
    revised = np.where(fh >= 0, diagonal-(1-fh)*(diagonal-dense), (1+fh)*dense)
    diag = np.diag(revised)
    if np.any(diag <= 0):
        raise ValueError('Revised diagonal is not positive')
    revised /= np.sqrt(diag[:, None]*diag[None, :])
    revised = (revised+revised.T)/2
    if not np.isfinite(revised).all() or np.min(revised) < 0:
        raise ValueError('Author propagation produced invalid nonnegative affinity; no clipping applied')
    return revised


def build_author_graphs(views, positive, negative, *, p=5, mu=.3, seed=0):
    if not views or any(x.shape[0] != views[0].shape[0] for x in views):
        raise ValueError('Require aligned nonempty views with samples in rows')
    z = base.constraint_matrix(views[0].shape[0], positive, negative)
    rng = np.random.default_rng(seed)
    return [author_propagation(author_hypergraph(x, p), z, mu, rng) for x in views]
