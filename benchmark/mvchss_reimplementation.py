"""Auditable MVCHSS building blocks, not an untouched MATLAB reproduction.

Primary source: DOI 10.1109/TCSVT.2023.3258926, Eqs. 12, 20, 22.
Author graph recipe: Psy680267/MVCHSS commit e02341fe1ff0b713d5cc40fffee95d6b4de1b6d7.
Inputs contain features, known k, and observed pairs only. No class labels.
Graphs are dense O(n^2); this module does not launch benchmark jobs.

Two solver profiles share the AUTHOR graph construction, which differs from
paper Eq. 6. 'eq20' omits the author's additional row normalization;
'author_normalized' retains it. Both repair the empty objective history.
Neither profile is claimed to reproduce MATLAB random numbers or stopping.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse


AUTHOR_COMMIT = 'e02341fe1ff0b713d5cc40fffee95d6b4de1b6d7'


def constraint_matrix(n, positive, negative):
    """Observed unordered pairs plus cost-free self identity, as author code."""
    z = np.eye(n)
    seen = set()
    for pairs, sign in ((positive, 1), (negative, -1)):
        raw = np.asarray(pairs)
        if raw.size == 0:
            continue
        if raw.ndim != 2 or raw.shape[1] != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Pairs must be integer (m, 2) arrays')
        for a, b in raw:
            a, b = int(a), int(b)
            if a == b or min(a, b) < 0 or max(a, b) >= n:
                raise ValueError('Invalid pair endpoint')
            key = tuple(sorted((a, b)))
            if key in seen:
                raise ValueError('Duplicate or contradictory queried pair')
            seen.add(key)
            z[a, b] = z[b, a] = sign
    return z


def author_hypergraph(x, p=5):
    """Rows are samples; p+1 nearest entries; center-to-member weights.

    Tie rule: stable sample index, matching repeated first minimum selection.
    Self is ordinarily included but is not forced when duplicate points tie.
    Degenerate zero bandwidth is rejected rather than silently changing it.
    """
    x = sparse.csr_matrix(x, dtype=float) if sparse.issparse(x) else np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x.data if sparse.issparse(x) else x).all():
        raise ValueError('Finite two-dimensional features required')
    n = x.shape[0]
    if not isinstance(p, (int, np.integer)) or not 0 <= p < n:
        raise ValueError('Require 0 <= p < number of samples')
    norms = np.asarray(x.multiply(x).sum(1)).ravel() if sparse.issparse(x) else np.sum(x*x, axis=1)
    gram = x @ x.T
    gram = gram.toarray() if sparse.issparse(gram) else gram
    d2 = np.maximum(norms[:, None] + norms[None, :] - 2*gram, 0)
    dist = np.sqrt(d2)
    sigma = float(dist.mean())
    if sigma <= 0:
        raise ValueError('Author bandwidth undefined for identical samples')
    members = np.argsort(dist, axis=1, kind='stable')[:, :p+1]
    h = np.zeros((n, n))  # edges by vertices, as MHPCP.m
    h[np.arange(n)[:, None], members] = 1
    affinity = np.exp(-d2 / sigma**2)
    weights = np.sum(affinity * h.T, axis=0)
    s = h.T @ ((weights / h.sum(1))[:, None] * h)
    return (s+s.T)/2


def author_propagation(s, z, mu, rng, steps=30):
    """Retain random Fv/Fh initialization and fixed 30 passes of MHPCP.m."""
    s, z = np.asarray(s, float), np.asarray(z, float)
    if s.shape != z.shape or s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError('Square S and Z of equal shape required')
    if not 0 <= mu < 1 or steps < 1:
        raise ValueError('Require 0 <= mu < 1 and positive passes')
    degree = s.sum(1)
    if np.any(degree <= 0):
        raise ValueError('Positive degrees required by author normalization')
    t = s / np.sqrt(degree[:, None]*degree[None, :])
    fv, fh = 2*rng.random(s.shape)-1, 2*rng.random(s.shape)-1
    for _ in range(steps):
        fv = mu*(t @ fv)+(1-mu)*z
    for _ in range(steps):
        fh = mu*(fh @ t)+(1-mu)*fv
    diagonal = np.diag(s)[:, None]
    revised = np.where(fh >= 0, diagonal-(1-fh)*(diagonal-s), (1+fh)*s)
    diag = np.diag(revised)
    if np.any(diag <= 0):
        raise ValueError('Revised diagonal is not positive')
    revised = revised / np.sqrt(diag[:, None]*diag[None, :])
    revised = (revised+revised.T)/2  # explicit author symmetrization
    if not np.isfinite(revised).all() or np.min(revised) < 0:
        raise ValueError('Author propagation produced invalid nonnegative affinity; no clipping applied')
    return revised


def build_author_graphs(views, positive, negative, *, p=5, mu=.3, seed=0):
    """Pair-only adapter; one random stream across views as in author loop."""
    if not views or any(x.shape[0] != views[0].shape[0] for x in views):
        raise ValueError('Require aligned nonempty views with samples in rows')
    z = constraint_matrix(views[0].shape[0], positive, negative)
    rng = np.random.default_rng(seed)
    return [author_propagation(author_hypergraph(x, p), z, mu, rng) for x in views]


def objective(graphs, factors, consensus, alpha, beta):
    total = 0.
    for s, v in zip(graphs, factors):
        lv = s.sum(1)[:, None]*v - s@v
        # Exact reassociation avoids constructing V V' for every iteration.
        total += np.sum(s*s)-2*np.sum(v*(s@v))+np.sum((v.T@v)**2)
        total += alpha*np.sum(v*lv)+beta*np.sum((v-consensus)**2)
    return float(total)


def gradient(s, v, consensus, alpha, beta):
    return 4*(v@(v.T@v)-s@v)+2*alpha*(s.sum(1)[:, None]*v-s@v)+2*beta*(v-consensus)


def update(s, v, consensus, alpha, beta, profile):
    if profile not in ('eq20', 'author_normalized'):
        raise ValueError('Unknown solver profile')
    numerator = (2+alpha)*(s@v)+beta*consensus
    denominator = 2*v@(v.T@v)+alpha*s.sum(1)[:, None]*v+beta*v
    out = v*numerator/np.maximum(denominator, 1e-10)
    if profile == 'author_normalized':
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-15)
    return out


@dataclass
class FitResult:
    factors: list
    consensus: np.ndarray
    labels: np.ndarray
    objective_history: np.ndarray
    profile: str
    iterations: int
    stop_reason: str


def fit_graphs(graphs, k, *, alpha, beta, seed=0, max_iter=300, profile='eq20'):
    """Fixed-schedule single start; no accuracy-based restart selection.

    Alpha/beta and iteration schedule are REQUIRED experiment protocol choices.
    A completed schedule is not reported as KKT convergence. Objective increases
    are retained, not discarded. Ground truth is never accepted by this API.
    """
    if not graphs or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(graphs[0]):
        raise ValueError('Invalid graphs or class count')
    n = len(graphs[0])
    graphs = [np.asarray(s, float) for s in graphs]
    if any(s.shape != (n,n) or not np.isfinite(s).all() or np.min(s) < 0 or
           not np.allclose(s, s.T, atol=1e-12, rtol=1e-12) for s in graphs):
        raise ValueError('Require finite nonnegative symmetric affinities')
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) < 0 or max_iter < 1:
        raise ValueError('Invalid coefficients or schedule')
    if profile not in ('eq20', 'author_normalized'):
        raise ValueError('Unknown solver profile')
    rng = np.random.default_rng(seed)
    def init():
        v = rng.random((n,k))
        return v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-15)
    factors, consensus = [init() for _ in graphs], init()
    history = [objective(graphs, factors, consensus, alpha, beta)]
    for _ in range(max_iter):
        factors = [update(s, v, consensus, alpha, beta, profile) for s, v in zip(graphs, factors)]
        consensus = np.mean(factors, axis=0)
        history.append(objective(graphs, factors, consensus, alpha, beta))
        if not np.isfinite(history[-1]):
            raise FloatingPointError('Nonfinite objective')
    return FitResult(factors, consensus, consensus.argmax(1), np.asarray(history),
                     profile, max_iter, 'fixed_schedule_not_kkt_certified')
