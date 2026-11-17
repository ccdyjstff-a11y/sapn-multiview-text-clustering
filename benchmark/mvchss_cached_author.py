"""Cache repeated products in the frozen MVCHSS author-map implementation.

The map, random draws, arithmetic grouping, iteration count and restart policy
are unchanged. This module is usable only after numerical qualification against
mvchss_reimplementation.fit_graphs. It never receives class labels.
"""
from pathlib import Path
import sys
import numpy as np

MAIN = Path(__file__).resolve().parent.parent
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))
from mvchss_reimplementation import FitResult


def cached_objective(factors, consensus, products, degrees, norms, alpha, beta):
    total = 0.
    for v, sv, degree, norm in zip(factors, products, degrees, norms):
        lv = degree * v - sv
        total += norm - 2 * np.sum(v * sv) + np.sum((v.T @ v) ** 2)
        total += alpha * np.sum(v * lv) + beta * np.sum((v - consensus) ** 2)
    return float(total)


def fit_graphs(graphs, k, *, alpha, beta, seed=0, max_iter=300, profile='eq20'):
    if not graphs or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(graphs[0]):
        raise ValueError('Invalid graphs or class count')
    n = len(graphs[0])
    graphs = [np.asarray(s, float) for s in graphs]
    if any(s.shape != (n, n) or not np.isfinite(s).all() or np.min(s) < 0 or
           not np.allclose(s, s.T, atol=1e-12, rtol=1e-12) for s in graphs):
        raise ValueError('Require finite nonnegative symmetric affinities')
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) < 0 or max_iter < 1:
        raise ValueError('Invalid coefficients or schedule')
    if profile not in ('eq20', 'author_normalized'):
        raise ValueError('Unknown solver profile')
    rng = np.random.default_rng(seed)

    def init():
        v = rng.random((n, k))
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-15)

    factors, consensus = [init() for _ in graphs], init()
    degrees = [s.sum(1)[:, None] for s in graphs]
    norms = [np.sum(s * s) for s in graphs]
    products = [s @ v for s, v in zip(graphs, factors)]
    history = [cached_objective(factors, consensus, products, degrees, norms, alpha, beta)]
    for _ in range(max_iter):
        updated = []
        for v, sv, degree in zip(factors, products, degrees):
            numerator = (2 + alpha) * sv + beta * consensus
            # Retain the original left-associative multiplication exactly.
            denominator = 2 * v @ (v.T @ v) + alpha * degree * v + beta * v
            out = v * numerator / np.maximum(denominator, 1e-10)
            if profile == 'author_normalized':
                out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-15)
            updated.append(out)
        factors = updated
        consensus = np.mean(factors, axis=0)
        products = [s @ v for s, v in zip(graphs, factors)]
        history.append(cached_objective(factors, consensus, products, degrees, norms, alpha, beta))
        if not np.isfinite(history[-1]):
            raise FloatingPointError('Nonfinite objective')
    return FitResult(factors, consensus, consensus.argmax(1), np.asarray(history),
                     profile, max_iter, 'fixed_schedule_not_kkt_certified')
