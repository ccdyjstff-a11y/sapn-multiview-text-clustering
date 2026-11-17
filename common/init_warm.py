# -*- coding: utf-8 -*-
"""Aligned warm start, historical route-A description.
Fit an NMF representation W_v per view, align normalized columns to a reference
using Hungarian matching, then average and normalize to initialize H.
also_hv=True additionally returns the aligned view representations.
The actual implementation below uses the first view as reference; consult the
manuscript for the frozen initialization sequence."""
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment


def _nmf(X, k, seed, max_iter=400):
    m = NMF(n_components=k, init="random", random_state=seed,
            max_iter=max_iter)
    return m.fit_transform(X)


def _align(W, ref):
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    rn = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-12)
    C = Wn.T @ rn                       # C[r,c]: correlation between view cluster r and reference cluster c.
    row, col = linear_sum_assignment(-C)
    Wa = np.zeros_like(W)
    for r, c in zip(row, col):
        Wa[:, c] = W[:, r]
    return Wa


def warm_init(Xs, k, seed, also_hv=False, max_iter=400):
    # Align to the first view NMF, using V fits and avoiding concatenated-feature NMF.
    ref = _nmf(Xs[0], k, seed, max_iter=max_iter)
    aligned = [ref]
    for X in Xs[1:]:
        aligned.append(_align(_nmf(X, k, seed, max_iter=max_iter), ref))
    H0 = np.mean(aligned, axis=0)
    H0 = H0 / (H0.sum(axis=1, keepdims=True) + 1e-12)
    if also_hv:
        hv = [(W / (W.sum(axis=1, keepdims=True) + 1e-12)).T
              for W in aligned]
        return H0.T, hv
    return H0.T
