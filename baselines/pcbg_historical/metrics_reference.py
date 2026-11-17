"""Exact metrics function extracted from the original scorer; no model imports."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score

def metrics(y,z):
    if not len(y):return dict(ACC=None,NMI=None,ARI=None)
    _,a=np.unique(y,return_inverse=True);_,b=np.unique(z,return_inverse=True)
    table=np.zeros((a.max()+1,b.max()+1),int);np.add.at(table,(a,b),1);r,c=linear_sum_assignment(-table)
    return dict(ACC=float(table[r,c].sum()/len(y)),NMI=float(normalized_mutual_info_score(y,z)),ARI=float(adjusted_rand_score(y,z)))
