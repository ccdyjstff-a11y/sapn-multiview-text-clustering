"""Label-blind query acquisition and node-disjoint, within-budget selection.

These helpers do not run clustering and have no dataset-specific exceptions.
The oracle is deliberately separate from sampling and fold construction.
"""
import numpy as np

def sample_query_stream(pool,total,seed):
    pool=np.asarray(pool,dtype=int)
    if pool.ndim!=1 or len(np.unique(pool))!=len(pool) or np.any(pool<0):raise ValueError('Unique nonnegative node IDs required')
    maximum=len(pool)*(len(pool)-1)//2
    if total<0 or total>maximum:raise ValueError('Budget exceeds distinct unordered pairs')
    rng=np.random.RandomState(seed);seen=set();pairs=[]
    # First occurrences in an iid uniform pair stream give a uniform ordered
    # sample without replacement, and prefixes do not depend on total budget.
    while len(pairs)<total:
        pair=tuple(sorted(int(x) for x in rng.choice(pool,2,replace=False)))
        if pair not in seen:seen.add(pair);pairs.append(pair)
    return np.asarray(pairs,dtype=int).reshape(-1,2)

def answer_queries(pairs,y):
    pairs=np.asarray(pairs,int).reshape(-1,2);y=np.asarray(y)
    if pairs.size and (pairs.min()<0 or pairs.max()>=len(y)):raise ValueError('Invalid query endpoint')
    return np.where(y[pairs[:,0]]==y[pairs[:,1]],1,-1)

def node_folds(pool,pairs,seed):
    pool=np.asarray(pool,int);pairs=np.asarray(pairs,int).reshape(-1,2)
    if len(np.unique(pool))!=len(pool) or not np.isin(pairs,pool).all():raise ValueError('Pairs must belong to a unique pool')
    shuffled=np.random.RandomState(seed).permutation(pool);halves=np.array_split(shuffled,2);folds=[]
    for held_nodes in halves:
        held=np.isin(pairs,held_nodes)
        validation=np.flatnonzero(held.all(axis=1));training=np.flatnonzero((~held).all(axis=1))
        crossing=np.flatnonzero(held.any(axis=1)&~held.all(axis=1))
        assert not np.intersect1d(pairs[training],pairs[validation]).size
        folds.append(dict(training=training,validation=validation,unused_crossing=crossing,
            validation_nodes=np.sort(held_nodes)))
    return folds

def signed_pairs(pairs,answers,indices=None):
    pairs=np.asarray(pairs,int).reshape(-1,2);answers=np.asarray(answers,int)
    if answers.shape!=(len(pairs),) or not np.isin(answers,[-1,1]).all():raise ValueError('One +/-1 answer per query')
    if indices is not None:pairs=pairs[indices];answers=answers[indices]
    return pairs[answers==1],pairs[answers==-1]

def balanced_pair_error(labels,pairs,answers):
    labels=np.asarray(labels);pairs=np.asarray(pairs,int).reshape(-1,2);answers=np.asarray(answers,int)
    if answers.shape!=(len(pairs),) or not np.isin(answers,[-1,1]).all():raise ValueError('Invalid query answers')
    if not np.any(answers==1) or not np.any(answers==-1):return None
    predicted=np.where(labels[pairs[:,0]]==labels[pairs[:,1]],1,-1)
    return float(.5*np.mean(predicted[answers==1]!=1)+.5*np.mean(predicted[answers==-1]!=-1))

def eligible_folds(folds,answers):
    answers=np.asarray(answers)
    return [f for f in folds if all(set(answers[f[part]].tolist())=={-1,1} for part in ['training','validation'])]

def select_configuration(scores,default_index=0):
    """Minimize mean validation error. Missing/failed folds invalidate a config.

    Caller must use the SAME eligible folds for every configuration/method.
    Default-first tie order prevents retrospective preferred parameter choice.
    Empty validation information falls back to declared defaults, not test ACC.
    """
    scores=np.asarray(scores,float)
    if scores.ndim!=2 or not 0<=default_index<len(scores):raise ValueError('Expected config-by-fold matrix')
    if scores.shape[1]==0:return default_index,'no_eligible_folds_default'
    means=np.where(np.all(np.isfinite(scores),axis=1),np.mean(scores,axis=1),np.inf)
    if not np.isfinite(means).any():return default_index,'all_validation_fits_failed_default_with_failure_flag'
    best=np.min(means);tied=np.flatnonzero(np.abs(means-best)<=1e-12)
    return (default_index if default_index in tied else int(tied[0])),'minimum_balanced_pair_error'
