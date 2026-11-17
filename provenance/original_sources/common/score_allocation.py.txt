"""Independent scoring of a complete sealed allocation run; never refits/selects."""
from pathlib import Path
import sys,argparse
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE))
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from query_protocol import sample_query_stream,answer_queries,balanced_pair_error
from allocation_io import read,sha,dump,immutable,validate_seal,paid_inputs

def score(output,labels_file):
    out=Path(output).resolve();labels_file=Path(labels_file).resolve()
    # Every requested final/CV outcome, identity, input, graph and source is checked
    # before the first class-vector load. No post-score candidate selection occurs.
    p,records=validate_seal(out)
    inp=paid_inputs(out/'inputs.npz',p['n'],p['budgets']);test=inp['test']
    y=np.load(labels_file,allow_pickle=False).reshape(-1)
    if len(y)!=p['n'] or len(np.unique(y))!=p['k']:raise ValueError('Class vector must match declared n and known k')
    ep=sample_query_stream(test,min(10000,len(test)*(len(test)-1)//2),p['seed']+51000)
    ea=answer_queries(ep,y)
    pairs_path=out/'postseal_evaluation_pairs.npz'
    if pairs_path.exists():
        with np.load(pairs_path,allow_pickle=False) as z:assert np.array_equal(z['pairs'],ep) and np.array_equal(z['answers'],ea)
    else:np.savez_compressed(pairs_path,pairs=ep,answers=ea)
    evaluated=[]
    for r in records:
        if r['status']=='ok':
            with np.load(out/(r['stem']+'.npz'),allow_pickle=False) as z:labels=z['labels'].copy()
            yt,zt=y[test],labels[test]
            _,a=np.unique(yt,return_inverse=True);_,b=np.unique(zt,return_inverse=True)
            table=np.zeros((a.max()+1,b.max()+1),int);np.add.at(table,(a,b),1)
            i,j=linear_sum_assignment(-table)
            metrics=dict(ACC=float(table[i,j].sum()/len(yt)),NMI=float(normalized_mutual_info_score(yt,zt)),
                ARI=float(adjusted_rand_score(yt,zt)),BER_train=balanced_pair_error(labels,inp['pairs'][:r['budget']],inp['answers'][:r['budget']]),
                BER_unqueried=balanced_pair_error(labels,ep,ea))
        else:metrics={key:None for key in ['ACC','NMI','ARI','BER_train','BER_unqueried']}
        evaluated.append(dict(r,**metrics))
    immutable(out/'evaluation.json',evaluated)
    immutable(out/'EVALUATION_MANIFEST.json',dict(global_seal_sha256=sha(out/'GLOBAL_PREDICTIONS_SEALED.json'),
        labels_sha256=sha(labels_file),evaluation_sha256=sha(out/'evaluation.json'),test_count=len(test),
        evaluation_pairs_sha256=sha(pairs_path),pair_count=len(ep),scoring_source_sha256=sha(Path(__file__)),
        eligible=sum(r['converged'] for r in evaluated),logical_records=len(records),
        interpretation='Fixed original selections and Full-lambda controls. Terminal scores for nonconverged runs are retained with eligibility false. No oracle coefficient selection.'))
    print('SCORED',len(records),'sealed logical finals',flush=True)
    return evaluated

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--labels',required=True)
    args=parser.parse_args();score(args.output,args.labels)
