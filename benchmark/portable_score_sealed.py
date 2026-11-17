"""Score only globally sealed portable-runner predictions; separate from training."""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from portable_budget_runner import read,dump,sha

def main(out,labels,queries):
    out=Path(out).resolve();labels=Path(labels).resolve();queries=Path(queries).resolve()
    seal=read(out/'PREDICTIONS_SEALED.json');protocol=read(out/'PROTOCOL.json');records=read(out/'runs.json')
    assert sha(out/'PROTOCOL.json')==seal['protocol_sha256'] and sha(out/'runs.json')==seal['runs_sha256']
    assert len(records)==seal['count']==2*len(protocol['budgets'])
    assert {(r['method'],r['total_queries']) for r in records}=={(m,Q) for m in protocol['methods'] for Q in protocol['budgets']}
    assert sha(queries)==protocol['input_hashes'][str(queries)]
    for name,digest in protocol['source_hashes'].items():assert sha(Path(__file__).resolve().parent/name)==digest
    assert sha(out/'used_initialization.npz')==seal['initialization_sha256']
    for name,digest in seal['graph_hashes'].items():assert sha(out/(name+'_graph.npz'))==digest
    with np.load(queries,allow_pickle=False) as f:pool=f['pool'].copy();pairs=f['pairs'].copy()
    test=np.setdiff1d(np.arange(protocol['n']),pool);assert not np.intersect1d(test,pairs).size
    # Load class IDs only after prediction/protocol seals have been verified.
    y=np.load(labels,allow_pickle=False).reshape(-1);assert len(y)==protocol['n'] and len(test)>0
    result=[]
    for r in records:
        assert sha(out/(r['stem']+'.npz'))==r['prediction_sha256']
        assert sha(out/(r['stem']+'_state.npz'))==r['state_sha256']
        assert sha(out/(r['stem'].removesuffix('_final')+'_selection.json'))==r['selection_sha256']
        with np.load(out/(r['stem']+'.npz'),allow_pickle=False) as f:
            z=f['labels'].copy();assert np.array_equal(z,f['H'].argmax(0)) and z.shape==y.shape
        yt,zt=y[test],z[test];_,a=np.unique(yt,return_inverse=True);_,b=np.unique(zt,return_inverse=True)
        table=np.zeros((a.max()+1,b.max()+1),int);np.add.at(table,(a,b),1);i,j=linear_sum_assignment(-table)
        result.append(dict(r,ACC=float(table[i,j].sum()/len(yt)),NMI=float(normalized_mutual_info_score(yt,zt)),ARI=float(adjusted_rand_score(yt,zt))))
    dump(out/'evaluation.json',result)
    dump(out/'EVALUATION_MANIFEST.json',dict(prediction_seal_sha256=sha(out/'PREDICTIONS_SEALED.json'),labels_sha256=sha(labels),query_file_sha256=sha(queries),evaluation_sha256=sha(out/'evaluation.json'),scoring_source_sha256=sha(Path(__file__)),test_nodes=len(test)))
    print('SCORED sealed predictions',len(result),'test nodes',len(test),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--labels',required=True);p.add_argument('--queries',required=True);a=p.parse_args();main(a.output,a.labels,a.queries)
