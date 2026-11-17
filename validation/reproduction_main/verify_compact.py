"""Verify the exported real reproduction; optionally compare a reader's fresh fit.

This read-only verifier runs no model and reads no original corpus or class vector.
It is a portable derivative of the recorded local comparison script, not its bytes.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[1]
IGNORE={'seconds','identity_sha256'}
METRICS=['ACC','NMI','ARI','BER_train','BER_unqueried']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def comparable(value):
    if isinstance(value,dict):return {k:comparable(v) for k,v in value.items() if k not in IGNORE}
    if isinstance(value,list):return [comparable(v) for v in value]
    return value
def arrays_equal(left,right):
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        assert a.files==b.files,(left.name,'keys')
        for key in a.files:assert np.array_equal(a[key],b[key]),(left.name,key)
        return len(a.files)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fresh-fit',type=Path,help='Optional reader-generated seed504/Q160 output to compare with retained compact predictions')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new report')
    export=read(HERE/'COMPACT_EXPORT.json')
    for r in export['records']:
        base=HERE if r['base']=='reproduction_main' else RELEASE
        assert sha(base/r['file'])==r['sha256'],r['file']
    receipt=read(HERE/'VERIFICATION.json');protocol=read(HERE/'PROTOCOL.json')
    assert receipt['status']=='PASS' and receipt['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    assert receipt['script_sha256']==sha(RELEASE/'provenance/original_sources/main_instance/verify_main_instance.py.txt')
    for name,h in protocol['sources'].items():assert sha(RELEASE/name)==h,name
    seal=read(HERE/'fit/GLOBAL_PREDICTIONS_SEALED.json')
    present=[]
    for name,h in seal['files'].items():
        if (HERE/'fit'/name).exists():assert sha(HERE/'fit'/name)==h,name;present.append(name)
    old=[r for r in read(HERE/'historical_reference/evaluation.json') if r['budget']==160]
    new=read(HERE/'fit/evaluation.json');assert len(new)==len(old)==5
    for row in new:
        previous=next(r for r in old if r['method']==row['method'])
        assert comparable(row)==comparable(previous),row['method']
        for metric in METRICS:assert row[metric]==previous[metric]
    mapping=read(RELEASE/'evidence/v9/allocation_repeat/PREDICTION_MAP.json')['records']
    references=[r for r in mapping if r['source_record']['dataset']=='3Sources' and r['source_record']['seed']==504 and r['source_record']['budget']==160]
    assert len(references)==5
    for record in references:
        s=record['source_record'];f=HERE/'fit'/(s['stem']+'.npz')
        assert sha(f)==record['prediction_sha256']==sha(RELEASE/record['prediction_member'])
        arrays_equal(f,RELEASE/record['prediction_member'])
    historical_selections=RELEASE/'evidence/v9/allocation_repeat/runs/3Sources/seed504'
    for f in (HERE/'fit').glob('*_selection.json'):
        assert comparable(read(f))==comparable(read(historical_selections/f.name)),f.name
    fresh_arrays=0;fresh_diagnostics=0
    if a.fresh_fit:
        for f in (HERE/'fit').glob('Q160*.npz'):fresh_arrays+=arrays_equal(f,a.fresh_fit/f.name)
        for f in (HERE/'fit').glob('Q160*_diagnostics.json'):
            assert comparable(read(f))==comparable(read(a.fresh_fit/f.name)),f.name;fresh_diagnostics+=1
        for f in (HERE/'fit').glob('*_selection.json'):assert comparable(read(f))==comparable(read(a.fresh_fit/f.name))
        assert comparable(read(HERE/'fit/evaluation.json'))==comparable(read(a.fresh_fit/'evaluation.json'))
    result=dict(status='PASS',kind='REAL_MAIN_COMPACT_EVIDENCE_REPLAY',exported_files_verified=len(export['records']),
        current_source_files_verified=len(protocol['sources']),included_global_seal_members_verified=len(present),
        omitted_fit_payloads=len(export['omitted_fit_payloads']),historical_logical_predictions_equal=5,
        historical_metric_values_equal=25,historical_selections_equal=3,
        reader_fresh_fit_compared=a.fresh_fit is not None,reader_fresh_array_checks=fresh_arrays,reader_fresh_diagnostics=fresh_diagnostics,
        real_training_performed_by_this_command=False,raw_labels_read=False,
        full_state_440_array_equality='Recorded in VERIFICATION.json; omitted full factors are not independently replayed here')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
