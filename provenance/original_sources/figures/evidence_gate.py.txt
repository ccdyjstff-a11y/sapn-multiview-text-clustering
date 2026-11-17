"""Require result PASS records bound to the actual manuscript evidence bytes."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parent
ROUND = ROOT.parent / 'final_improvement_v5'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def require_mechanisms():
    folder=ROUND/'mechanisms'
    record=read(folder/'RESULTS_READY.json')
    assert record.get('ready') is True and record.get('status')=='PASS'
    fields={'audit_sha256':'INDEPENDENT_RESULTS_AUDIT.json',
        'summary_sha256':'summary.json','per_run_sha256':'per_run.csv',
        'protocol_sha256':'GLOBAL_PROTOCOL.json',
        'global_seal_sha256':'GLOBAL_PREDICTIONS_SEALED.json',
        'weight_diagnostics_sha256':'WEIGHT_DIAGNOSTICS.json'}
    for key,name in fields.items():
        assert record[key]==sha(folder/name),(key,name)
    audit=read(folder/'INDEPENDENT_RESULTS_AUDIT.json')
    assert audit['status']=='PASS'
    for key,name in [('evaluation_sha256','evaluation.json'),('summary_sha256','summary.json'),('protocol_sha256','GLOBAL_PROTOCOL.json')]:
        assert audit[key]==sha(folder/name)
    return dict(ready_sha256=sha(folder/'RESULTS_READY.json'),records=read(folder/'summary.json')['logical_records'])


def require_baseline():
    folder=ROUND/'baseline_diagnostics'
    record=read(folder/'RESULTS_READY.json')
    assert record['status']=='PASS'
    assert record['files']
    for path,digest in record['files'].items():
        p=Path(path)
        assert p.resolve().is_relative_to(folder.resolve())
        assert sha(p)==digest,p
    assert read(folder/'FINAL_NUMERIC_AUDIT.json')['status']=='PASS'
    return dict(ready_sha256=sha(folder/'RESULTS_READY.json'),records=record['budget_state_count'])


def require_reproduction():
    folder=ROUND/'reproduction'
    record=read(folder/'RESULTS_READY.json')
    assert record['status']=='PASS'
    assert record['bundle_manifest_sha256']==sha(folder/'bundle/BUILD_MANIFEST.json')
    manifest=read(folder/'bundle/BUILD_MANIFEST.json')
    for name,digest in manifest['files'].items():
        p=(folder/'bundle'/name).resolve()
        assert p.is_relative_to((folder/'bundle').resolve())
        assert sha(p)==digest,name
    for name,digest in record['bound_evidence'].items():
        assert sha(folder/name)==digest,name
    return dict(ready_sha256=sha(folder/'RESULTS_READY.json'),bundle_manifest_sha256=record['bundle_manifest_sha256'])


def require_posthoc():
    folder=ROUND/'mechanisms/posthoc_no_correction_fixed_lambda'
    record=read(folder/'RESULTS_READY.json')
    assert record['status']=='PASS' and record['posthoc'] is True
    fields={'protocol_sha256':'PROTOCOL.json','seal_sha256':'PREDICTIONS_SEALED.json',
        'summary_sha256':'summary.json','evaluation_sha256':'evaluation.json',
        'per_run_sha256':'per_run.csv','audit_sha256':'AUDIT.json'}
    for key,name in fields.items(): assert record[key]==sha(folder/name),(key,name)
    score=read(folder/'INDEPENDENT_SCORE_CHECK.json')
    assert score['status']=='PASS'
    assert record['independent_score_check_sha256']==sha(folder/'INDEPENDENT_SCORE_CHECK.json')
    assert record['independent_checker_source_sha256']==sha(folder/'independent_score_check.py')
    return dict(ready_sha256=sha(folder/'RESULTS_READY.json'),
        independent_score_check_sha256=sha(folder/'INDEPENDENT_SCORE_CHECK.json'))


def require_all():
    return dict(mechanisms=require_mechanisms(),baseline=require_baseline(),reproduction=require_reproduction(),posthoc=require_posthoc())


if __name__=='__main__': print(json.dumps(require_all(),indent=2))
