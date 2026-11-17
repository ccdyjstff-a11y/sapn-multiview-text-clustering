"""Prepare/validate a draft; freezing is a separate explicit future action."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import shutil
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
S1 = ROOT.parent / 'ssa_formal_v1'
CORPORA = ['3Sources', 'BBC', 'Cora', 'CiteSeer']
EXECUTION_ORDER = '3Sources Q400,Q160; BBC Q400,Q160; Cora Q400,Q160; CiteSeer Q400,Q160; three explicit seeds23,24,25 consecutively per group; fold-major within condition'
OWN_INHERITED = ['ssa_formal_worker.m', 'ssa_clock_bridge.m', 'phase_clock.py', 'owned_processes.py', 'query_protocol.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.m')) + list((ROOT/'private_generated').glob('*.m'))
    return {str(path.resolve()): sha(path) for path in sorted(files)}


def group_spec(datasets):
    return [dict(corpus=ds,Q=q,seeds=[23,24,25],n=datasets[ds]['n'])
        for ds in CORPORA for q in [400,160]]


def prepare():
    if not __debug__:raise RuntimeError('Assertions must remain enabled for protocol checks')
    from successor_gate import source_bindings
    assert not (ROOT/'PROTOCOL_DRAFT.json').exists(), 'Do not overwrite draft identities'
    original = read(S1/'FROZEN_PROTOCOL.json')
    runtime = read(S1/'RUNTIME_PROTOCOL.json')
    p = json.loads(json.dumps(original))
    groups = group_spec(p['datasets'])
    indexed = {(c['dataset'], c['Q'], c['seed']):c for c in p['conditions']}
    supported = [indexed[(g['corpus'],g['Q'],seed)] for g in groups for seed in g['seeds']]
    unsupported = [c for c in p['conditions'] if c['unsupported']]
    assert len(supported)==24 and len(unsupported)==18 and all(not c['unsupported'] for c in supported)
    assert all(len(c['eligible_folds'])==2 for c in supported)
    p['conditions'] = supported + unsupported
    (ROOT/'inputs').mkdir()
    for case in p['cases']:
        source = Path(case['input'])
        assert source.parent.resolve() == (S1/'inputs').resolve()
        assert sha(source) == case['sha256']
        target = ROOT/'inputs'/source.name
        shutil.copyfile(source,target)
        assert sha(target)==case['sha256']
        case['input']=str(target.resolve())
    inheritance = {}
    for name in OWN_INHERITED:
        assert sha(ROOT/name)==original['source_hashes'][str((S1/name).resolve())]
        inheritance[name]=sha(ROOT/name)
    for path in (ROOT/'private_generated').glob('*.m'):
        old = S1/'private_generated'/path.name
        assert sha(path)==original['source_hashes'][str(old.resolve())]
        inheritance['private_generated/'+path.name]=sha(path)
    core = ROOT.parent/'runtime_transition/future_controller_core/controller_core.py'
    assert sha(core)==sha(ROOT/'controller_core.py')
    p.update(status='DRAFT_NOT_FROZEN_NOT_EXECUTED', batch='SSA_GROUPED_V2', groups=groups,
        prior_batch_prepared_utc=original['prepared_utc'],prepared_utc=datetime.now(timezone.utc).isoformat(),
        prior_batch_protocol_sha256=sha(S1/'FROZEN_PROTOCOL.json'),
        prior_runtime_protocol_sha256=sha(S1/'RUNTIME_PROTOCOL.json'),
        prior_batch_results_imported=False, fixed_input_reuse_only=True,
        author_root=str((ROOT/'private_generated').resolve()), source_hashes=sources(),
        reviewed_closure_sources=source_bindings(),
        inherited_exact_source_sha256=inheritance,
        core_source_sha256=sha(core),
        execution_order=EXECUTION_ORDER,
        group_memory_gate=dict(small_n_below=2000,small_minimum_GiB=3,large_minimum_GiB=4.5,
            continuous_qualified_seconds=60,total_wait_cap_seconds=1800),
        resource_stop=dict(consecutive_required_launcher_resource_caps=2,
            counted_statuses=['RESOURCE_START_WAIT_CAP','RESOURCE_SUSPEND_WAIT_CAP'],
            skipped_slots_do_not_increment_or_reset=True,completed_conditions_preserved=True),
        local_caps=['WORKER_ACTIVE_TIME_CAP','RESOURCE_START_WAIT_CAP','RESOURCE_SUSPEND_WAIT_CAP'],
        completed_semantics='An attempt lifecycle has terminated and owned cleanup is proven; not optimizer convergence or a completed native algorithm.',
        short_circuit='Only irreversible local ineligibility skips later folds of the same candidate; trigger_case and original receipt retained. No BER-based pruning.',
        stop_request=dict(file='STOP_AFTER_CONDITION.json',schema=1,request='STOP_AFTER_CONDITION',
            identity_fields=['run_id','protocol_sha256','runtime_sha256'],
            acknowledge_only_after_atomic_complete_condition_commit=True),
        no_implicit_resume=True,
        resource_scope='New group-memory qualification and persistent-resource-stop policy. Same per-fit caps and one numerical thread; not the same realized computational budget and no measured speedup claim.',
        evidence_hashes={str((S1/'FROZEN_PROTOCOL.json').resolve()):sha(S1/'FROZEN_PROTOCOL.json'),
            str((S1/'RUNTIME_PROTOCOL.json').resolve()):sha(S1/'RUNTIME_PROTOCOL.json'),
            **{path:digest for path,digest in original['evidence_hashes'].items() if 'WORKER_PREPARATION' not in path}})
    assert len(p['cases'])==1296 and len(p['base_inputs'])==72
    exclusive(ROOT/'PROTOCOL_DRAFT.json',p)
    r = dict(original_scientific_protocol_sha256=sha(ROOT/'PROTOCOL_DRAFT.json'),
        bound_files={**p['source_hashes'],**p['reviewed_closure_sources']},bridge_evidence=runtime['bridge_evidence'],
        global_wall_seconds=48*3600,previous_batch_root=str(S1.resolve()),
        previous_batch_closure=None,closure_verifier_status='REQUIRED_BEFORE_FREEZE_AND_EXECUTION')
    exclusive(ROOT/'RUNTIME_DRAFT.json',r)
    print(json.dumps(dict(status='DRAFT_PREPARED_NOT_FROZEN',protocol_sha256=sha(ROOT/'PROTOCOL_DRAFT.json'),
        runtime_sha256=sha(ROOT/'RUNTIME_DRAFT.json'),copied_small_requests=1296,shared_large_inputs=72)))


def refresh_draft_sources():
    """Explicit draft-only source rebinding; preserve prior bytes, never fit/resume."""
    if not __debug__:raise RuntimeError('Assertions must remain enabled')
    from successor_gate import source_bindings
    assert not any((ROOT/name).exists() for name in ['FROZEN_PROTOCOL.json','RUNTIME_PROTOCOL.json','EXECUTION_STARTED.json'])
    p=read(ROOT/'PROTOCOL_DRAFT.json');r=read(ROOT/'RUNTIME_DRAFT.json')
    for name,digest in p['inherited_exact_source_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'controller_core.py')==p['core_source_sha256']
    expected_old_groups=[dict(corpus=ds,Q=q,seeds=[23,24,25],n=p['datasets'][ds]['n'])
        for q in [400,160] for ds in CORPORA]
    expected_new_groups=group_spec(p['datasets'])
    assert p['groups'] in [expected_old_groups,expected_new_groups], 'Only the explicitly declared RAM-based order amendment is permitted'
    indexed={(c['dataset'],c['Q'],c['seed']):c for c in p['conditions']}
    assert len(indexed)==len(p['conditions'])==42
    supported=[indexed[(g['corpus'],g['Q'],seed)] for g in expected_new_groups for seed in g['seeds']]
    unsupported=[c for c in p['conditions'] if c['unsupported']]
    assert len(supported)==24 and len(unsupported)==18 and all(not c['unsupported'] for c in supported)
    for case in p['cases']:assert sha(case['input'])==case['sha256']
    prior=sha(ROOT/'PROTOCOL_DRAFT.json')
    history=ROOT/'draft_history';history.mkdir(exist_ok=True)
    for name in ['PROTOCOL_DRAFT.json','RUNTIME_DRAFT.json']:
        path=ROOT/name;target=history/(sha(path)+'_'+name)
        if not target.exists():
            with target.open('xb') as stream:stream.write(path.read_bytes())
        assert sha(target)==sha(path)
    p['prior_batch_prepared_utc']=p.get('prior_batch_prepared_utc',p['prepared_utc'])
    p['prepared_utc']=datetime.now(timezone.utc).isoformat()
    p['previous_draft_sha256']=prior;p['source_hashes']=sources()
    p['groups']=expected_new_groups;p['conditions']=supported+unsupported
    p['execution_order']=EXECUTION_ORDER
    p['reviewed_closure_sources']=source_bindings()
    from run_grouped import atomic
    atomic(ROOT/'PROTOCOL_DRAFT.json',p)
    r['original_scientific_protocol_sha256']=sha(ROOT/'PROTOCOL_DRAFT.json')
    r['bound_files']={**p['source_hashes'],**p['reviewed_closure_sources']}
    atomic(ROOT/'RUNTIME_DRAFT.json',r)
    print(json.dumps(dict(status='DRAFT_SOURCES_REBOUND_NOT_FROZEN',previous_draft_sha256=prior,
        protocol_sha256=sha(ROOT/'PROTOCOL_DRAFT.json'),runtime_sha256=sha(ROOT/'RUNTIME_DRAFT.json'))))


def validate(expected):
    if not __debug__:raise RuntimeError('Assertions must remain enabled for protocol checks')
    from successor_gate import source_bindings
    assert sha(ROOT/'PROTOCOL_DRAFT.json')==expected
    p,r=read(ROOT/'PROTOCOL_DRAFT.json'),read(ROOT/'RUNTIME_DRAFT.json')
    assert r['original_scientific_protocol_sha256']==expected and p['source_hashes']==sources()
    assert p['reviewed_closure_sources']==source_bindings()
    assert r['bound_files']=={**p['source_hashes'],**p['reviewed_closure_sources']}
    for path,digest in p['reviewed_closure_sources'].items():assert sha(path)==digest,path
    for path,digest in p['evidence_hashes'].items():
        assert sha(path)==digest,path
    for path,digest in p['base_inputs'].items():
        assert sha(path)==digest,path
    for path,digest in p['query_sources'].items():
        assert sha(path)==digest,path
    for case in p['cases']:
        assert Path(case['input']).parent.resolve()==(ROOT/'inputs').resolve()
        assert sha(case['input'])==case['sha256']
        assert p['base_inputs'][case['source_input']]==case['source_input_sha256']
    assert p['prior_batch_results_imported'] is False
    return dict(passed=True,protocol_sha256=expected,runtime_sha256=sha(ROOT/'RUNTIME_DRAFT.json'),
        no_matlab_invoked=True,no_truth_read=True,scope='Streaming byte identity and declared mappings; full paid/schema checks run at actual entry.')


def freeze(expected,gate):
    from successor_gate import verify,source_bindings,REVIEWED_SOURCE_SHA256
    assert gate and all(gate.values()),'Reviewed closure verifier and exact closure evidence SHA required'
    result=validate(expected)
    gate=dict(gate,verifier_source_sha256=dict(REVIEWED_SOURCE_SHA256))
    p=read(ROOT/'PROTOCOL_DRAFT.json')
    closed=verify(S1,p['prior_batch_protocol_sha256'],gate)
    assert not any((ROOT/name).exists() for name in ['FROZEN_PROTOCOL.json','RUNTIME_PROTOCOL.json','FREEZE_RECEIPT.json','EXECUTION_STARTED.json'])
    from run_grouped import atomic
    frozen_bytes=(ROOT/'PROTOCOL_DRAFT.json').read_bytes()
    # Preserve the exact scientific draft bytes; a pre-existing file is refused.
    with (ROOT/'FROZEN_PROTOCOL.json').open('xb') as stream:stream.write(frozen_bytes)
    runtime=read(ROOT/'RUNTIME_DRAFT.json')
    runtime['previous_batch_closure']=gate;runtime['closure_check_at_freeze']=closed
    runtime['bound_files'].update(source_bindings())
    atomic(ROOT/'RUNTIME_PROTOCOL.json',runtime,exclusive=True)
    runtime_sha=sha(ROOT/'RUNTIME_PROTOCOL.json')
    result['draft_runtime_sha256']=result.pop('runtime_sha256')
    result['runtime_sha256']=runtime_sha
    atomic(ROOT/'FREEZE_RECEIPT.json',dict(protocol_sha256=expected,runtime_sha256=runtime_sha,preflight=result),exclusive=True)
    print(json.dumps(dict(status='FROZEN_NOT_EXECUTED',protocol_sha256=expected,runtime_sha256=runtime_sha)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true');group.add_argument('--validate-draft-sha');group.add_argument('--freeze-draft-sha')
    group.add_argument('--refresh-draft-sources',action='store_true')
    parser.add_argument('--closure-verifier');parser.add_argument('--closure-verifier-sha')
    parser.add_argument('--closure');parser.add_argument('--closure-sha')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.refresh_draft_sources:refresh_draft_sources()
    elif args.validate_draft_sha:print(json.dumps(validate(args.validate_draft_sha)))
    elif args.freeze_draft_sha:freeze(args.freeze_draft_sha,dict(verifier_path=args.closure_verifier,
        verifier_sha256=args.closure_verifier_sha,closure_path=args.closure,closure_sha256=args.closure_sha))
    else:print('READ_ONLY_PREPARATION: no data opened, no protocol frozen, no process launched')
