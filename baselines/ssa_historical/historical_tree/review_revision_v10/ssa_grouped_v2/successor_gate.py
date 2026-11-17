"""Fail-closed adapter to an independently reviewed, read-only S1 closure verifier.

This module does not infer retirement from PID absence or observation snapshots.
No verifier or closure evidence is created here; absence blocks freeze/execute.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

REVIEWED_SOURCE_SHA256={
    'closure_verifier.py':'b1dabe9a131232b6ef2f05af0d93b51ddb99078f7351cf04ed5252fdacba5b40',
    'retirement_evidence.py':'7d986f02f736abcda9e5fdb1c7783c1d6012c73dbf81da48de6095147c74e387',
}
REVIEWED_DIRECTORY=Path(__file__).resolve().parent.parent/'runtime_transition/retirement_review'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def validate_result(result,prior_sha):
    if not __debug__:raise RuntimeError('Assertions must remain enabled for closure checks')
    assert isinstance(result,dict)
    assert result.get('passed') is True and result.get('fresh_process_observation') is True
    assert result.get('closure_kind') in ['NORMAL_TERMINATION','OPERATIONAL_ABORT']
    assert result.get('prior_protocol_sha256')==prior_sha
    for key in ['live_previous_batch_processes','unverified_previous_batch_processes']:
        assert type(result.get(key)) is int and result[key]==0,key
    return result


def source_bindings():
    return {str((REVIEWED_DIRECTORY/name).resolve()):digest for name,digest in REVIEWED_SOURCE_SHA256.items()}


def check_sources():
    for name,digest in REVIEWED_SOURCE_SHA256.items():
        assert sha(REVIEWED_DIRECTORY/name)==digest,'Reviewed closure source changed: '+name


def verify(s1_root,prior_sha,spec):
    if not __debug__:raise RuntimeError('Assertions must remain enabled for closure checks')
    assert isinstance(spec,dict),'An independently verified predecessor closure is required'
    assert all(spec.get(k) for k in ['verifier_path','verifier_sha256','closure_path','closure_sha256'])
    source=Path(spec['verifier_path']).resolve();closure=Path(spec['closure_path']).resolve()
    expected=REVIEWED_DIRECTORY/'closure_verifier.py'
    assert source==expected.resolve(),'Use the independently reviewed predecessor-closure entry'
    assert spec['verifier_sha256']==REVIEWED_SOURCE_SHA256['closure_verifier.py']
    assert spec.get('verifier_source_sha256')==REVIEWED_SOURCE_SHA256,'Both reviewed closure sources must be bound'
    check_sources()  # Both files checked before any verifier/dependency import.
    raw=closure.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==spec['closure_sha256'],'Closure evidence changed'
    record=json.loads(raw.decode('utf-8-sig'))
    assert record.get('verifier_source_sha256')==REVIEWED_SOURCE_SHA256,'Closure declares different verifier/dependency sources'
    definition=importlib.util.spec_from_file_location('reviewed_s1_closure_verifier',source)
    module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
    function=getattr(module,'verify_closed_for_successor',None)
    assert callable(function),'Observation-only tools are not closure verifiers'
    result=validate_result(function(str(Path(s1_root).resolve()),str(closure),spec['closure_sha256']),prior_sha)
    check_sources()  # Recheck dependency too, after the fresh process observation.
    assert sha(closure)==spec['closure_sha256'],'Closure evidence changed during verification'
    return result
