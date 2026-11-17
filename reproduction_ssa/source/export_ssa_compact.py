"""Original-workspace projection into a portable SSA evidence packet.

Explicit actual report SHA required. No model calls, class vectors or scored
performance records are decoded. This exporter requires SciPy and original MAT
paths; the separate replay entry requires only NumPy and packet-relative paths.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import io
import json
import shutil
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R10 = HERE.parent
FORMAL = R10 / 'ssa_grouped_v2'
SCORED = R10 / 'ssa_grouped_reporting'
REPORT = SCORED / 'result_packet'
PAID_MEMBERS = ('pool.npy', 'test.npy', 'pairs.npy', 'answers.npy')
OWN_SOURCE = ('export_ssa_compact.py', 'replay_ssa_compact.py', 'portable_primitives.py',
    'ledger_validation.py', 'receipt_policy.py', 'test_grouped_compact.py', 'README.md')
ROOT_EVIDENCE = ('FROZEN_PROTOCOL.json','RUNTIME_PROTOCOL.json','FREEZE_RECEIPT.json',
    'EXECUTION_STARTED.json','EXECUTION_ENDED.json','CONTROLLER_LEDGER.json',
    'CV_OUTCOMES.json','FINAL_OUTCOMES.json','SELECTIONS.json')


def scheduler_member(name):
    return (name in ROOT_EVIDENCE or name in {'STOP_AFTER_CONDITION.json','STOP_REQUEST_ACKNOWLEDGED.json'}
        or (name.endswith('.json') and name.split('/')[0] in {'guards','dispatches','decisions','conditions'})
        or (name.startswith('runs/') and name.endswith('/repeat1/RUN_IDENTITY.json')))


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def bound(path, digest):
    need(sha(path) == digest, 'SHA mismatch: ' + str(path))


def child(base, relative):
    target = (Path(base) / relative.replace('\\', '/')).resolve()
    need(target.is_relative_to(Path(base).resolve()), 'Member escapes packet: ' + relative)
    return target


def array_identity(array):
    import numpy as np
    a = np.asarray(array)
    need(not a.dtype.hasobject and not a.dtype.fields, 'Only plain numeric arrays')
    header = json.dumps(dict(dtype=a.dtype.str, shape=list(a.shape)), sort_keys=True,
        separators=(',', ':')).encode('ascii')
    return dict(dtype=a.dtype.str, shape=list(a.shape),
        sha256=hashlib.sha256(header + b'\n' + a.tobytes(order='C')).hexdigest())


def write_npz(path, members):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, raw)


def export_array(array, target, member):
    import numpy as np
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    write_npz(target, {member: payload})
    identity = array_identity(array)
    return dict(original_array=identity, exported_array=identity,
        npy_member_sha256=hashlib.sha256(payload).hexdigest(), npy_member_byte_length=len(payload))


def project_paid(source, target):
    bound(source['path'], source['sha256'])
    with zipfile.ZipFile(source['path']) as archive:
        names = archive.namelist()
        need(all(names.count(name) == 1 for name in PAID_MEMBERS), 'Missing/duplicate paid member')
        payload = {name: archive.read(name) for name in PAID_MEMBERS}
    bound(source['path'], source['sha256'])
    write_npz(target, payload)
    return {name: dict(sha256=hashlib.sha256(raw).hexdigest(), byte_length=len(raw)) for name, raw in payload.items()}


def admit(report_sha,audit_sha):
    """Use the existing actual report/post-score binding, without another gate."""
    bound(REPORT / 'REPORT_MANIFEST.json', report_sha)
    report = read(REPORT / 'REPORT_MANIFEST.json')
    need(report['status'] == 'AUDIT_GATED_COMPARISON_REPORT', 'Expected completed audit-gated report')
    need(report.get('batch')=='SSA_GROUPED_V2' and report.get('prior_S1_results_imported') is False,
        'Only independent S2 reports can admit this export')
    need(report['logical_rows'] == 42 and report['corpus_query_groups'] == 14, 'Wrong report scope')
    for key, path in [('protocol_sha256', FORMAL / 'FROZEN_PROTOCOL.json'),
        ('runtime_protocol_sha256', FORMAL / 'RUNTIME_PROTOCOL.json'),
        ('global_seal_sha256', SCORED / 'GLOBAL_PREDICTIONS_SEALED.json'),
        ('evaluation_manifest_sha256', SCORED / 'EVALUATION_MANIFEST.json')]:
        bound(path, report[key])
    post_path = Path(report['post_score_audit_path'])
    need(report['post_score_audit_sha256']==audit_sha,'Explicit audit SHA differs from report')
    bound(post_path, report['post_score_audit_sha256'])
    post = read(post_path)
    need(post.get('passed') is True and post.get('kind') == 'POST_SCORE'
        and post.get('all42_rows_checked') is True and post.get('all14_groups_checked') is True,
        'Actual post-score audit incomplete')
    for field in ['protocol_sha256', 'global_seal_sha256', 'evaluation_manifest_sha256', 'evaluation_sha256', 'summary_sha256']:
        need(post[field] == report[field], 'Post-score/report identity mismatch: ' + field)
    need(post['runtime_sha256'] == report['runtime_protocol_sha256'], 'Post-score runtime mismatch')
    seal = read(SCORED / 'GLOBAL_PREDICTIONS_SEALED.json')
    need(seal['status'] == 'SEALED' and seal['logical_outcomes'] == 42 and seal['CV_outcomes'] == 864, 'Wrong seal scope')
    need(seal.get('batch')=='SSA_GROUPED_V2' and seal.get('prior_batch_results_imported') is False,'S2 seal required')
    need(post.get('all_CV_and_selection_records_recollected') is True,'S2 collector audit incomplete')
    need(seal['protocol_sha256'] == report['protocol_sha256'] and seal['runtime_sha256'] == post['runtime_sha256'], 'Wrong seal identities')
    scoring = read(SCORED / 'EVALUATION_MANIFEST.json')
    need(scoring['status'] == 'SCORED_PENDING_INDEPENDENT_REVIEW'
        and scoring['global_seal_sha256'] == report['global_seal_sha256'], 'Scoring boundary differs')
    for key in ['evaluation_sha256', 'summary_sha256']:
        need(scoring[key] == report[key], 'Scoring/export identity differs')
    for name, digest in seal['members_sha256'].items():
        bound(child(FORMAL, name), digest)
    need(not (FORMAL / 'CONTROLLER_FAILURE.json').exists() and not (FORMAL / 'UNRESOLVED_OWNED_PROCESSES.json').exists(),
        'Unresolved execution cannot be exported')
    finish = read(FORMAL / 'EXECUTION_ENDED.json')
    need(finish['status'] in {'COMPLETED','SAFE_STOP_AT_CONDITION_BOUNDARY','EXECUTION_HALTED'}, 'Training is not terminal')
    need(finish['no_class_scoring'] is True,'Invalid S2 training boundary')
    protocol=read(FORMAL/'FROZEN_PROTOCOL.json')
    need(protocol['batch']=='SSA_GROUPED_V2' and protocol['prior_batch_results_imported'] is False,'S1 results cannot be reused')
    need(sha(HERE/'ledger_validation.py')==seal['reporting_source_sha256']['ledger_validation.py'],
        'Portable structural checker differs from actual sealed collector')
    need(sha(HERE/'receipt_policy.py')==protocol['source_hashes'][str((FORMAL/'adapter_policy.py').resolve())],
        'Portable receipt policy differs from frozen S2')
    return report, post_path, seal


def export(report_sha,audit_sha):
    # SciPy is needed only for original MAT projection, not portable replay.
    import numpy as np
    from scipy.io import loadmat
    report, post_path, seal = admit(report_sha,audit_sha)
    protocol = read(FORMAL / 'FROZEN_PROTOCOL.json')
    cv, final, selections = (read(FORMAL / name) for name in ['CV_OUTCOMES.json', 'FINAL_OUTCOMES.json', 'SELECTIONS.json'])
    need(len(cv) == 864 and len(final) == 42 and len(selections) == 24, 'Incomplete logical ledgers')
    need(len(protocol['cases']) == 1296 and len(protocol['grid']) == 18, 'Unexpected formal profile')
    cases = {c['case_id']: c for c in protocol['cases']}
    need(len(cases) == 1296 and len({r['condition_id'] for r in final}) == 42, 'Duplicate case/condition')
    normalized_seal = {name.replace('\\', '/'): digest for name, digest in seal['members_sha256'].items()}
    need(len(normalized_seal) == len(seal['members_sha256']), 'Ambiguous sealed path aliases')
    def sealed(name):
        need(name in normalized_seal, 'Source is not sealed: ' + name)
        return normalized_seal[name]
    out = HERE / 'result_packet'
    need(not out.exists(), 'Never overwrite a partial or completed compact export')
    source_files = {name: HERE / name for name in OWN_SOURCE}
    need(all(path.is_file() for path in source_files.values()), 'Portable own-source closure incomplete')
    out.mkdir()
    for name in ['evidence', 'receipts', 'partitions', 'consensus', 'paid', 'source','scheduler']:
        (out / name).mkdir()
    scheduler={}
    for name in ROOT_EVIDENCE:
        bound(FORMAL / name, sealed(name))
        shutil.copyfile(FORMAL / name, out / 'evidence' / name)
        scheduler[name]=dict(file='evidence/'+name,sha256=sealed(name),original_absolute_path=str((FORMAL/name).resolve()))
    for name,digest in normalized_seal.items():
        if scheduler_member(name) and name not in scheduler:
            target=child(out,'scheduler/'+name);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(child(FORMAL,name),target)
            scheduler[name]=dict(file='scheduler/'+name,sha256=digest,original_absolute_path=str(child(FORMAL,name)))
    extra = [(SCORED / 'GLOBAL_PREDICTIONS_SEALED.json', 'GLOBAL_PREDICTIONS_SEALED.json'),
        (post_path, 'POST_SCORE_AUDIT.json'), (SCORED / 'EVALUATION_MANIFEST.json', 'EVALUATION_MANIFEST.json'),
        (REPORT / 'REPORT_MANIFEST.json', 'REPORT_MANIFEST.json')]
    for source, name in extra:
        shutil.copyfile(source, out / 'evidence' / name)
    public_source = []
    for name, path in source_files.items():
        relative = 'source/' + name
        shutil.copyfile(path, out / relative)
        public_source.append(relative)
    streams = []
    for dataset, card in protocol['datasets'].items():
        for seed, source in card['query_inputs'].items():
            relative = f'paid/{dataset}_s{seed}.npz'
            target = child(out, relative)
            payloads = project_paid(source, target)
            streams.append(dict(dataset=dataset, seed=int(seed), n=card['n'], source=source,
                projection=dict(file=relative, sha256=sha(target)), members=payloads))
    need(len(streams) == len({(s['dataset'], s['seed']) for s in streams}) == 21, 'All 21 original streams required')
    write(out / 'PAID_STREAMS.json', streams)
    mappings = {}
    source_checks = dict(partition_arrays_compared=0, final_consensus_arrays_compared=0, paid_NPY_payloads_preserved=84)
    for phase, rows in [('CV', cv), ('FINAL', final)]:
        records = []
        for row in rows:
            case_id = row.get('case_id')
            record = dict(outcome=row, source=None, partition=None, consensus=None,
                consensus_status='not_applicable_CV' if phase == 'CV' else 'no_retained_partition')
            if case_id is not None:
                need(case_id in cases and '/' not in case_id and '\\' not in case_id, 'Unsafe or unknown case')
                case = cases[case_id]
                record['source'] = dict(case_id=case_id, input_sha256=case['sha256'], n=case['n'],
                    receipt=None, native_mat=None)
            if row['attempted']:
                need(case_id is not None, 'Attempt lacks case identity')
                prefix = f'runs/{case_id}/repeat1/'
                receipt_member = prefix + 'launcher_receipt.json'
                digest = sealed(receipt_member)
                need(digest == row['receipt_sha256'], 'Ledger receipt hash differs')
                original = child(FORMAL, receipt_member)
                bound(original, digest)
                receipt = read(original)
                need(receipt['case_id'] == case_id and receipt['input_sha256'] == case['sha256'], 'Receipt case differs')
                relative = f'receipts/{case_id}.json'
                shutil.copyfile(original, out / relative)
                record['source']['receipt'] = dict(file=relative, sha256=digest, original_member=receipt_member)
                scheduler[receipt_member]=dict(file=relative,sha256=digest,original_absolute_path=str(original))
                if 'native_outputs.mat' in receipt['output_hashes']:
                    mat_member = prefix + 'native_outputs.mat'
                    mat_sha = sealed(mat_member)
                    need(receipt['output_hashes']['native_outputs.mat'] == mat_sha, 'MAT receipt/seal differs')
                    record['source']['native_mat'] = dict(original_member=mat_member, sha256=mat_sha, field='label')
            if row.get('finite_partition_retained'):
                need(row['attempted'] and record['source']['native_mat'] is not None, 'Unbound finite partition')
                mat = record['source']['native_mat']
                path = child(FORMAL, mat['original_member'])
                bound(path, mat['sha256'])
                names = ['label', 'V_star'] if phase == 'FINAL' else ['label']
                arrays = loadmat(path, variable_names=names)
                bound(path, mat['sha256'])
                labels = np.asarray(arrays['label'])
                need(labels.dtype.kind in 'iuf' and labels.size == case['n'] and np.isfinite(labels).all()
                    and np.all(labels >= 1) and np.all(labels == np.floor(labels)), 'Invalid retained finite partition')
                relative = f'partitions/{case_id}.npz'
                identity = export_array(labels, out / relative, 'partition.npy')
                record['partition'] = dict(file=relative, sha256=sha(out / relative), member='partition.npy', **identity)
                with np.load(out / relative, allow_pickle=False) as exported:
                    need(array_identity(exported['partition']) == array_identity(labels), 'Partition projection altered array')
                source_checks['partition_arrays_compared'] += 1
                if phase == 'FINAL':
                    need('V_star' in arrays, 'Retained final output lacks its consensus field')
                    consensus = np.asarray(arrays['V_star'])
                    if consensus.dtype.kind in 'iufc' and consensus.shape == (case['n'], case['c']):
                        relative = f'consensus/{case_id}.npz'
                        identity = export_array(consensus, out / relative, 'V_star.npy')
                        record['consensus'] = dict(file=relative, sha256=sha(out / relative), member='V_star.npy', **identity)
                        record['source']['final_consensus_field'] = 'V_star'
                        record['consensus_status'] = 'exported'
                        with np.load(out / relative, allow_pickle=False) as exported:
                            need(array_identity(exported['V_star']) == array_identity(consensus), 'Consensus projection altered array')
                        source_checks['final_consensus_arrays_compared'] += 1
                    else:
                        record['consensus_status'] = 'unexpected_consensus_shape_or_dtype_not_exported'
                    del consensus
                del arrays, labels
            else:
                need(not row['eligible'], 'Eligible row lacks partition')
            records.append(record)
        mappings[phase] = records
        write(out / (phase + '_PARTITIONS.json'), records)
    counts = dict(CV_arrays=sum(r['partition'] is not None for r in mappings['CV']),
        final_arrays=sum(r['partition'] is not None for r in mappings['FINAL']),
        final_consensus_arrays=sum(r['consensus'] is not None for r in mappings['FINAL']))
    need(source_checks['partition_arrays_compared'] == counts['CV_arrays'] + counts['final_arrays'], 'Partition source-check count differs')
    write(out/'SCHEDULER_MEMBERS.json',scheduler)
    manifest = dict(kind='SSA_COMPACT_EXPORT', status='EXPORTED_WITH_SOURCE_IDENTITY_CHECKS',
        schema=2,batch='SSA_GROUPED_V2',prior_S1_results_imported=False,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), exporter_sha256=sha(Path(__file__)),
        protocol_sha256=report['protocol_sha256'], runtime_sha256=report['runtime_protocol_sha256'],
        global_seal_sha256=report['global_seal_sha256'], post_score_audit_sha256=report['post_score_audit_sha256'],
        evaluation_manifest_sha256=report['evaluation_manifest_sha256'], report_manifest_sha256=report_sha,
        evaluation_sha256=report['evaluation_sha256'], summary_sha256=report['summary_sha256'],
        counts=counts, logical_counts=dict(CV=864, final=42, paid_streams=21), selection_records=24,
        source_identity_checks=source_checks, public_source=public_source,
        implementation_source_bindings=dict(ledger_validation=sha(HERE/'ledger_validation.py'),receipt_policy=sha(HERE/'receipt_policy.py')),
        array_identity_rule='SHA256(canonical compact sorted ASCII JSON {dtype,shape} + LF + C-order array bytes)',
        eligibility_authority='Original sealed receipts and actual root post-score audit. Compact arrays do not reconstruct dense-state qualification.',
        no_model_calls=True, no_oracle_arrays_read=True, no_evaluation_metrics_decoded=True,
        consensus_scope='Only readable final consensus fields associated with retained finite partitions; no CV consensus arrays. Nonfinite final consensus values, if present, remain diagnostic.',
        excluded=['raw feature matrices', 'full class vectors', 'original MAT files', 'dense fitted states',
            'private_generated', 'author_source_private_analysis', 'author algorithm source', 'MATLAB preferences/software'],
        members_sha256={str(path.relative_to(out)).replace('\\', '/'): sha(path)
            for path in sorted(out.rglob('*')) if path.is_file()})
    # Bind output evidence to the exact source identities once more, without
    # altering any frozen source metadata or replacing its absolute provenance.
    for field, name in [('protocol_sha256', 'FROZEN_PROTOCOL.json'), ('runtime_sha256', 'RUNTIME_PROTOCOL.json'),
        ('global_seal_sha256', 'GLOBAL_PREDICTIONS_SEALED.json'), ('post_score_audit_sha256', 'POST_SCORE_AUDIT.json'),
        ('evaluation_manifest_sha256', 'EVALUATION_MANIFEST.json'), ('report_manifest_sha256', 'REPORT_MANIFEST.json')]:
        need(manifest[field] == manifest['members_sha256']['evidence/' + name], 'Copied provenance identity differs')
    write(out / 'EXPORT_MANIFEST.json', manifest)
    print(json.dumps(dict(status=manifest['status'], manifest_sha256=sha(out / 'EXPORT_MANIFEST.json'), counts=counts,
        replay_performed=False, full_training_reproduced=False), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-report-sha')
    parser.add_argument('--audit-sha')
    args = parser.parse_args()
    if not args.execute_report_sha:
        need(not args.audit_sha,'Exact report SHA is also required')
        print('READ_ONLY_PREPARATION: no result arrays, query streams or performance records opened')
        return
    need(args.audit_sha,'Explicit exact POST_SCORE audit SHA required')
    export(args.execute_report_sha,args.audit_sha)


if __name__ == '__main__':
    main()
