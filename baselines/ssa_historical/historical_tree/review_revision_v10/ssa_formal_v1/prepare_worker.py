"""Prepare the declared SSA finite-grid worker; never launch MATLAB or load labels."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
QUAL = HERE.parent / 'ssa_minimal_qualification' / 'private_generated'
OLD = HERE.parent.parent / 'review_revision_v9/baseline_gap/native_formal_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one literal source match: ' + old)
    return text.replace(old, new)

def main():
    target = HERE / 'private_generated'
    if target.exists() or (HERE / 'WORKER_PREPARATION.json').exists():
        raise FileExistsError('Do not overwrite a prepared worker profile')
    target.mkdir()
    source = QUAL / 'ssa_minimal_profile.m'
    if sha(source) != '899cd281b7f4e201d84ad4a3615dc927228f8dc7e115e07dff88454de63474e2':
        raise ValueError('Unexpected qualified minimal source')
    content = source.read_text(encoding='utf-8')
    edits = [
        ('function [V_star, diagnostics, state] = ssa_minimal_profile(fea, known_k, ML, CL, seed)',
         'function [V_star, diagnostics, state] = ssa_formal_profile(fea, known_k, ML, CL, seed, config)'),
        ('kap = 0.95;', 'kap = config.kap;'),
        ('beta = 0.1; % B|S-Z|', 'beta = config.beta; % B|S-Z|'),
        ('lam = 0.2;  % manifold', 'lam = config.lam;  % manifold'),
    ]
    for old, new in edits:
        content = replace_once(content, old, new)
    (target / 'ssa_formal_profile.m').write_text(content, encoding='utf-8', newline='\n')
    names = ['Create_KNNgraph.m', 'NormalizationK.m', 'normalization.m',
             'Update_V_v.m', 'Update_W_v.m', 'Update_Teta.m']
    for name in names:
        (target / name).write_bytes((QUAL / name).read_bytes())
    cost = (QUAL / 'Cost_function.m').read_text(encoding='utf-8')
    cost = replace_once(cost, 'sum(diag(L_M{v}*S{v}))', "sum(sum(L_M{v} .* S{v}.'))")
    (target / 'Cost_function.m').write_text(cost, encoding='utf-8', newline='\n')
    bridge = (OLD / 'pcbg_clock_bridge.m').read_text(encoding='utf-8')
    bridge = bridge.replace('pcbg_clock_bridge', 'ssa_clock_bridge').replace('pcbg_formal_worker', 'ssa_formal_worker')
    bridge = bridge.replace('% Runtime instrumentation only. The original worker and author code are unchanged.',
                            '% Runtime phase instrumentation around the explicitly adapted SSA worker.')
    (HERE / 'ssa_clock_bridge.m').write_text(bridge, encoding='utf-8', newline='\n')
    files = [HERE / 'prepare_worker.py', HERE / 'ssa_formal_worker.m', HERE / 'ssa_clock_bridge.m'] + sorted(target.glob('*.m'))
    manifest = {
        'status': 'PREPARED_NOT_EXECUTED',
        'profile': 'SSA-SNMF-minimal-two-repair-author-profile',
        'implementation': 'finite beta/lam/kap grid; trace scalar evaluation adaptation',
        'source_profile_sha256': sha(source),
        'source_cost_sha256': sha(QUAL / 'Cost_function.m'),
        'source_bridge_sha256': sha(OLD / 'pcbg_clock_bridge.m'),
        'profile_literal_edits': [{'old': a, 'new': b} for a, b in edits],
        'cost_literal_edit': {'old': 'sum(diag(L_M{v}*S{v}))', 'new': "sum(sum(L_M{v} .* S{v}.'))"},
        'byte_identical_non_cost_helpers': {name: sha(QUAL / name) for name in names},
        'files': {str(p.relative_to(HERE)): sha(p) for p in files},
        'matlab_entry': 'ssa_clock_bridge(requestMat, outputDir, privateGeneratedRoot, runId)',
        'request_fields': ['source_input', 'source_sha256', 'config', 'case_id'],
        'finite_parameter_values': {'beta': [.1, .2, .3], 'lam': [.2, .3], 'kap': [.8, .9, .95]},
        'source_mat_loaded_fields_only': ['X', 'c', 'ML', 'CL', 'seed'],
        'source_hash_verification': 'The parent Python launcher must bind and verify source_input SHA before and after every fit; the MATLAB worker records this binding without duplicate large-file hashing.',
        'private_analysis_only': True,
        'full_class_truth_loaded': False,
        'matlab_started': False,
        'bridge_replay_required_before_execution': True,
    }
    (HERE / 'WORKER_PREPARATION.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'manifest_sha256': sha(HERE / 'WORKER_PREPARATION.json'), 'files': len(files)}))

if __name__ == '__main__':
    main()
