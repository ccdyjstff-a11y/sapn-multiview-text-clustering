"""Prepare AG News from the verified fastai mirror using the unchanged builder.

The mirror container differs from the historical author archive. All four regular
members must match the verified original members before preparation can begin.
This adapter does not train or evaluate any clustering model.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import tarfile
import time

import numpy as np
from scipy import sparse


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_archive(path, identity):
    require(sha(path) == identity['mirror_archive_sha256'], 'Mirror archive SHA256 mismatch')
    require(path.stat().st_size == identity['mirror_archive_bytes'], 'Mirror archive size mismatch')
    found = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.isdir():
                require(member.name.rstrip('/') == 'ag_news_csv', 'Unexpected archive directory')
                continue
            require(member.isfile(), 'Links and special archive members are not accepted')
            require(member.name not in found, 'Duplicate regular archive member')
            require(member.name in identity['regular_members'], 'Unrecognized regular archive member')
            value = archive.extractfile(member).read()
            found[member.name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
            require(member.size == len(value), 'Archive member size differs from its payload')
    require(found == identity['regular_members'], 'Regular member names, sizes or contents differ')
    return found


def compare_reference(matrices, labels, provenance, vocabularies, idfs, dropped, reference):
    report = {'ordered_labels_equal': bool(np.array_equal(labels, np.load(reference / 'y.npy', allow_pickle=False))),
              'ordered_source_ids_and_provenance_equal': provenance == read(reference / 'sample_provenance.json'),
              'ordered_vocabularies_equal': vocabularies == read(reference / 'vocabularies.json'),
              'idf_values_equal': idfs == read(reference / 'idf_values.json'),
              'deduplication_exclusions_equal': dropped == read(reference / 'deduplication_exclusions.json'),
              'views': []}
    for view, matrix in enumerate(matrices):
        expected = sparse.load_npz(reference / f'X{view}.npz').tocsr()
        require(matrix.shape == expected.shape, f'View {view} shape mismatch')
        difference = matrix - expected
        difference.eliminate_zeros()
        report['views'].append({'view': view, 'shape': list(matrix.shape),
            'all_ordered_values_equal': difference.nnz == 0,
            'maximum_absolute_error': float(np.max(np.abs(difference.data), initial=0)),
            'csr_indptr_equal': bool(np.array_equal(matrix.indptr, expected.indptr)),
            'csr_indices_equal': bool(np.array_equal(matrix.indices, expected.indices)),
            'csr_data_equal': bool(np.array_equal(matrix.data, expected.data))})
    report['passed'] = all(report[key] for key in report if key != 'views') and all(
        row['all_ordered_values_equal'] and row['csr_indptr_equal'] and
        row['csr_indices_equal'] and row['csr_data_equal'] for row in report['views'])
    require(report['passed'], 'Direct comparison with historical prepared inputs failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mirror-identity', type=Path,
                        default=Path(__file__).with_name('agnews_mirror_identity.json'))
    parser.add_argument('--base-preparer', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'benchmark' / 'prepare_field_corpora.py')
    parser.add_argument('--identities', type=Path,
                        help='Defaults to FIELD_CORPUS_IDENTITIES.json beside the unchanged base preparer')
    parser.add_argument('--verify-against', type=Path,
                        help='Optional original prepared directory for full elementwise verification')
    args = parser.parse_args()
    started = time.perf_counter()
    archive, output = args.archive.resolve(), args.output.resolve()
    require(not output.exists(), 'Output must be a new directory')
    identity = read(args.mirror_identity)
    member_checks = validate_archive(archive, identity)
    base_path = args.base_preparer.resolve()
    require(sha(base_path) == identity['unchanged_base_preparer_sha256'], 'Original preparer bytes changed')
    field_identity_path = (args.identities.resolve() if args.identities else
                           base_path.with_name('FIELD_CORPUS_IDENTITIES.json'))
    require(sha(field_identity_path) == identity['field_identities_sha256'], 'Frozen field identity file changed')
    spec = importlib.util.spec_from_file_location('unchanged_field_preparer', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': base.scipy.__version__, 'scikit-learn': base.sklearn.__version__}
    require(versions == base.VERSIONS, 'Use the original pinned numerical environment')
    expected = read(field_identity_path)['datasets']['agnews']
    require(expected['source_archive_sha256'] == identity['equivalent_original_archive_sha256'],
            'Original archive identities disagree')
    # Reuse the exact original construction function; its CLI archive gate is not
    # invoked. This wrapper supplies a stricter mirror-and-member gate instead.
    with base.threadpool_limits(limits=1):
        matrices, labels, provenance, vocabularies, idfs, extra, dropped = base.prepare_agnews(archive)
    scientific = base.scientific_identity(matrices, labels, provenance, vocabularies)
    require(scientific == expected['scientific_identity'], 'Frozen scientific identity mismatch')
    require(sha(archive) == identity['mirror_archive_sha256'], 'Archive changed during preparation')
    require(sha(base_path) == identity['unchanged_base_preparer_sha256'], 'Original preparer changed during use')
    direct = None
    if args.verify_against:
        direct = compare_reference(matrices, labels, provenance, vocabularies, idfs, dropped,
                                   args.verify_against.resolve())
    metadata = dict(extra, n=len(labels), k=len(np.unique(labels)), views=len(matrices),
                    dimensions=[x.shape[1] for x in matrices], class_counts=np.bincount(labels),
                    nonzeros=[int(x.nnz) for x in matrices],
                    zero_rows=[int(np.sum(np.asarray(x.multiply(x).sum(axis=1)).ravel() == 0))
                               for x in matrices], prepared_sha256=scientific['prepared_sha256'],
                    archive_sha256=identity['mirror_archive_sha256'],
                    archive_bytes=identity['mirror_archive_bytes'], source_url=identity['mirror_url'],
                    equivalent_original_archive_sha256=identity['equivalent_original_archive_sha256'],
                    archive_container_byte_identical_to_original=False,
                    all_regular_members_byte_identical_to_original=True,
                    mirror_regular_member_validation=member_checks,
                    preparation_source_sha256=sha(__file__),
                    unchanged_base_preparer_sha256=sha(base_path), versions=versions,
                    isolated=bool(sys.flags.isolated),
                    training_labels='y.npy is only for query oracle and sealed scoring, not a model input',
                    note='The actual mirror container SHA is recorded. Only member contents and scientific outputs equal the original.',
                    raw_texts_exported=False, clustering_training_performed=False,
                    clustering_evaluation_performed=False)
    for key, value in expected['historical_metadata_identity'].items():
        require(base.json_value(metadata[key]) == value, 'Historical scientific metadata differs: ' + key)
    output.mkdir(parents=True)
    for view, matrix in enumerate(matrices):
        sparse.save_npz(output / f'X{view}.npz', matrix)
    np.save(output / 'y.npy', labels, allow_pickle=False)
    for name, value in [('sample_provenance', provenance), ('vocabularies', vocabularies),
                        ('idf_values', idfs), ('deduplication_exclusions', dropped), ('metadata', metadata)]:
        base.dump(output / (name + '.json'), value)
    verification = dict(status='PASS', actual_archive_sha256=sha(archive),
                        equivalent_original_archive_sha256=identity['equivalent_original_archive_sha256'],
                        regular_members_exact=True, scientific_identity=scientific,
                        scientific_identity_exact=True, historical_scientific_metadata_equal=True,
                        direct_reference_comparison=direct,
                        base_preparer_unchanged=True, base_preparer_sha256=sha(base_path),
                        identity_manifest_sha256=sha(field_identity_path),
                        mirror_identity_sha256=sha(args.mirror_identity),
                        adapter_sha256=sha(__file__), versions=versions,
                        isolated=bool(sys.flags.isolated), numerical_thread_limit=1,
                        elapsed_seconds=time.perf_counter() - started, trained_models=0,
                        file_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    base.dump(output / 'PREPARATION_VERIFICATION.json', verification)
    print(json.dumps({'status': 'PASS', 'dataset': 'agnews', 'archive_sha256': sha(archive),
                      'prepared_sha256': scientific['prepared_sha256'],
                      'dimensions': metadata['dimensions'], 'isolated': bool(sys.flags.isolated),
                      'direct_reference_comparison_passed': direct['passed'] if direct else None,
                      'trained_models': 0}), flush=True)


if __name__ == '__main__':
    main()
