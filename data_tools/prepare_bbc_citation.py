"""Prepare BBC, Cora, or CiteSeer from explicitly supplied original inputs.

This standalone program needs only NumPy, SciPy, and its identity JSON.
It does not import research modules, construct graphs, initialize, or train.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tarfile
import zipfile

import numpy as np
import scipy
from scipy import io as sio, sparse


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(np.isfinite(matrix.data).all(), 'Nonfinite matrix value')
    require((matrix.data >= 0).all(), 'Negative matrix value')
    return matrix


def prepared_hash(matrices, labels):
    # This is the original preparation identity, including ordered CSR arrays.
    h = hashlib.sha256(np.ascontiguousarray(labels).tobytes())
    for matrix in matrices:
        h.update(str(matrix.shape).encode())
        for value in (matrix.indptr, matrix.indices, matrix.data):
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def sequence_hash(values):
    return sha_bytes(('\n'.join(values) + '\n').encode('utf-8'))


def checked_bytes(path, expected):
    value = Path(path).read_bytes()
    require(sha_bytes(value) == expected, f'Input identity mismatch: {path}')
    return value


def prepare_bbc(source, identity):
    if source.is_dir():
        raw = {name: checked_bytes(source / name, digest)
               for name, digest in identity['raw_files_sha256'].items()}
    else:
        require(source.is_file(), 'BBC --source must be the raw directory or original ZIP')
        value = checked_bytes(source, identity['archive_sha256'])
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            raw = {}
            for name, digest in identity['raw_files_sha256'].items():
                member = archive.getinfo(name)
                require(not member.is_dir(), f'Not a ZIP file member: {name}')
                raw[name] = archive.read(member)
                require(sha_bytes(raw[name]) == digest, f'ZIP member identity mismatch: {name}')
    matrix = sio.mmread(io.BytesIO(raw['bbc.mtx'])).tocsr()
    require(matrix.shape == (9635, 2225), 'Unexpected BBC term/document shape')
    docs = raw['bbc.docs'].decode('utf-8').splitlines()
    terms = raw['bbc.terms'].decode('utf-8').splitlines()
    require(len(docs) == matrix.shape[1], 'BBC document mapping length mismatch')
    require(len(set(docs)) == len(docs), 'Duplicate BBC document IDs')
    require(len(terms) == matrix.shape[0], 'BBC vocabulary length mismatch')
    labels = np.full(matrix.shape[1], -1, dtype=np.int64)
    for line in raw['bbc.classes'].decode('utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        index, category = map(int, line.split())
        require(0 <= index < len(labels), 'BBC label index outside document range')
        require(labels[index] == -1 and category >= 0, 'Duplicate/invalid BBC label')
        labels[index] = category
    require((labels >= 0).all(), 'Missing BBC label')
    classes, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    permutation = np.random.RandomState(42).permutation(matrix.shape[0])
    indices = [permutation[:4817], permutation[4817:]]
    require(not np.intersect1d(*indices).size, 'BBC feature partitions overlap')
    matrices = []
    for index in indices:
        # Match the historical document-by-feature float64 min-max arithmetic.
        dense = matrix[index, :].T.toarray().astype(float)
        lower = dense.min(axis=0)
        span = dense.max(axis=0) - lower
        dense = (dense - lower) / np.where(span > 1e-12, span, 1.)
        dense[:, span <= 1e-12] = 0.
        matrices.append(canonical(dense))
    for view, index in enumerate(indices):
        digest = sha_bytes(np.asarray(index, dtype='<i8').tobytes())
        require(digest == identity['feature_index_int64_sha256'][view],
                'BBC feature order differs from the verified seed-42 partition')
    extra = {'sample_ids': docs, 'label_original_values': classes.tolist(),
             'view_meaning': ['First seed-42 vocabulary partition',
                              'Second seed-42 vocabulary partition'],
             'preprocessing': 'Seed-42 disjoint feature split; per-feature min-max; '
                              'span <= 1e-12 becomes zero; no row normalization'}
    return matrices, labels, extra, indices, terms


def prepare_citation(dataset, source, identity):
    require(source.is_file(), 'Citation --source must be the original Grimal tar.gz')
    require(sha_file(source) == identity['archive_sha256'],
            f'{dataset} archive identity mismatch')
    raw = {}
    # Read named regular members in memory; never extract an archive to disk.
    with tarfile.open(source, 'r:gz') as archive:
        for name, digest in identity['members_sha256'].items():
            member = archive.getmember(name)
            require(member.isfile(), f'Not a regular archive member: {name}')
            value = archive.extractfile(member).read()
            require(sha_bytes(value) == digest, f'Archive member identity mismatch: {name}')
            raw[Path(name).name] = value
    tag = dataset.lower()
    matrices = [canonical(sio.mmread(io.BytesIO(raw[tag + suffix])))
                for suffix in ('_content.mtx', '_cites.mtx')]
    original = np.loadtxt(io.BytesIO(raw[tag + '_act.txt']), dtype=np.int64).ravel()
    classes, labels = np.unique(original, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    mapping = raw['documents-mapping.txt'].decode('utf-8').splitlines()
    docs = []
    for index, line in enumerate(mapping):
        row, identifier = line.split('||', 1)
        require(int(row) == index, 'Document mapping is not in consecutive row order')
        docs.append(identifier)
    require(len(docs) == len(labels), 'Document mapping/label length mismatch')
    require(len(set(docs)) == len(docs), 'Duplicate source document IDs')
    names = raw['labels.txt'].decode('utf-8').splitlines()
    require(classes.tolist() == list(range(1, len(names) + 1)),
            'Unexpected one-based Grimal label mapping')
    extra = {'sample_ids': docs, 'label_names': names,
             'label_original_values': classes.tolist(),
             'view_meaning': ['Released content features',
                              'Released citation counts, inbound plus outbound'],
             'preprocessing': 'Released document-by-feature content and cites matrices; '
                              'no min-max, no row normalization, no feature selection, '
                              'no row dropping; labels mapped to sorted zero-based IDs'}
    return matrices, labels, extra, None, None


def compare_prepared(matrices, labels, folder):
    # This checks every ordered matrix value, including implicit sparse zeros.
    expected_labels = np.load(folder / 'y.npy', allow_pickle=False)
    result = {'reference_directory': str(folder),
              'labels_equal': bool(np.array_equal(labels, expected_labels)), 'views': []}
    for view, actual in enumerate(matrices):
        expected = canonical(sparse.load_npz(folder / f'X{view}.npz'))
        require(actual.shape == expected.shape, f'View {view} shape mismatch')
        difference = actual - expected
        difference.eliminate_zeros()
        result['views'].append({
            'view': view, 'shape': list(actual.shape),
            'all_ordered_values_equal': bool(difference.nnz == 0),
            'maximum_absolute_error': float(np.max(np.abs(difference.data), initial=0)),
            'csr_indptr_equal': bool(np.array_equal(actual.indptr, expected.indptr)),
            'csr_indices_equal': bool(np.array_equal(actual.indices, expected.indices)),
            'csr_data_equal': bool(np.array_equal(actual.data, expected.data))})
    result['passed'] = result['labels_equal'] and all(
        view['all_ordered_values_equal'] and view['csr_indptr_equal'] and
        view['csr_indices_equal'] and view['csr_data_equal'] for view in result['views'])
    require(result['passed'], 'Elementwise comparison with frozen preparation failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['BBC', 'Cora', 'CiteSeer'], required=True)
    parser.add_argument('--source', type=Path, required=True,
                        help='BBC raw four-file directory or ZIP, or exact Grimal tar.gz archive')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--identity', type=Path,
                        default=Path(__file__).with_name('bbc_citation_identity.json'))
    parser.add_argument('--verify-against', type=Path,
                        help='Optional frozen prepared directory for elementwise verification')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.exists(), 'Output already exists; choose a new directory')
    identities = json.loads(args.identity.read_text(encoding='utf-8'))
    identity = identities['datasets'][args.dataset]
    if args.dataset == 'BBC':
        matrices, labels, extra, indices, terms = prepare_bbc(source, identity)
    else:
        matrices, labels, extra, indices, terms = prepare_citation(args.dataset, source, identity)
    require([list(x.shape) for x in matrices] == identity['prepared_shapes'],
            'Prepared view shapes differ from the frozen configuration')
    require(all(x.shape[0] == len(labels) for x in matrices), 'Inconsistent object count')
    digest = prepared_hash(matrices, labels)
    require(digest == identity['prepared_sha256'], 'Ordered prepared data identity mismatch')
    require(sequence_hash(extra['sample_ids']) == identity['ordered_document_ids_sha256'],
            'Ordered source document identity mismatch')
    verification = None
    if args.verify_against is not None:
        verification = compare_prepared(matrices, labels, args.verify_against.resolve())
    output.mkdir(parents=True)
    for view, matrix in enumerate(matrices):
        sparse.save_npz(output / f'X{view}.npz', matrix)
    np.save(output / 'y.npy', labels, allow_pickle=False)
    np.save(output / 'labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy', labels,
            allow_pickle=False)
    if indices is not None:
        np.savez_compressed(output / 'feature_indices.npz', view0=indices[0], view1=indices[1])
        dump(output / 'feature_names.json', [[terms[i] for i in ix] for ix in indices])
    metadata = dict(dataset=args.dataset, n=len(labels), k=int(np.unique(labels).size),
                    views=len(matrices), dimensions=[x.shape[1] for x in matrices],
                    class_counts=np.bincount(labels).tolist(),
                    zero_rows=[int(np.sum(np.diff(x.indptr) == 0)) for x in matrices],
                    preparation_kind=identity['preparation_kind'], prepared_sha256=digest,
                    source_input=str(source), source_urls=identity['source_urls'],
                    input_identity=identity, preparation_source_sha256=sha_file(__file__),
                    identity_file_sha256=sha_file(args.identity),
                    runtime={'python': platform.python_version(), 'numpy': np.__version__,
                             'scipy': scipy.__version__, 'isolated': bool(sys.flags.isolated)},
                    training_labels='Oracle and sealed scoring only; never pass labels to training',
                    **extra)
    dump(output / 'metadata.json', metadata)
    if verification is not None:
        dump(output / 'ELEMENTWISE_VERIFICATION.json', verification)
    dump(output / 'INPUT_SHA256.json', {p.name: sha_file(p)
                                      for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({'dataset': args.dataset, 'prepared_sha256': digest,
                      'shapes': [list(x.shape) for x in matrices],
                      'isolated': bool(sys.flags.isolated),
                      'elementwise_verified': verification is not None,
                      'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
