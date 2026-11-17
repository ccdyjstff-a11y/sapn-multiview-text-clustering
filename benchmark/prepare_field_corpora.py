"""Rebuild the paper's two field corpora from the public, hash-pinned archives.

This file is standalone. It imports no research-tree modules, performs no model
training, and never extracts archive members onto the filesystem. Run with -I.
"""
from pathlib import Path
from email.parser import Parser
from email import policy
import argparse
import csv
import hashlib
import html
import inspect
import io
import json
import platform
import re
import sys
import tarfile
import time

import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets._twenty_newsgroups import strip_newsgroup_footer, strip_newsgroup_quoting
from threadpoolctl import threadpool_limits


ARCHIVES = {
    '20ng': '8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610',
    'agnews': '4e2fc37d369e6e317a8b62d0d2f11d379f8c3dde60c07c35a46b069a246498ad',
}
VERSIONS = {'python': '3.10.22', 'numpy': '2.2.6', 'scipy': '1.15.3', 'scikit-learn': '1.7.2'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def json_identity(value):
    return hashlib.sha256(json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def array_identity(a):
    a = np.asarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def prepared_identity(X, y):
    # The historical research identity, including CSR structure and numeric bytes.
    h = hashlib.sha256(np.ascontiguousarray(y).tobytes())
    for x in X:
        x = sparse.csr_matrix(x)
        x.sort_indices()
        h.update(str(x.shape).encode())
        for a in (x.indptr, x.indices, x.data):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def scientific_identity(X, y, provenance, vocabularies):
    return dict(prepared_sha256=prepared_identity(X, y), labels=array_identity(y),
                matrices=[dict(shape=list(x.shape), indptr=array_identity(x.indptr),
                               indices=array_identity(x.indices), data=array_identity(x.data)) for x in X],
                sample_provenance_sha256=json_identity(provenance),
                vocabularies_sha256=json_identity(vocabularies))


def clean_20ng(raw):
    header, separator, body = raw.partition('\n\n')
    if not separator:
        return '', ''
    message = Parser(policy=policy.default).parsestr(header + '\n\n', headersonly=True)
    subject = str(message.get('Subject', ''))
    subject = re.sub(r'^(?:(?:re|fw|fwd)(?:\[\d+\])?\s*:\s*)+', '', subject, flags=re.I).strip()
    body = strip_newsgroup_quoting(strip_newsgroup_footer(body)).strip()
    return subject, body


def prepare_20ng(archive):
    groups, seen = {}, {}
    duplicates = conflicts = empty = 0
    # Read compressed bytes once, then apply the historical lexical member order.
    # This changes I/O only, avoiding repeated gzip seeks for sorted member reads.
    raw_records = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.isfile() and member.name.startswith('20news-bydate-train/'):
                if member.name in raw_records:
                    raise ValueError('Duplicate archive member name')
                raw_records[member.name] = tar.extractfile(member).read()
    for member_name in sorted(raw_records):
            raw = raw_records[member_name].decode('latin1').replace('\r\n', '\n')
            subject, body = clean_20ng(raw)
            category = member_name.split('/')[1]
            if not subject.strip() or not body.strip():
                empty += 1
                continue
            canonical = re.sub(r'\s+', ' ', subject + '\n' + body).strip().casefold()
            fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            if fingerprint in seen:
                duplicates += 1
                conflicts += int(seen[fingerprint] != category)
                continue
            seen[fingerprint] = category
            groups.setdefault(category, []).append((member_name, subject, body, fingerprint))
    categories = sorted(groups)
    if len(categories) != 20:
        raise ValueError('Expected all 20 training categories')
    rng = np.random.RandomState(20261010)
    selected, labels = [], []
    for i, category in enumerate(categories):
        records = groups[category]
        if len(records) < 100:
            raise ValueError('Insufficient eligible records in ' + category)
        choices = rng.permutation(len(records))[:100]
        selected.extend(records[j] for j in choices)
        labels.extend([i] * 100)
    y = np.asarray(labels, dtype=np.int64)
    X, vocabs, idfs = [], [], []
    for field, cap in ((1, 5000), (2, 10000)):
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=2, max_df=.95,
                                    sublinear_tf=True, norm='l2', max_features=cap)
        x = vectorizer.fit_transform([r[field] for r in selected]).tocsr()
        x.sort_indices()
        X.append(x)
        vocabs.append(vectorizer.get_feature_names_out().tolist())
        idfs.append(vectorizer.idf_.tolist())
    provenance = [dict(path=r[0], cleaned_text_sha256=r[3]) for r in selected]
    extra = dict(dataset='20NG_SubjectBody_2000', categories=categories, view_meaning=['subject', 'body'],
                 empty_cleaned_removed_before_sampling=empty, duplicate_cleaned_records_removed=duplicates,
                 conflicting_category_duplicate_records=conflicts,
                 eligible_per_class={c: len(groups[c]) for c in categories},
                 sample_seed=20261010, samples_per_class=100,
                 cleaner_helpers_sha256=hashlib.sha256((inspect.getsource(strip_newsgroup_footer) +
                                                        inspect.getsource(strip_newsgroup_quoting)).encode()).hexdigest())
    return X, y, provenance, vocabs, idfs, extra, None


def archive_bytes(tar, suffix):
    members = [m for m in tar.getmembers() if m.isfile() and (m.name == suffix or m.name.endswith('/' + suffix))]
    if len(members) != 1:
        raise ValueError('Expected one archive member ' + suffix)
    return tar.extractfile(members[0]).read()


def prepare_agnews(archive):
    seen, groups, dropped, raw_counts = {}, {i: [] for i in range(1, 5)}, [], {i: 0 for i in range(1, 5)}
    empty = duplicates = conflicts = raw_total = 0
    with tarfile.open(archive, 'r:gz') as tar:
        train_bytes = archive_bytes(tar, 'train.csv')
        categories = archive_bytes(tar, 'classes.txt').decode('utf-8').splitlines()
    with io.TextIOWrapper(io.BytesIO(train_bytes), encoding='utf-8', newline='') as stream:
        for row_id, row in enumerate(csv.reader(stream), 1):
            if len(row) != 3 or int(row[0]) not in groups:
                raise ValueError('Invalid original CSV row ' + str(row_id))
            label = int(row[0])
            raw_counts[label] += 1
            raw_total += 1
            title, description = [re.sub(r'\s+', ' ', html.unescape(v)).strip() for v in row[1:]]
            if not title or not description:
                empty += 1
                dropped.append(dict(row_id=row_id, reason='empty_cleaned_field'))
                continue
            canonical = re.sub(r'\s+', ' ', title + '\n' + description).strip().casefold()
            digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            if digest in seen:
                first = seen[digest]
                duplicates += 1
                conflict = first['label'] != label
                conflicts += int(conflict)
                dropped.append(dict(row_id=row_id, reason='duplicate_combined_cleaned_key', retained_row_id=first['row_id'],
                                    label_conflict=conflict, text_key_sha256=digest))
                continue
            record = dict(row_id=row_id, label=label, title=title, description=description, text_key_sha256=digest)
            seen[digest] = record
            groups[label].append(record)
    if raw_total != 120000 or any(v != 30000 for v in raw_counts.values()):
        raise ValueError('Unexpected original CSV counts')
    rng = np.random.RandomState(20261015)
    selected = []
    for label in range(1, 5):
        if len(groups[label]) < 500:
            raise ValueError('Insufficient eligible rows')
        selected.extend(groups[label][int(i)] for i in rng.permutation(len(groups[label]))[:500])
    y = np.asarray([r['label'] - 1 for r in selected], dtype=np.int64)
    X, vocabs, idfs = [], [], []
    for field, cap in [('title', 5000), ('description', 10000)]:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=2, max_df=.95,
                                    sublinear_tf=True, norm='l2', ngram_range=(1, 1), max_features=cap)
        x = vectorizer.fit_transform([r[field] for r in selected]).tocsr()
        x.sort_indices()
        X.append(x)
        vocabs.append(vectorizer.get_feature_names_out().tolist())
        idfs.append(vectorizer.idf_.tolist())
    provenance = [dict(object_index=i, original_train_record=r['row_id'], original_class_id=r['label'],
                       cleaned_combined_text_sha256=r['text_key_sha256']) for i, r in enumerate(selected)]
    extra = dict(dataset='AGNews_TitleDescription_2000', categories=categories, view_meaning=['title', 'description'],
                 original_records=raw_total, original_class_counts=raw_counts, empty_cleaned_removed=empty,
                 duplicate_cleaned_records_removed=duplicates, conflicting_category_duplicate_records=conflicts,
                 eligible_per_class={str(k): len(v) for k, v in groups.items()}, sample_seed=20261015,
                 samples_per_class=500, source_csv_sha256=hashlib.sha256(train_bytes).hexdigest())
    return X, y, provenance, vocabs, idfs, extra, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=sorted(ARCHIVES))
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--identities', type=Path, default=Path(__file__).resolve().with_name('FIELD_CORPUS_IDENTITIES.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}
    if versions != VERSIONS:
        raise RuntimeError('Use the pinned environment. Expected ' + repr(VERSIONS) + '; got ' + repr(versions))
    archive = args.archive.resolve()
    if file_sha(archive) != ARCHIVES[args.dataset]:
        raise ValueError('Public source archive SHA256 does not match')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new or empty: ' + str(out))
    identities = json.loads(args.identities.read_text(encoding='utf-8'))
    expected = identities['datasets'][args.dataset]
    with threadpool_limits(limits=1):
        X, y, provenance, vocabs, idfs, extra, dropped = (prepare_20ng if args.dataset == '20ng' else prepare_agnews)(archive)
    actual = scientific_identity(X, y, provenance, vocabs)
    if actual != expected['scientific_identity']:
        raise ValueError('Rebuilt features/labels/row identities/vocabularies differ from the historical identity')
    out.mkdir(parents=True, exist_ok=True)
    for v, x in enumerate(X):
        sparse.save_npz(out / f'X{v}.npz', x)
    np.save(out / 'y.npy', y, allow_pickle=False)
    dump(out / 'sample_provenance.json', provenance)
    dump(out / 'vocabularies.json', vocabs)
    dump(out / 'idf_values.json', idfs)
    if dropped is not None:
        dump(out / 'deduplication_exclusions.json', dropped)
    metadata = dict(extra, n=len(y), k=len(np.unique(y)), views=len(X), dimensions=[x.shape[1] for x in X],
                    class_counts=np.bincount(y), nonzeros=[int(x.nnz) for x in X],
                    zero_rows=[int(np.sum(np.asarray(x.multiply(x).sum(axis=1)).ravel() == 0)) for x in X],
                    prepared_sha256=actual['prepared_sha256'], archive_sha256=ARCHIVES[args.dataset],
                    preparation_source_sha256=file_sha(__file__), versions=versions,
                    training_labels='y.npy is only for query oracle and sealed scoring, not a model input',
                    note='Metadata includes new provenance fields and is not byte-identical to the historical metadata.',
                    raw_texts_exported=False, clustering_training_performed=False, clustering_evaluation_performed=False)
    dump(out / 'metadata.json', metadata)
    dump(out / 'PREPARATION_VERIFICATION.json', dict(status='PASS', dataset=args.dataset,
         archive_sha256=file_sha(archive), identity_manifest_sha256=file_sha(args.identities),
         scientific_identity=actual, scientific_identity_exact=True,
         file_hashes={p.name: file_sha(p) for p in out.iterdir() if p.is_file()},
         versions=versions, single_process=True, numerical_thread_limit=1,
         elapsed_seconds=time.perf_counter() - started, trained_models=0))
    print(json.dumps(dict(status='PASS', dataset=args.dataset, n=len(y), dimensions=metadata['dimensions'],
                          zero_rows=metadata['zero_rows'], prepared_sha256=actual['prepared_sha256']), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
