"""Reproduce the inspected IJMLC model-figure typography variant.

This is a fixed figure reproduction check, not a model or experiment runner.
The exact plot program is retained; only its output directory is supplied.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import subprocess
import sys

import pymupdf as pdf

ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode())


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (pdf.Point, pdf.Rect, pdf.Quad, list, tuple)):
        return [normalized(item) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    return value


def pdf_fingerprint(path):
    with pdf.open(path) as doc:
        if len(doc) != 1 or doc[0].get_images():
            raise ValueError('Expected one vector-only model schematic')
        page = doc[0]
        drawings = [normalized({key: row[key] for key in (
            'items', 'type', 'color', 'fill', 'width', 'dashes', 'lineCap',
            'lineJoin', 'closePath', 'stroke_opacity', 'fill_opacity')})
            for row in page.get_drawings()]
        spans = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    spans.append(normalized({key: span[key] for key in
                        ('text', 'font', 'size', 'flags', 'color', 'origin', 'bbox')}))
        embedded = all(doc.extract_font(font[0])[3] for font in page.get_fonts())
        if not embedded:
            raise ValueError('A font is not embedded')
        return dict(drawing_fingerprint=digest(drawings),
                    text_font_position_fingerprint=digest(spans),
                    page_size_pt=[page.rect.width, page.rect.height],
                    vector_paths=len(drawings), text_spans=len(spans),
                    embedded_fonts=True)


def verify_sources():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if path.parent != ROOT or sha(path.read_bytes()) != expected:
            raise ValueError('Package file differs from manifest: ' + name)
    provenance = json.loads((ROOT / 'PROVENANCE.json').read_text(encoding='utf-8'))
    raw = (ROOT / 'build_model_figure_ijmlc.py').read_bytes()
    if sha(raw) != provenance['derived_source_sha256']:
        raise ValueError('The inspected plotting source changed')
    original = raw.decode('utf-8')
    for edit in reversed(provenance['edits']):
        if original.count(edit['after']) != edit['occurrences']:
            raise ValueError('A reversed glyph edit is nonunique: ' + edit['id'])
        original = original.replace(edit['after'], edit['before'])
    if sha(original.encode('utf-8')) != provenance['original_source_sha256']:
        raise ValueError('The original public source is not restored exactly')
    tree = ast.parse(raw)
    reverse = {ast.literal_eval(edit['after']): ast.literal_eval(edit['before'])
               for edit in provenance['edits']}
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in reverse:
            node.value = reverse[node.value]
            count += 1
    if ast.dump(tree) != ast.dump(ast.parse(original)) or count != 11:
        raise ValueError('An unexpected non-label operation changed')
    return dict(original_source_restored=True, nonlabel_AST_unchanged=True,
                changed_label_literals=count, derived_source_sha256=sha(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory; existing outputs are not overwritten')
    parser.add_argument('--audit-tools', type=Path,
                        help='Optional external nature-figure audit scripts')
    parser.add_argument('--qa-dependencies', type=Path,
                        help='Optional external dependencies for the audits')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output directory must be new')
    source_check = verify_sources()
    expected = json.loads((ROOT / 'EXPECTED_FIGURE.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    command = [sys.executable, '-X', 'utf8', '-B', str(ROOT / 'build_model_figure_ijmlc.py'),
               '--output', str(output)]
    if args.audit_tools:
        command += ['--audit-tools', str(args.audit_tools.resolve())]
    if args.qa_dependencies:
        command += ['--qa-dependencies', str(args.qa_dependencies.resolve())]
    run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    (output / 'render.stdout.txt').write_text(run.stdout, encoding='utf-8')
    (output / 'render.stderr.txt').write_text(run.stderr, encoding='utf-8')
    if run.returncode:
        raise RuntimeError('Rendering failed; inspect preserved render logs')
    folder = output / 'figures'
    stem = 'fig00_model_v4'
    actual = pdf_fingerprint(folder / (stem + '.pdf'))
    if actual != expected['pdf_fingerprint']:
        raise ValueError('Approved PDF geometry/text/font positions differ; check runtime and fonts')
    if sha((folder / (stem + '.png')).read_bytes()) != expected['approved_png_sha256']:
        raise ValueError('Approved PNG bytes differ; check runtime and fonts')
    contract = json.loads((output / 'figure_qa' / (stem + '_contract.json')).read_text(encoding='utf-8'))
    if digest({key: contract[key] for key in ('nodes', 'edges', 'mapping', 'text')}) != expected['model_dependency_fingerprint']:
        raise ValueError('Approved model dependencies differ')
    receipt = dict(status='PASS', scope='Model schematic typography reproduction only; no experimental data or new model fit',
        source=source_check, pdf=actual, model_dependencies_equal=True, png_byte_identical=True,
        pdf_byte_identical=sha((folder / (stem + '.pdf')).read_bytes()) == expected['approved_pdf_sha256'],
        pdf_note='PDF file bytes may differ due to export metadata; inspected content fingerprints are compared separately.',
        external_audits_executed=args.audit_tools is not None,
        output_sha256={ext: sha((folder / (stem + '.' + ext)).read_bytes()) for ext in ('pdf', 'svg', 'png', 'tiff')})
    (output / 'REPRODUCTION_QA.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('PASS: inspected model geometry, text, font positions, dependency mapping and PNG reproduced')


if __name__ == '__main__':
    main()
