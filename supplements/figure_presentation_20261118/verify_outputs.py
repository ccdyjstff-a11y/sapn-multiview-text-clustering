"""Study-authored equivalence checks against the seven approved figure assets.

This checks actual PDF vector geometry/text, embedded fonts, model dependencies,
and the exact scientific-artist traces. It does not claim a general collision
audit: the approved assets already underwent that separate rendered review.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pymupdf as pdf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def normalized(value):
    if isinstance(value,dict):
        return {key:normalized(item) for key,item in value.items()}
    if isinstance(value,(pdf.Point,pdf.Rect,pdf.Quad,list,tuple)):
        return [normalized(item) for item in value]
    if isinstance(value,float):
        return round(value,3)
    return value


def verify(output,root):
    expected=json.loads((root/'EXPECTED_OUTPUTS.json').read_text(encoding='utf8'))
    checks=[]
    for reference in expected['figures']:
        stem,group=reference['figure'],reference['group']
        folder=output/group/('figures' if group=='model' else '')
        path=folder/(stem+'.pdf')
        with pdf.open(path) as doc:
            if len(doc)!=1 or doc[0].get_images():
                raise ValueError('Expected one vector-only page: '+stem)
            page=doc[0]
            drawings=[normalized({key:item[key] for key in (
                'items','type','color','fill','width','dashes','lineCap','lineJoin',
                'closePath','stroke_opacity','fill_opacity')}) for item in page.get_drawings()]
            if digest(drawings)!=reference['pdf_drawing_fingerprint']:
                raise ValueError('Approved vector geometry/style changed: '+stem)
            if digest(' '.join(page.get_text().split()))!=reference['text_fingerprint']:
                raise ValueError('Approved text changed: '+stem)
            if [page.rect.width,page.rect.height]!=reference['page_size_pt']:
                raise ValueError('Approved canvas changed: '+stem)
            if not all(doc.extract_font(f[0])[3] for f in page.get_fonts()):
                raise ValueError('Unembedded font: '+stem)
        if group=='model':
            contract=json.loads((output/'model/figure_qa'/('fig00_model_v4_contract.json')).read_text(encoding='utf8'))
            value={key:contract[key] for key in ['nodes','edges','mapping','text']}
            if digest(value)!=reference['model_dependency_fingerprint']:
                raise ValueError('Approved model dependency contract changed')
        else:
            trace=json.loads((folder/(stem+'.geometry.json')).read_text(encoding='utf8'))
            if digest(trace)!=reference['scientific_artist_fingerprint']:
                raise ValueError('Approved scientific-artist coordinates changed: '+stem)
        # Exact pixel equality is an additional typography/appearance check,
        # not a substitute for the independent numerical and vector checks.
        if sha(folder/(stem+'.png'))!=reference['approved_png_sha256']:
            raise ValueError('Approved raster appearance changed; check fonts/runtime: '+stem)
        checks.append(dict(figure=stem,relative_pdf=path.relative_to(output).as_posix(),
            approved_geometry_and_style=True,approved_text=True,embedded_fonts=True,
            model_dependencies_or_scientific_artists_equal=True,png_byte_identical=True,
            vector_paths=len(drawings),pdf_sha256=sha(path)))
    return dict(status='PASS',figures=checks,figure_count=len(checks),
        expected_outputs_sha256=sha(root/'EXPECTED_OUTPUTS.json'),
        scope='Independent equivalence against already reviewed presentation assets; no new experiment or statistical analysis')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=verify(args.output.resolve(),Path(__file__).resolve().parent)
    (args.output/'EQUIVALENCE_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('PASS: all seven approved scientific figures reproduced')


if __name__=='__main__':
    main()
