"""Reproduce seven presentation-only figure revisions without private inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package(root):
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        path=root/item['path']
        if path.resolve().is_relative_to(root) is False or sha(path)!=item['sha256']:
            raise ValueError('Package input identity changed: '+item['path'])
    return manifest


def main():
    if not __debug__:
        raise RuntimeError('Run normally, without -O/-OO')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True,help='New output directory outside this package')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Choose a new output directory outside the package')
    verify_package(root)
    # Import all runtime requirements before creating outputs. No external
    # audit helper, manuscript source, raw corpus or remote service is needed.
    import numpy
    import matplotlib
    import PIL
    import pymupdf
    from matplotlib.font_manager import FontProperties,findfont
    try:
        findfont(FontProperties(family='Arial'),fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError('Arial is required to match the approved appearance. Use a lawful system Arial installation; font files are not redistributed here.') from exc
    out.mkdir(parents=True)
    jobs=[('model',['build_model_figure.py','--output',str(out/'model')]),
          ('si',['plot_si_readable_fonts.py','--release-root',str(root/'release_inputs'),'--output',str(out/'si')]),
          ('raster',['plot_raster_figures_as_vectors.py','--release-root',str(root/'release_inputs'),'--output',str(out/'raster')])]
    for name,arguments in jobs:
        result=subprocess.run([sys.executable,'-B',str(root/arguments[0]),*arguments[1:]],
            capture_output=True,text=True,encoding='utf8',cwd=out)
        if result.returncode:
            raise RuntimeError(name+' replay failed:\n'+result.stderr)
        print(name+': rendered',flush=True)
    from verify_outputs import verify
    result=verify(out,root)
    verify_package(root)
    result.update(manifest_sha256=sha(root/'MANIFEST.json'),
        runtime=dict(python=sys.version.split()[0],numpy=numpy.__version__,
                     matplotlib=matplotlib.__version__,Pillow=PIL.__version__,PyMuPDF=pymupdf.VersionBind),
        no_network_fit_selection_or_class_scoring=True,package_inputs_unchanged=True)
    (out/'EQUIVALENCE_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('PASS: 7/7 figures match approved scientific geometry and appearance',flush=True)


if __name__=='__main__':
    main()
