"""Materialize external author code only after users obtain/access it themselves.

No author source is redistributed in this package. Exact URLs and checksums are
provided in THIRD_PARTY_INPUTS.json. This script verifies an existing checkout,
copies required author files locally and generates diagnostic derivatives.
"""
from pathlib import Path
import argparse,shutil,subprocess,sys
from portable_budget_runner import read,sha

def main(author):
    root=Path(__file__).resolve().parent;author=Path(author).resolve();manifest=read(root/'THIRD_PARTY_INPUTS.json')
    target=root/'literature/FSSMSC/master';target.mkdir(parents=True,exist_ok=True)
    for r in manifest['FSSMSC']['required_files']:
        source=author/r['path'];assert source.is_file() and sha(source)==r['sha256'],str(source)
        dest=target/r['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==r['sha256']
        else:shutil.copyfile(source,dest)
    subprocess.run([sys.executable,str(root/'fssmsc/build_adaptation.py')],check=True,cwd=root)
    print('Verified author sources and generated local diagnostics. Redistribution permission is not implied.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--author-checkout',required=True);a=p.parse_args();main(a.author_checkout)
