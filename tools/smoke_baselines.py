"""Exercise the supplied paper-based CMVNMF/MVCHSS interfaces on synthetic data."""
from pathlib import Path
import argparse, json, os, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new report')
    results=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='')
    with tempfile.TemporaryDirectory(prefix='.qa_baseline_',dir=ROOT) as td:
        t=Path(td);data=t/'data';run=t/'run';query=t/'queries.npz'
        specs=dict(prepare=dict(dataset='Synthetic',format='synthetic_smoke',source='unused',output=str(data)),
          oracle=dict(data=str(data),labels=str(data/'labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy'),seed=17,total_queries=100,output=str(query)),
          train=dict(data=str(data),queries=str(query),output=str(run),seed=17,budgets=[100],methods=['CMVNMF','MVCHSS']),
          score=dict(output=str(run),labels=str(data/'labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy')))
        for phase,c in specs.items():
            config=t/(phase+'.json');config.write_text(json.dumps(c),encoding='utf-8')
            script=ROOT/'benchmark'/('prepare_data.py' if phase=='prepare' else 'reproduce.py')
            cmd=[sys.executable,'-B',str(script)]+([] if phase=='prepare' else [phase])+['--config',str(config)]
            q=subprocess.run(cmd,cwd=t,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
            results.append(dict(phase=phase,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr))
            if q.returncode:raise RuntimeError(phase+' failed\n'+q.stderr)
        evaluations=json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
        diagnostics=[]
        for f in run.rglob('diagnostics.json'):
            j=json.loads(f.read_text(encoding='utf-8'));assert j.get('status')!='exception',(f,j)
            diagnostics.append(dict(method=j['method'],eligible=j['eligible'],path=f.relative_to(run).as_posix()))
        result=dict(status='PASS',scope='Synthetic interface execution, not new corpus performance or author-code equivalence',
          methods=['CMVNMF','MVCHSS'],commands=results,fit_diagnostics=diagnostics,evaluation_records=len(evaluations),
          MATLAB_launched=False,third_party_code_downloaded=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['commands','fit_diagnostics']}))

if __name__=='__main__':main()
