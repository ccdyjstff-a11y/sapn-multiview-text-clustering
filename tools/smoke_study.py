"""Exercise the portable study barrier with synthetic data and deliberate invalid requests."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value),encoding='utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new report')
    results=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='')
    with tempfile.TemporaryDirectory(prefix='.qa_study_',dir=ROOT) as temporary:
        t=Path(temporary);data=t/'data';out=t/'run';queries=t/'queries'
        prep=t/'prepare.json';dump(prep,dict(dataset='Synthetic',format='synthetic_smoke',source='unused',output=str(data)))
        def run(name,cmd,success=True):
            q=subprocess.run(cmd,cwd=t,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
            assert (q.returncode==0)==success,(name,q.returncode,q.stdout,q.stderr)
            results.append(dict(check=name,returncode=q.returncode,expected_success=success,
                stdout=q.stdout,stderr=q.stderr))
            return q
        run('prepare',[sys.executable,'-B',str(ROOT/'benchmark/prepare_data.py'),'--config',str(prep)])
        dm=t/'data.json';lm=t/'labels.json';dump(dm,dict(Synthetic=str(data)));dump(lm,dict(Synthetic=str(data/'labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy')))
        shared=['--profile','custom','--corpora','Synthetic','--seeds','17','--budgets','100',
            '--data-map',str(dm),'--queries-root',str(queries),'--output',str(out)]
        def command(phase,labels=False):return [sys.executable,'-B','-I',str(ROOT/'tools/run_common_study.py'),phase,*shared]+(['--labels-map',str(lm)] if labels else [])
        run('training_rejects_label_argument',command('train',True),False)
        run('oracle',command('oracle',True))
        plan=json.loads(run('plan',command('plan')).stdout)
        assert plan['expected_blocks']==1 and plan['expected_logical_finals']==5
        assert plan['eta']==[0.,.5,1.] and plan['lambda_grid_default_first']==[20.,2.,200.]
        run('score_before_training_rejected',command('score',True),False)
        run('train',command('train'))
        assert not list(out.rglob('evaluation.json'))
        seal=out/'STUDY_PREDICTIONS_SEALED.json';saved=seal.read_bytes();seal.unlink()
        run('block_seals_without_study_seal_rejected',command('score',True),False)
        assert not list(out.rglob('evaluation.json'));seal.write_bytes(saved)
        altered=command('score',True);altered[altered.index('--seeds')+1]='18'
        run('changed_roster_rejected',altered,False)
        run('score',command('score',True))
        rows=json.loads((out/'STUDY_EVALUATION.json').read_text(encoding='utf-8'))
        assert len(rows)==5
        result=dict(status='PASS',scope='Synthetic one-block study orchestration; no real-corpus training',
            launcher_sha256=sha(ROOT/'tools/run_common_study.py'),checks=results,logical_finals=len(rows),
            independent_full_label_commands=['oracle','score'],no_evaluation_before_global_seal=True,
            eta=plan['eta'],lambda_grid_default_first=plan['lambda_grid_default_first'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
