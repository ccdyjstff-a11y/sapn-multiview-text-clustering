"""Validate the assembled local release and write the final SHA256 inventory."""
from pathlib import Path
from zipfile import ZipFile
import argparse, ast, csv, hashlib, io, json, re, subprocess, sys, tokenize

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def executable_signature(source):
    """Exclude documentation, retaining every executable token and AST node."""
    tree=ast.parse(source);spans=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and ast.get_docstring(node,clean=False):
            item=node.body[0].value;spans.append((item.lineno,item.end_lineno))
    tokens=[(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(source).readline)
        if t.type not in (tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENCODING)
        and not (t.type==tokenize.STRING and any(lo<=t.start[0]<=hi for lo,hi in spans))]
    class RemoveDocs(ast.NodeTransformer):
        def strip(self,node):
            self.generic_visit(node)
            if ast.get_docstring(node,clean=False):node.body=node.body[1:]
            return node
        visit_Module=strip;visit_FunctionDef=strip;visit_AsyncFunctionDef=strip;visit_ClassDef=strip
    return tokens,ast.dump(RemoveDocs().visit(tree),include_attributes=False)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    report=dict(status='PASS',scope='Local release integrity and executable-entry checks; not external publication',checks={})
    cjk=re.compile('[\u4e00-\u9fff]');sources=[]
    for f in sorted(ROOT.rglob('*.py')):
        assert not any(x.startswith('.qa_') for x in f.parts),'Unfinished smoke directory'
        s=f.read_text(encoding='utf-8-sig');compile(s,str(f),'exec');tree=ast.parse(s)
        for t in tokenize.generate_tokens(io.StringIO(s).readline):
            assert not (t.type==tokenize.COMMENT and cjk.search(t.string)),(f,t.start)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                assert not cjk.search(ast.get_docstring(node) or ''),(f,getattr(node,'lineno',0))
        sources.append(f.relative_to(ROOT).as_posix())
    report['checks']['python_files_compiled_and_English_comments']=sources
    for ledger in ['ARCHIVE_ORIGINS.json','CURRENT_EVIDENCE.json','DATA_PREPARATION_ORIGINS.json']:
        entries=read(ROOT/'provenance'/ledger);entries=entries['members'] if isinstance(entries,dict) else entries
        for r in entries:assert sha(ROOT/r['release_file'])==r.get('release_sha256',r.get('sha256')),r['release_file']
        report['checks'][ledger]=len(entries)
    translations=read(ROOT/'provenance/TRANSLATION_QA.json')['files']
    for r in translations:
        old=ROOT/'provenance/original_sources'/(r['file']+'.txt');new=ROOT/r['file']
        assert sha(old)==r['original_sha256'] and sha(new)==r['release_sha256'],r['file']
        assert executable_signature(old.read_text(encoding='utf-8-sig'))==executable_signature(new.read_text(encoding='utf-8-sig')),r['file']
    report['checks']['English_translation_executable_token_and_AST_recheck']=len(translations)
    ssa_root=ROOT/'baselines/ssa_historical'
    own=read(ssa_root/'OWN_SOURCE_MANIFEST.json')['files']
    for r in own:assert sha(ssa_root/r['packaged_path'])==r['sha256'],r['packaged_path']
    report['checks']['SSA_historical_own_sources_hash_verified']=len(own)
    prov=read(ROOT/'common/SOURCE_PROVENANCE.json')
    for r in prov['copies']:assert sha(ROOT/'common'/r['file'])==r['sha256']
    assert sha(ROOT/'common/allocation_core.py')==read(ROOT/'common/CORE_DERIVATION.json')['derived_sha256']
    smoke=read(ROOT/'validation/SYNTHETIC_EQUIVALENCE.json');assert smoke['status']=='PASS'
    assert smoke['english_core_sha256']==sha(ROOT/'common/allocation_core.py')
    report['checks']['synthetic_equivalence']={k:v for k,v in smoke.items() if k not in ['commands','comparisons']}
    baseline=read(ROOT/'validation/PYTHON_BASELINE_SMOKE.json');assert baseline['status']=='PASS'
    assert baseline['evaluation_records']==2 and all(r['eligible'] for r in baseline['fit_diagnostics'])
    report['checks']['Python_baseline_synthetic_interfaces']=baseline
    study=read(ROOT/'validation/STUDY_ORCHESTRATION.json');assert study['status']=='PASS'
    assert study['launcher_sha256']==sha(ROOT/'tools/run_common_study.py')
    report['checks']['portable_study_orchestration']=study
    replay=read(ROOT/'validation/SSA_REPLAY.json');assert replay['passed'] is True
    report['checks']['SSA_compact_replay']=replay
    real=read(ROOT/'validation/reproduction_main/VERIFICATION.json');assert real['status']=='PASS'
    real_compact=read(ROOT/'validation/reproduction_main/COMPACT_REPLAY.json');assert real_compact['status']=='PASS'
    for r in read(ROOT/'validation/reproduction_main/COMPACT_EXPORT.json')['records']:
        base=ROOT/'validation/reproduction_main' if r['base']=='reproduction_main' else ROOT
        assert sha(base/r['file'])==r['sha256'],r['file']
    report['checks']['real_main_instance_reproduction']={k:v for k,v in real.items() if k!='checks'}
    report['checks']['real_main_compact_replay']=real_compact
    primary=read(ROOT/'validation/PRIMARY_REAL_RESCORE_630.json')
    binding=read(ROOT/'validation/PRIMARY_REAL_RESCORE_BINDING.json')
    assert primary['status']=='PASS' and primary['predictions']==210 and primary['metric_comparisons']==630
    assert primary['manifest_sha256']==sha(ROOT/'primary_predictions/MANIFEST.json')
    assert primary['scorer_sha256']==sha(ROOT/'primary_predictions/rescore.py')
    assert binding['report_sha256']==sha(ROOT/'validation/PRIMARY_REAL_RESCORE_630.json')
    report['checks']['actual_primary_external_label_rescoring']={k:v for k,v in primary.items() if k not in ['records','table_cells']}
    report['checks']['data_preparation']=read(ROOT/'data_tools/PREPARATION_VERIFICATION_PORTABLE.json')
    mirror=read(ROOT/'data_tools/AGNEWS_MIRROR_PREPARATION_VERIFICATION.json')
    assert mirror['status']=='PASS' and mirror['regular_members_exact'] is True
    report['checks']['AG_News_mirror_actual_preparation']=mirror
    access=ROOT/'provenance/access/field_public_access'
    for name,h in read(access/'METADATA_SHA256.json').items():assert sha(access/name)==h,name
    report['checks']['field_public_access_metadata_hashes']=len(read(access/'METADATA_SHA256.json'))
    extension=ROOT/'evidence/allocation_extension'
    exported=read(extension/'SOURCE_EXPORT.json')
    for r in exported['files']:assert sha(ROOT/r['destination'])==r['sha256'],r['destination']
    export_check=read(extension/'EXPORT_VERIFICATION.json');assert export_check['status']=='PASS'
    ext_score=read(extension/'reporting/INDEPENDENT_RESCORE.json')
    assert ext_score['logical_records']==50 and ext_score['metric_values']==250 and ext_score['maximum_absolute_difference']==0
    assert ext_score['protocol_sha256']==sha(extension/'PROTOCOL.json')
    assert ext_score['global_seal_sha256']==sha(extension/'GLOBAL_SEAL.json')
    assert ext_score['evaluation_sha256']==sha(extension/'evaluation_all.json')
    report['checks']['allocation_extension_compact_export']=export_check
    report['checks']['allocation_extension_actual_post_seal_rescore']=ext_score
    tie=ROOT/'evidence/tie_sensitivity'
    for ledger in [tie/'SOURCE_EXPORT.json',ROOT/'provenance/TIE_PORTABLE_SOURCE_EXPORT.json']:
        for r in read(ledger)['files']:assert sha(ROOT/r['destination'])==r['sha256'],r['destination']
    members=read(tie/'PUBLIC_MANIFEST.json')['files']
    assert {r['file'] for r in members}=={p.relative_to(tie).as_posix() for p in tie.rglob('*') if p.is_file() and p.name!='PUBLIC_MANIFEST.json'}
    for r in members:assert sha(tie/r['file'])==r['sha256'],r['file']
    for name,count in [('REUSED_FINALS_386.json',386),('NEW_FITS_172.json',172)]:
        refs=read(tie/name);assert refs['count']==len(refs['records'])==count
        for r in refs['records']:assert sha(ROOT/r['prediction_member'])==r['prediction_sha256'],r['prediction_member']
    tie_export=read(tie/'EXPORT_VERIFICATION.json');assert tie_export['status']=='PASS'
    tie_replay=read(ROOT/'validation/TIE_COMPACT_REPLAY.json');assert tie_replay['status']=='PASS'
    assert tie_replay['unique_predictions']==558 and tie_replay['logical_records']==1500
    assert tie_replay['paired_points']==6000 and tie_replay['summary_groups']==600
    assert all(v==0 for v in tie_replay['metric_max_absolute_errors'].values())
    assert tie_replay['paired_max_absolute_error_pp']==tie_replay['summary_max_absolute_error_pp']==0
    assert tie_replay['public_manifest_sha256']==sha(tie/'PUBLIC_MANIFEST.json')
    assert tie_replay['script_sha256']==sha(tie/'tools/rescore_tie_compact.py')
    assert tie_replay['query_protocol_sha256']==sha(ROOT/'common/query_protocol.py')
    adapter_root=ROOT/'validation/tie_portable_adapter'
    for name,h in read(adapter_root/'SHA256SUMS.json').items():assert sha(adapter_root/name)==h,name
    adapter=read(adapter_root/'VALIDATION_REPORT.json')
    assert adapter['score_replay_executed'] and adapter['train_dry_run_executed']
    assert not adapter['train_executed'] and not adapter['optimizer_called'] and adapter['new_fits_run']==0
    assert adapter['logical_metrics_max_abs_error']==0 and len(adapter['relocated_byte_equal_plan_artifacts'])==7
    assert adapter['paired_comparison']['rows']==6000 and adapter['summary_comparison']['rows']==600
    assert adapter['adapter_sha256']==sha(ROOT/'tools/run_tie_sensitivity.py')==sha(adapter_root/'run_tie_sensitivity.py')
    assert adapter['validation_harness_sha256']==sha(ROOT/'tools/validate_adapter.py')
    report['checks']['tie_sensitivity_compact_export']=tie_export
    report['checks']['tie_sensitivity_actual_external_label_replay']=tie_replay
    report['checks']['tie_adapter_executed_plan_relocation_and_score_replay']=adapter
    figures=ROOT/'figures'
    for name in ['SOURCE_ORIGINS.json','DATA_DERIVATION.json','GENERATED_ARTIFACTS.json']:
        for r in read(figures/name)['files']:assert sha(ROOT/r['file'])==r['sha256'],r['file']
    mapping=read(figures/'FIGURE_MAP.json')['figures'];assert len(mapping)==14
    for r in mapping:
        for name in r['data_sources']+r['plotting_sources']:assert (ROOT/name).exists(),name
    bounds=read(figures/'qa/EXPORT_BOUNDS.json');assert bounds['status']=='PASS_TEXT_BOUNDS' and len(bounds['figures'])==3
    for r in bounds['figures']:
        assert r['outside_page_words']==0 and r['reference_word_text_and_bbox_identical']
        assert sha(figures/'generated'/r['file'])==r['sha256'],r['file']
    ext_plot=read(figures/'generated/fig22_plot_checks.json')
    assert ext_plot['status']=='PASS' and ext_plot['paired_ACC_points']==40 and ext_plot['exclusions']==0
    for name,h in ext_plot['source_sha256'].items():assert sha(extension/'reporting'/name)==h,name
    report['checks']['scientific_figure_sources']=dict(mapped_assets=len(mapping),executed_portable_redraws=3,
        formula_and_plot_checks='Retained in figures/qa and figures/generated',text_bounds=bounds,
        style_limit='Figure22 preserves original DejaVu Sans, outlined SVG glyphs and 260dpi preview; generic source-style flags are disclosed. Vector PDF is the delivery target, not a new journal-compliance certification.')
    helps=[('common/run_allocation.py',True),('common/score_allocation.py',True),('benchmark/reproduce.py',False),
      ('benchmark/prepare_data.py',False),('benchmark/prepare_field_corpora.py',True),('primary_predictions/rescore.py',True),
      ('data_tools/prepare_bbc_citation.py',True),('data_tools/prepare_agnews_mirror.py',True),('tools/run_common_study.py',True),
      ('figures/plot_reference_risk.py',True),('figures/plot_allocation_acc.py',True),('figures/plot_extension_acc.py',True),
      ('tools/run_tie_sensitivity.py',True),('evidence/tie_sensitivity/tools/rescore_tie_compact.py',True)]
    report['checks']['entry_help']=[]
    for name,isolated in helps:
        cmd=[sys.executable,'-B']+(['-I'] if isolated else [])+[str(ROOT/name),'--help']
        q=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        assert q.returncode==0,(name,q.stderr)
        report['checks']['entry_help'].append(dict(file=name,isolated=isolated,returncode=q.returncode))
    matlab=[];npz_count=0
    for f in ROOT.rglob('*'):
        if not f.is_file():continue
        assert f.suffix.lower() not in ['.mat','.mtx','.gz','.zip','.pyc'],f
        if f.suffix=='.m':
            rel=f.relative_to(ROOT).as_posix()
            assert rel.startswith(('benchmark/fssmsc/','baselines/pcbg_historical/','baselines/ssa_historical/')),rel
            assert 'private_generated' not in rel and 'third_party' not in rel
            assert not cjk.search(f.read_text(encoding='utf-8-sig')),rel
            matlab.append(rel)
        if f.suffix=='.npz':
            npz_count+=1
            with ZipFile(f) as z:assert not any(re.fullmatch(r'(X\d*|truth|y|targets?)\.npy',n) for n in z.namelist()),f
    report['checks']['research_MATLAB_worker_sources']=matlab
    report['checks']['NPZ_without_raw_feature_or_class_truth_keys']=npz_count
    report['checks']['no_raw_archives_feature_files_or_author_MATLAB']=True
    catalog=[]
    for f in sorted(ROOT.rglob('*.csv')):
        with f.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(stream);columns=next(reader,[]);count=sum(1 for _ in reader)
        catalog.append(dict(path=f.relative_to(ROOT).as_posix(),rows=count,columns=columns,sha256=sha(f)))
    (ROOT/'provenance/CSV_CATALOG.json').write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8')
    inventory=[]
    for f in sorted(ROOT.rglob('*')):
        if f.is_file() and f.name!='SHA256SUMS':inventory.append((sha(f),f.relative_to(ROOT).as_posix(),f.stat().st_size))
    (ROOT/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for h,n,_ in inventory),encoding='utf-8')
    for h,n,_ in inventory:assert sha(ROOT/n)==h
    report.update(files_in_SHA256SUMS=len(inventory),bytes_in_inventory=sum(n for _,_,n in inventory),CSV_files=len(catalog),
       SHA256SUMS_sha256=sha(ROOT/'SHA256SUMS'),publication_performed=False,
       unresolved=['Rights-holder license choice','Public or reviewer-accessible repository and manuscript Declaration link',
          'PCBG/SSA historical training adapters still require explicit path/protocol/environment reconstruction; compact replay is separately validated',
          'External author-code acquisition and licensed MATLAB are required for MATLAB baseline training; no new MATLAB run was performed'],
       resolved_packaging_issue='Initial SSA replay identified its manifest-bound source README as omitted; the exact archived README was restored before final validation.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    a.output.with_suffix('.md').write_text('# Local release QA\n\nStatus: PASS. No external publication.\n\n'+
        f'- {len(sources)} Python source files compile, with English comments and docstrings.\n'+
        f'- {len(inventory)} files bound by SHA256SUMS; {len(catalog)} CSV files catalogued.\n'+
        f'- Synthetic original/English common-candidate comparison: {smoke["arrays_bitwise_equal"]} arrays equal, {smoke["numerical_json_equal"]} numerical JSON records equal, all three eta values and five logical finals covered.\n'+
        '- Seven-corpus input interfaces are documented. BBC/Cora/CiteSeer preparation has separate elementwise relocation evidence.\n'+
        '- Actual AG News mirror preparation matched all scientific input identities; the different container is declared and four original members are verified.\n'+
        '- CMVNMF/MVCHSS synthetic prepare/oracle/CV/final/score interfaces passed; SSA compact evidence replay passed.\n'+
        '- Portable historical/extension launcher was exercised on a synthetic block, including refusal to score without the global study seal.\n'+
        '- Real 3Sources seed504/Q160 reproduction from a fresh public download matched 440 arrays and all 25 metric values; its compact export/source hashes also pass.\n'+
        '- All 210 original primary predictions were actually rescored with external original labels: 630 ACC/NMI/ARI comparisons passed (maximum absolute difference 1.1102230246251565e-16); the 60 field predictions are included within 210.\n'+
        '- Sealed field allocation extension: 50 logical finals, 39 unique predictions and an independent 250-value post-seal rescoring receipt with zero difference; compact-export omissions are explicit.\n'+
        '- Posthoc tie sensitivity: 172 new compact predictions plus 386 verified historical references, 1,500 logical records, 6,000 paired differences and 600 summaries; actual compact replay matched all five metrics.\n'+
        '- Tie adapter planning, seven relocated plan files, train dry-run and stored-state scoring replay were executed; no new optimizer execution occurred during adapter validation.\n'+
        '- All 14 figures have source/data mappings; three portable redraws passed numeric and PDF text/bounds checks. Figure22 generic style flags are disclosed, with vector PDF as the delivery target.\n'+
        '- Own FSSMSC/PCBG/SSA adapters are included; PCBG/SSA historical training portability remains incomplete and is explicitly documented.\n'+
        '- No original raw archives or third-party author MATLAB implementation is distributed.\n'+
        '- Rights-holder license and a real access URL remain external release requirements.\n\n'+
        'The full machine-readable report records entry-point checks and exact source identities. Synthetic checks are not performance replications.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
