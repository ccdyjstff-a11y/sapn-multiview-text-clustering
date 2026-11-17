"""Build a PRIVATE SSA-SNMF diagnostic adapter. Does not invoke MATLAB.

Only the missing L_M definition and Cartesian cannot-link initialization are
algorithmic repairs. The adapter bypasses demonstration label generation and
scoring. Source spans are extracted with normalized line endings and verified
after removing the exact instrumentation/repair insertions. Third-party source
is private analysis material and must not enter a public package.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re

HERE=Path(__file__).resolve().parent
MAIN=HERE.parent.parent
ORIGIN=MAIN/'review_revision_v6/literature_followup'
SRC=ORIGIN/'author_source_private_analysis'
OUT=HERE/'private_generated'
COMMIT='4d37f15621ee7bd9d412ad224408658ad243699c'
HELPERS=['Create_KNNgraph.m','NormalizationK.m','normalization.m','Update_V_v.m','Update_W_v.m','Update_Teta.m','Cost_function.m']
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p): return sha_bytes(p.read_bytes())
def need(v,m):
    if not v: raise ValueError(m)
download=json.loads((ORIGIN/'SSA_download_manifest.json').read_text(encoding='utf-8-sig'))
need(download['commit']==COMMIT,'Unexpected source commit')
expected={r['path']:r['sha256'] for r in download['files']}
closure={name:sha(SRC/name) for name in ['SSA_SNMF.m']+HELPERS}
need(all(closure[n]==expected[n] for n in closure),'Author source changed')
source=(SRC/'SSA_SNMF.m').read_text(encoding='utf-8')
def span(start,end):
    need(source.count(start)==1 and source.count(end)==1,'Source marker changed')
    return source.split(start,1)[1].split(end,1)[0]
original={
 'normalization':span('%% data normalization\n','%% parameters\n'),
 'parameters':span('%% parameters\n','%% creating the similarity matrix M '),
 'graph':span('%% creating the similarity matrix M ','%% generated results '),
 'initialization':span('%% initialization ','%% iterative procses\n'),
 'iterations_aggregation':span('%% iterative procses\n','%% evaluation '),
}
patches=[]
def replace(block,old,new,reason):
    need(block.count(old)==1,'Nonunique replacement: '+reason)
    patches.append(dict(old=old,new=new,reason=reason))
    return block.replace(old,new)
blocks=dict(original)
blocks['graph']=replace(blocks['graph'],'D_M = cell(1,nv);','D_M = cell(1,nv);\nL_M = cell(1,nv);','repair1_allocate_missing_Laplacian')
blocks['graph']=replace(blocks['graph'],'    D_M{v} = diag(sum(M{v}));','    D_M{v} = diag(sum(M{v}));\n    L_M{v} = D_M{v} - M{v};','repair1_define_missing_Laplacian')
blocks['graph']=replace(blocks['graph'],'    M{v}(isnan(M{v}))=0;','    graph_before_clear(v,:) = [nnz(isnan(M{v})), nnz(isinf(M{v}))];\n    M{v}(isnan(M{v}))=0;','observe_graph_before_original_NaN_clear')
for left,right in [('1','2'),('2','1')]:
    old=f"    S{{v}}(cannot_links(:,{left})',cannot_links(:,{right})') = 0;"
    new=f'    S{{v}}(sub2ind([n n],cannot_links(:,{left}),cannot_links(:,{right}))) = 0;'
    blocks['initialization']=replace(blocks['initialization'],old,new,'repair2_pairwise_symmetric_CL_index_'+left+right)
iterblock=blocks['iterations_aggregation']
for slot,(name,line) in enumerate([
 ('S{v}','        S{v}(isnan(S{v})) =0 ;'),
 ('S_tilde{v}','        S_tilde{v}(isnan(S_tilde{v})) =0 ;'),
 ('V{v}','        V{v}(isnan(V{v})) = 0;'),
 ('W{v}','        W{v}(isnan(W{v})) = 0;')],1):
    inserted=f'        clear_counts(itr,v,{slot},1) = nnz(isnan({name}));\n        clear_counts(itr,v,{slot},2) = nnz(isinf({name}));\n'
    iterblock=replace(iterblock,line,inserted+line,'observe_before_original_clear_'+name)
iterblock=replace(iterblock,'    Teta(isnan(Teta))=0;',
    '    theta_clear_counts(itr,:) = [nnz(isnan(Teta)), nnz(isinf(Teta))];\n    Teta(isnan(Teta))=0;','observe_before_original_theta_clear')
marker='   if abs(Of_old - Of_new) <= eps'
observe='''   objective_trace(itr) = Of_new;
   delta_trace(itr) = abs(Of_old - Of_new);
   residual_trace(itr,:) = profile_residuals(V,W,Teta,S,cl_linear,NC);
'''
iterblock=replace(iterblock,marker,observe+marker,'observe_objective_residuals_without_update_change')
blocks['iterations_aggregation']=iterblock
# The complete normalized source spans are recovered by exactly undoing the
# recorded literal substitutions. No whitespace-insensitive comparison is used.
recovered=dict(blocks)
for p in reversed(patches):
    hits=[name for name,value in recovered.items() if p['new'] in value]
    need(len(hits)==1,'Ambiguous repair/instrumentation removal')
    recovered[hits[0]]=recovered[hits[0]].replace(p['new'],p['old'],1)
need(recovered==original,'Non-whitelisted algorithm source drift')

header=r'''function [V_star, diagnostics, state] = ssa_minimal_profile(fea, known_k, ML, CL, seed)
% PRIVATE diagnostic derivative, not an unchanged-author or equation-faithful claim.
% No point labels, oracle lookup, scoring, parameter search or file reads.
assert(iscell(fea) && isvector(fea) && numel(fea)>=2,'SSA:Input','Need at least two aligned views.');
nv = length(fea); n = size(fea{1},1); NC = known_k;
assert(n>=11 && isscalar(NC) && isfinite(NC) && NC==floor(NC) && NC>=2 && NC<=n,'SSA:Input','Invalid dimensions or known k.');
assert(isscalar(seed) && isfinite(seed) && seed>=0 && seed==floor(seed),'SSA:Input','Invalid seed.');
for vv=1:nv
    x = fea{vv};
    assert(isnumeric(x) && isreal(x) && ismatrix(x) && size(x,1)==n && size(x,2)>=2 && all(isfinite(x(:))),'SSA:Input','Features must be finite, aligned and have >=2 dimensions.');
    if any(sum(x.^2,2)==0)
        error('SSA:UnsupportedZeroRow','Original column L2 normalization cannot process a zero object vector.');
    end
end
ML = validate_pairs(ML,n); CL = validate_pairs(CL,n);
assert(isempty(intersect(sort(ML,2),sort(CL,2),'rows')),'SSA:Input','A queried pair has both signs.');
rng(seed,'twister');
graph_before_clear=zeros(nv,2);
'''
query=r'''
% Explicit anonymous pair interface. Unknown pairs retain the author's zero Z prior.
Z = zeros(n);
if ~isempty(ML)
    Z(sub2ind([n n],ML(:,1),ML(:,2)))=1;
end
Z = (Z+Z');
Z = Z + eye(n);
cannot_links=CL;
cl_linear=[sub2ind([n n],CL(:,1),CL(:,2)); sub2ind([n n],CL(:,2),CL(:,1))];
'''
before_iteration=r'''
initial_cl_nonzeros=zeros(1,nv);
for vv=1:nv, initial_cl_nonzeros(vv)=nnz(S{vv}(cl_linear)); end
objective_trace=nan(maxitr,1); delta_trace=nan(maxitr,1);
clear_counts=zeros(maxitr,nv,4,2);
theta_clear_counts=zeros(maxitr,2);
residual_trace=nan(maxitr,4);
'''
footer=r'''
% This observes, but does not replace, the original stopping condition.
[~,predicted] = max(V_star,[],2);
finite_state=all(isfinite(V_star(:))) && isreal(V_star) && all(V_star(:)>=0);
for vv=1:nv
    finite_state=finite_state && all(isfinite(S{vv}(:))) && isreal(S{vv}) && all(S{vv}(:)>=0) ...
        && all(isfinite(V{vv}(:))) && isreal(V{vv}) && all(V{vv}(:)>=0) ...
        && all(isfinite(W{vv}(:))) && isreal(W{vv}) && all(W{vv}(:)>=0);
end
finite_state=finite_state && all(isfinite(Teta(:))) && isreal(Teta) && all(Teta(:)>=0);
if isfinite(delta_trace(itr)) && delta_trace(itr)<=eps
    stop_reason='author_absolute_objective_change';
else
    stop_reason='author_iteration_cap';
end
diagnostics=struct('profile','SSA-SNMF-minimal-two-repair-author-profile', ...
    'author_commit','4d37f15621ee7bd9d412ad224408658ad243699c', ...
    'n',n,'views',nv,'known_k',NC,'seed',seed,'ML_count',size(ML,1),'CL_count',size(CL,1), ...
    'parameters',struct('kap',kap,'beta',beta,'lam',lam,'knn',k,'alpha',alpha,'gam',gam,'et',et,'maxitr',maxitr,'stop_abs_delta',eps), ...
    'actual_iterations',itr,'stop_reason',stop_reason,'objective',objective_trace(1:itr), ...
    'objective_delta',delta_trace(1:itr),'first_delta_is_infinite_by_author_initialization',true, ...
    'graph_before_clear_NaN_Inf',graph_before_clear, ...
    'before_clear_S_Stilde_V_W_NaN_Inf',clear_counts(1:itr,:,:,:), ...
    'before_clear_Theta_NaN_Inf',theta_clear_counts(1:itr,:), ...
    'residual_columns',{{'max_V_row_sum_residual','max_W_orthogonality_Frobenius','max_Theta_row_sum_residual','max_CL_absolute_S_entry'}}, ...
    'residuals',residual_trace(1:itr,:), ...
    'initial_CL_nonzeros',initial_cl_nonzeros,'final_CL_max_abs',residual_trace(itr,4), ...
    'actual_clusters',numel(unique(predicted)),'finite_nonnegative_state',finite_state, ...
    'objective_all_finite',all(isfinite(objective_trace(1:itr))), ...
    'full_class_truth_used',false,'clustering_scores_computed',false);
diagnostics.operational_contract=finite_state && diagnostics.objective_all_finite ...
    && diagnostics.actual_clusters==NC && all(initial_cl_nonzeros==0) && diagnostics.final_CL_max_abs==0;
diagnostics.operational_contract_note='Finite nonnegative state, finite recorded objective, requested observed k, and exact initialized CL zero-lock. Not original-paper hard-domain feasibility, stationarity, convergence or performance qualification.';
state=struct('S',{S},'S_tilde',{S_tilde},'V',{V},'W',{W},'Theta',Teta,'Z',Z, ...
    'M',{M},'D_M',{D_M},'L_M',{L_M},'predicted',predicted);
end

function pairs=validate_pairs(pairs,n)
assert(isnumeric(pairs) && isreal(pairs) && ismatrix(pairs) && size(pairs,2)==2 && all(isfinite(pairs(:))) ...
    && all(pairs(:)==floor(pairs(:))) && all(pairs(:)>=1) && all(pairs(:)<=n),'SSA:Input','Pairs must be n-by-2 valid 1-based indices.');
pairs=double(pairs);
assert(all(pairs(:,1)~=pairs(:,2)) && size(unique(sort(pairs,2),'rows'),1)==size(pairs,1),'SSA:Input','Repeated or self pairs are not allowed.');
end

function r=profile_residuals(V,W,Teta,S,cl_linear,NC)
r=[0 0 max(abs(sum(Teta,2)-1)) 0];
for vv=1:numel(V)
    r(1)=max(r(1),max(abs(sum(V{vv},2)-1)));
    r(2)=max(r(2),norm(W{vv}*W{vv}'-eye(NC),'fro'));
    if ~isempty(cl_linear), r(4)=max(r(4),max(abs(S{vv}(cl_linear)))); end
end
end
'''
adapter=header+blocks['normalization']+blocks['parameters']+blocks['graph']+query+blocks['initialization']+before_iteration+blocks['iterations_aggregation']+footer
need(not re.search(r'\b(gt|Y|SelLabSam_Semi_2|Clustering8Measure)\b',adapter),'Oracle/demo dependency entered adapter')
need(not OUT.exists(),'Never overwrite a generated profile')
OUT.mkdir()
(OUT/'ssa_minimal_profile.m').write_text(adapter,encoding='utf-8',newline='\n')
for name in HELPERS:
    with (OUT/name).open('xb') as f: f.write((SRC/name).read_bytes())
manifest=dict(kind='PRIVATE_SSA_MINIMAL_PROFILE_PREPARATION',status='PREPARED_NOT_EXECUTED',created_utc=datetime.now(timezone.utc).isoformat(),
    preparer_sha256=sha(Path(__file__)),synthetic_driver_sha256=sha(HERE/'run_ssa_synthetic_qualification.m'),
    author_commit=COMMIT,source_closure_sha256=closure,source_normalization='CRLF/CR normalized to LF by Python universal-newline decoding before exact span extraction.',
    source_spans={name:dict(normalized_utf8_sha256=sha_bytes(value.encode()),characters=len(value),exact_recovery_after_recorded_substitutions=True) for name,value in original.items()},
    literal_substitutions=patches,generated_profile_sha256=sha(OUT/'ssa_minimal_profile.m'),
    copied_helper_sha256={name:sha(OUT/name) for name in HELPERS},
    preserved_author_differences=['Theta norm power, fixed eta and absent normalization','Identity-initialized multiplicative W update','Union graph support and bandwidth','Original penalty-based V/W constraints and stopping objective','Original NaN/Inf clearing and max500/absolute0.001 stop'],
    interface_changes=['Known k supplied independently of labels','Z constructed only from provided anonymous ML plus identity; unknown entries remain zero','CL supplied directly without demo point-label enumeration','Feature/shape/pair validation and explicit zero-row unsupported result','Seed fixed explicitly; diagnostics do not draw random numbers'],
    synthetic_plan=dict(n=24,view_dimensions=[3,4,5],known_k=3,data_seed=90210,fit_seed=48272,cases=['mixed','ML_only','CL_only','mixed_repeat','zero_row_rejected_before_fit'],maximum_algorithm_calls=4),
    real_corpus_fits=0,matlab_invoked=False,source_originals_modified=False,private_only_no_distribution_license_inferred=True)
with (HERE/'PREPARATION.json').open('x',encoding='utf-8') as f: json.dump(manifest,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=manifest['status'],generated_profile_sha256=manifest['generated_profile_sha256'],helper_files=len(HELPERS),exact_source_span_recovery=True),indent=2))
