function ssa_formal_worker(inputPath, outputDir, privateRoot)
% Explicit finite-profile SSA adapter. No oracle labels or class scores.
% The launcher verifies both request MAT and referenced source MAT SHA-256.
here=fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(here,'-begin');addpath(privateRoot,'-begin');
cd(outputDir);
summary=struct('status','STARTED','execution_returned',false, ...
    'finite_partition_retained',false,'benchmark_contract_met',false, ...
    'formula_exactness_claimed',false,'stage','path_audit', ...
    'full_class_truth_used',false,'clustering_scores_computed',false, ...
    'profile','SSA-SNMF-minimal-two-repair-author-profile', ...
    'cost_evaluation','trace scalar evaluated with elementwise product');
fitRaw=struct();wall=tic;
try
    maxNumCompThreads(1);
    names={'ssa_formal_profile','Create_KNNgraph','NormalizationK','normalization', ...
        'Update_V_v','Update_W_v','Update_Teta','Cost_function'};
    paths=struct();
    for j=1:numel(names)
        hits=which(names{j},'-all');if ischar(hits);hits={hits};end
        expected=fullfile(privateRoot,[names{j} '.m']);
        assert(numel(hits)==1 && strcmpi(normalpath(hits{1}),normalpath(expected)), ...
            'SSAF:PathMismatch','Unexpected function resolution for %s',names{j});
        paths.(names{j})=hits;
    end
    summary.path_audit=paths;summary.matlab_version=version;summary.platform=computer;
    vars=whos('-file',inputPath);
    assert(isequal(sort({vars.name}),sort({'source_input','source_sha256','config','case_id'})), ...
        'SSAF:RequestSchema','Unexpected request variables');
    request=load(inputPath,'source_input','source_sha256','config','case_id');
    assert(ischar(request.source_input) && isrow(request.source_input) && isfile(request.source_input), ...
        'SSAF:SourcePath','Referenced source MAT does not exist');
    assert(ischar(request.source_sha256) && isrow(request.source_sha256) ...
        && ~isempty(regexp(request.source_sha256,'^[0-9a-f]{64}$','once')), ...
        'SSAF:SourceHash','Expected hexadecimal SHA-256 binding');
    assert(ischar(request.case_id) && isrow(request.case_id) && ~isempty(request.case_id), ...
        'SSAF:CaseId','Invalid request case identifier');
    summary.case_id=request.case_id;summary.source_input=request.source_input;
    summary.source_input_sha256=request.source_sha256;
    summary.source_hash_checked_by='Parent Python launcher before and after fit';
    summary.stage='input_checks';
    vars=whos('-file',request.source_input);
    assert(isequal(sort({vars.name}),sort({'X','c','ML','CL','seed','config','case_id'})), ...
        'SSAF:SourceSchema','Unexpected source MAT variables');
    % Deliberately do not load original source config or case_id.
    d=load(request.source_input,'X','c','ML','CL','seed');
    X=d.X;c=d.c;ML=d.ML;CL=d.CL;opts=request.config;
    assert(isstruct(opts)&&isscalar(opts)&&isequal(sort(fieldnames(opts)),sort({'beta';'lam';'kap'})), ...
        'SSAF:ConfigSchema','Expected beta/lam/kap only');
    assert(isscalar(opts.beta)&&isreal(opts.beta)&&any(opts.beta==[.1,.2,.3]) ...
        &&isscalar(opts.lam)&&isreal(opts.lam)&&any(opts.lam==[.2,.3]) ...
        &&isscalar(opts.kap)&&isreal(opts.kap)&&any(opts.kap==[.8,.9,.95]), ...
        'SSAF:ParameterGrid','Undeclared parameter value');
    assert(iscell(X)&&isvector(X)&&numel(X)>=2,'SSAF:ViewSchema','Expected aligned view cells');
    n=size(X{1},1);nv=numel(X);
    assert(isnumeric(c)&&isreal(c)&&isscalar(c)&&isfinite(c)&&c==fix(c)&&c>=2&&c<n);
    summary.n=n;summary.c=c;summary.views=nv;summary.native_options=opts;
    summary.preprocess=cell(nv,1);
    for v=1:nv
        x=X{v};
        assert(isnumeric(x)&&isreal(x)&&ismatrix(x)&&size(x,1)==n&&size(x,2)>=2, ...
            'SSAF:ViewShape','Invalid feature dimensions');
        values=nonzeros(x);
        assert(all(isfinite(values))&&all(values>=0),'SSAF:Features','Nonfinite or negative raw features');
        zero=find(sum(x.^2,2)==0);
        summary.preprocess{v}=struct('shape',size(x),'zero_rows_before',zero(:)', ...
            'finite_nonnegative_before',true,'normalization','Author column L2 after transposition');
        assert(isempty(zero),'SSA:UnsupportedZeroRow','Original normalization cannot process zero object vectors');
    end
    checkpairs(ML,CL,n);
    summary.relations=struct('original_ML',size(ML,1),'original_CL',size(CL,1));
    summary.seed=d.seed;summary.stage='profile_fit';ft=tic;
    [V_star,diagnostics,state]=ssa_formal_profile(X,c,ML,CL,d.seed,opts);
    summary.native_fit_seconds=toc(ft);summary.execution_returned=true;
    summary.stage='terminal_diagnostics';
    [~,pred]=max(V_star,[],2);
    checks=inspectstate(state,V_star,n,c,nv);
    fitRaw.V_star=V_star;fitRaw.pred=pred;fitRaw.label=pred;
    fitRaw.diagnostics=diagnostics;fitRaw.state_checks=checks;
    summary.numeric_outputs_finite=checks.all_finite && diagnostics.objective_all_finite;
    summary.output_shapes_valid=checks.all_shapes_valid && isequal(size(pred),[n,1]) ...
        &&isequal(size(diagnostics.objective),[diagnostics.actual_iterations,1]);
    summary.finite_partition_retained=isreal(pred)&&all(isfinite(pred(:))) ...
        &&numel(pred)==n&&all(pred(:)==fix(pred(:)))&&all(pred(:)>=1)&&all(pred(:)<=c);
    summary.sample_partition_count=numel(unique(pred));summary.strict_c=summary.sample_partition_count==c;
    summary.native_iterations=diagnostics.actual_iterations;
    summary.outer_cap_reached=diagnostics.actual_iterations>=500;
    summary.stop_reason=diagnostics.stop_reason;
    summary.final_objective=diagnostics.objective(end);
    summary.final_objective_absolute_change=diagnostics.objective_delta(end);
    summary.state_checks=checks;
    summary.residual_columns=diagnostics.residual_columns;
    summary.final_hard_constraint_residuals=diagnostics.residuals(end,:);
    summary.hard_constraint_residuals_are_diagnostic_only=true;
    summary.before_clear_S_Stilde_V_W_NaN_Inf_totals=squeeze(sum(sum(diagnostics.before_clear_S_Stilde_V_W_NaN_Inf,1),2));
    summary.before_clear_Theta_NaN_Inf_totals=sum(diagnostics.before_clear_Theta_NaN_Inf,1);
    summary.graph_before_clear_NaN_Inf=diagnostics.graph_before_clear_NaN_Inf;
    summary.initial_CL_nonzeros=diagnostics.initial_CL_nonzeros;
    summary.final_CL_max_abs=diagnostics.final_CL_max_abs;
    summary.must_link_violations=sum(pred(ML(:,1))~=pred(ML(:,2)));
    summary.cannot_link_violations=sum(pred(CL(:,1))==pred(CL(:,2)));
    summary.benchmark_contract_met=summary.numeric_outputs_finite && summary.output_shapes_valid ...
        &&summary.finite_partition_retained&&summary.strict_c&&checks.nonnegative_where_required ...
        &&checks.all_membership_rows_positive&&diagnostics.operational_contract;
    summary.native_stop_certificate='Recorded author stopping rule or finite iteration cap only; no stationarity, hard-domain feasibility, or convergence claim';
    summary.compact_output_scope='Membership, labels, complete per-iteration diagnostic traces and terminal state checks; dense model matrices are not retained';
    if summary.benchmark_contract_met
        summary.status='RETURNED_FINITE';
    elseif summary.numeric_outputs_finite&&summary.output_shapes_valid&&summary.finite_partition_retained
        summary.status='RETURNED_FINITE_CONTRACT_FAILED';
    else
        summary.status='RETURNED_NONFINITE_OR_INVALID';
    end
catch err
    summary.status='EXCEPTION';summary.exception_identifier=err.identifier;
    summary.exception_message=err.message;summary.exception_stack=err.stack;
end
summary.wall_seconds_inside_worker=toc(wall);summary.rng_final=rng;
save(fullfile(outputDir,'native_outputs.mat'),'-struct','fitRaw','-v7');
writejson(fullfile(outputDir,'worker_summary.json'),summary);
fprintf('SSA_FORMAL_STATUS: %s\n',summary.status);
end

function checks=inspectstate(state,V_star,n,c,nv)
fields={'S','S_tilde','V','W','M','D_M','L_M'};
checks=struct('all_finite',isnumeric(V_star)&&isreal(V_star)&&all(isfinite(V_star(:))), ...
    'all_shapes_valid',isequal(size(V_star),[n,c]), ...
    'nonnegative_where_required',all(V_star(:)>=0), ...
    'all_membership_rows_positive',all(sum(V_star,2)>0), ...
    'minimum_consensus_row_mass',min(sum(V_star,2)), ...
    'minimum_consensus_entry',min(V_star(:)),'matrices',struct());
for i=1:numel(fields)
    name=fields{i};group=state.(name);
    validcell=iscell(group)&&numel(group)==nv;
    checks.all_shapes_valid=checks.all_shapes_valid&&validcell;
    if ~validcell,checks.all_finite=false;continue;end
    records=cell(nv,1);
    for v=1:nv
        value=group{v};expected=[n,n];
        if strcmp(name,'V'),expected=[n,c];end
        if strcmp(name,'W'),expected=[c,c];end
        finite=isnumeric(value)&&isreal(value)&&all(isfinite(value(:)));
        shape=isequal(size(value),expected);
        positive=all(value(:)>=0);
        checks.all_finite=checks.all_finite&&finite;
        checks.all_shapes_valid=checks.all_shapes_valid&&shape;
        if ~strcmp(name,'L_M'),checks.nonnegative_where_required=checks.nonnegative_where_required&&positive;end
        records{v}=struct('shape',size(value),'finite_real',finite,'expected_shape',shape, ...
            'minimum',min(value(:)),'maximum',max(value(:)), ...
            'nonnegative_required',~strcmp(name,'L_M'),'nonnegative',positive, ...
            'minimum_row_mass',min(sum(value,2)));
        % Per-view row masses are diagnostic; only consensus rows must be positive.
    end
    checks.matrices.(name)=records;
end
for item={'Theta','Z'}
    name=item{1};value=state.(name);expected=[nv,nv];if strcmp(name,'Z'),expected=[n,n];end
    finite=isnumeric(value)&&isreal(value)&&all(isfinite(value(:)));shape=isequal(size(value),expected);
    nonnegative=all(value(:)>=0);
    checks.all_finite=checks.all_finite&&finite;checks.all_shapes_valid=checks.all_shapes_valid&&shape;
    checks.nonnegative_where_required=checks.nonnegative_where_required&&nonnegative;
    checks.matrices.(name)=struct('shape',size(value),'finite_real',finite,'expected_shape',shape, ...
        'minimum',min(value(:)),'maximum',max(value(:)),'nonnegative_required',true, ...
        'nonnegative',nonnegative,'minimum_row_mass',min(sum(value,2)));
end
checks.prediction_matches_state=isequal(state.predicted,maxlabels(V_star));
checks.all_shapes_valid=checks.all_shapes_valid&&checks.prediction_matches_state;
end

function pred=maxlabels(V)
[~,pred]=max(V,[],2);
end

function checkpairs(ML,CL,n)
for item={ML,CL}
    a=item{1};assert(isnumeric(a)&&isreal(a)&&size(a,2)==2&&all(isfinite(a(:))) ...
        &&all(a(:)==fix(a(:)))&&all(a(:)>=1)&&all(a(:)<=n)&&all(a(:,1)~=a(:,2)), ...
        'SSAF:Pairs','Invalid pair list');
    assert(size(unique(sort(a,2),'rows'),1)==size(a,1),'SSAF:Pairs','Repeated pair');
end
assert(isempty(intersect(sort(ML,2),sort(CL,2),'rows')),'SSAF:Pairs','Conflicting signed pair');
end

function out=normalpath(path)
out=strrep(char(path),'\','/');
end

function writejson(path,obj)
f=fopen(path,'w','n','UTF-8');assert(f>=0);cleanup=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(obj));
end
