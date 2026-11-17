function pcbg_formal_worker(inputPath, outputDir, authorRoot)
% Fixed-profile native qualification. Never calls demo, oracle, or scores.
here = fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(here);
addpath(authorRoot);
addpath(fullfile(authorRoot,'funs'));
addpath(fullfile(authorRoot,'funsYU'));
cd(outputDir);
summary = struct('status','STARTED','execution_returned',false, ...
    'finite_partition_retained',false,'benchmark_contract_met',false, ...
    'formula_exactness_claimed',false,'stage','path_audit');
fitRaw = struct();
wall = tic;
try
    maxNumCompThreads(1);
    names = {'PCBG','Precondition','Organize_PCsYU','BipartiteGraph_YU', ...
      'graphgen_anchor','L2_distance_1','SolveY_bestYU','Solve_yk', ...
      'SimplexQP_acc','EProjSimplex_new','Optimal_integration4','eig1'};
    relative = {'PCBG.m','funsYU/Precondition.m','funsYU/Organize_PCsYU.m', ...
      'funsYU/BipartiteGraph_YU.m','funs/graphgen_anchor.m','funs/L2_distance_1.m', ...
      'SolveY_bestYU.m','Solve_yk.m','funs/SimplexQP_acc.m', ...
      'funs/EProjSimplex_new.m','Optimal_integration4.m','funs/eig1.m'};
    paths = struct();
    for j=1:numel(names)
        hits=which(names{j},'-all');
        if ischar(hits); hits={hits}; end
        expected=fullfile(authorRoot,strrep(relative{j},'/',filesep));
        assert(numel(hits)==1 && strcmpi(normalpath(hits{1}),normalpath(expected)), ...
          'PCBGQ:PathMismatch','Unexpected resolution for %s',names{j});
        paths.(names{j})=hits;
    end
    % Training never invokes these functions even if root folder contains demo.
    summary.path_audit=paths;
    summary.matlab_version=version;
    summary.platform=computer;
    vars=whos('-file',inputPath);
    expected={'X','c','ML','CL','seed','config','case_id'};
    assert(isequal(sort({vars.name}),sort(expected)),'PCBGQ:InputSchema','Unexpected MAT variables');
    d=load(inputPath,'X','c','ML','CL','seed','config','case_id');
    summary.case_id=d.case_id;summary.stage='input_checks';
    X=d.X; c=d.c; ML=d.ML; CL=d.CL; opts=d.config;
    assert(iscell(X)&&numel(X)>=2);
    n=size(X{1},1);
    assert(c==fix(c)&&c>=2&&c<n);
    checkpairs(ML,CL,n);
    expectedConfig=struct('anchor_rate',.7,'gama',.1,'Iter',30,'style',0,'IterMax',20, ...
      'islocal',1,'Organize_CL',1,'w',1,'Theorem2',1,'Theorem3',1,'Isbreak_Loss',1,'Convergence',1e-4);
    assert(isscalar(opts.anchor_rate) && any(opts.anchor_rate==[.7,.1,.3]), ...
        'PCBGQ:AnchorGrid','Undeclared anchor rate');
    expectedConfig.anchor_rate=opts.anchor_rate;
    keys=fieldnames(expectedConfig);
    assert(isequal(sort(fieldnames(opts)),sort(keys)),'PCBGQ:ConfigSchema','Config fields differ');
    for j=1:numel(keys);assert(isequal(opts.(keys{j}),expectedConfig.(keys{j})),'PCBGQ:ConfigMismatch','Config differs');end
    opts.k=max(3,min(10,fix(opts.anchor_rate*.5*n/c)));
    m=fix(n*opts.anchor_rate);
    assert(m>=opts.k+1 && m>=c+1,'PCBGQ:Dimension','Anchor dimension too small');
    rng(d.seed,'twister');summary.rng_initial=rng;
    summary.n=n;summary.c=c;summary.m=m;summary.k=opts.k;
    summary.native_options=opts;
    summary.preprocess=cell(numel(X),1);
    summary.stage='author_preprocessing';pt=tic;
    for v=1:numel(X)
        assert(size(X{v},1)==n && isreal(X{v}),'PCBGQ:ViewShape','Invalid feature view');
        vals=nonzeros(X{v});
        assert(all(isfinite(vals))&&all(vals>=0),'PCBGQ:RawFeatures','Nonfinite/negative raw features');
        zero=find(sum(X{v}.^2,2)==0);
        summary.preprocess{v}=struct('shape',size(X{v}),'zero_rows_before',zero(:)', ...
          'finite_before',true);
        X{v}=Precondition(X{v},2);
        summary.preprocess{v}.finite_after=all(isfinite(nonzeros(X{v})));
    end
    summary.preprocessing_seconds=toc(pt);
    assert(all(cellfun(@(r)r.finite_after,summary.preprocess)), ...
        'PCBGQ:AuthorPreprocessNonfinite','Native row normalization produced nonfinite features');
    summary.stage='relation_organization';
    [organizedML,MLgroups,t,organizedCL,CLgroups,CLgroups1]=Organize_PCsYU(ML,CL);
    summary.relations=struct('original_ML',size(ML,1),'original_CL',size(CL,1), ...
      'organized_ML',size(organizedML,1),'organized_CL',size(organizedCL,1),'virtual_anchors',t);
    fitRaw.organizedML=organizedML;fitRaw.organizedCL=organizedCL;
    fitRaw.MLgroups=MLgroups;fitRaw.CLgroups=CLgroups;fitRaw.CLgroups1=CLgroups1;
    summary.stage='separate_initializer_diagnostic';it=tic;
    [diagnosticA,diagnosticB,diagnosticIndices]=BipartiteGraph_YU(X,opts.anchor_rate,opts);
    summary.initializer_diagnostic_seconds=toc(it);
    summary.initializer_diagnostic_is_extra_call=true;
    summary.initializer=struct('anchor_indices',diagnosticIndices, ...
        'unique_anchor_count',numel(unique(diagnosticIndices)),'views',[]);
    bv=cell(numel(diagnosticB),1);
    for v=1:numel(diagnosticB)
        z=diagnosticB{v};
        bv{v}=struct('finite',all(isfinite(z(:))),'zero_degree_rows',find(sum(z,2)==0)', ...
          'zero_degree_columns',find(sum(z,1)==0),'minimum',min(z(:)), ...
          'row_sum_max_abs_residual',max(abs(sum(z,2)-1)));
    end
    summary.initializer.views=bv;
    fitRaw.diagnosticIndices=diagnosticIndices;
    fitRaw.diagnosticB=diagnosticB;
    assert(all(cellfun(@(r)r.finite,bv)),'PCBGQ:InitializerNonfinite','Author initializer nonfinite');
    summary.stage='native_fit';ft=tic;
    [P,Y,alpha,A,label,P0,B,Loss,iter]=PCBG(X,c,opts.anchor_rate,ML,CL,opts.gama,opts.Iter,opts);
    summary.native_fit_seconds=toc(ft);
    summary.execution_returned=true;summary.stage='terminal_diagnostics';
    fitRaw.P=P;fitRaw.Y=Y;fitRaw.alpha=alpha;fitRaw.A=A;
    fitRaw.label=label;fitRaw.P0=P0;fitRaw.B=B;fitRaw.Loss=Loss;fitRaw.iter=iter;
    summary.numeric_outputs_finite=all(isfinite(P(:)))&&all(isfinite(Y(:))) ...
      &&all(isfinite(alpha(:)))&&all(isfinite(P0(:)))&&all(isfinite(Loss(:))) ...
      &&all(isfinite(label(:)))&&all(isfinite(iter(:)));
    summary.finite_partition_retained=all(isfinite(label(:)))&&numel(label)==n ...
      &&all(label(:)==fix(label(:)))&&all(label(:)>=1);
    summary.output_shapes_valid=isequal(size(P),[n,m]) && isequal(size(P0),[n,t]) ...
      &&isequal(size(Y),[n+m+t,size(organizedCL,1)]) ...
      &&isequal(size(alpha),[numel(X),iter(1)+1]) && isequal(size(label),[n,1]) ...
      &&iscell(B)&&numel(B)==numel(X)&&all(cellfun(@(z)isequal(size(z),[n,m]),B)) ...
      &&size(Loss,2)==3&&size(Loss,1)>=1&&isequal(size(iter),[1,2]);
    summary.native_iterations=iter;
    summary.outer_cap_reached=(iter(1)>=opts.Iter);
    summary.inner_mean=iter(2);
    summary.inner_exit_reason='NOT_EXPOSED_BY_AUTHOR_API';
    summary.native_stop_certificate='Finite return/iteration metadata only; not stationary or fixed-objective convergence';
    summary.initializer_repeated_B_exact=isequaln(B,diagnosticB);
    summary.author_A_output_empty=all(cellfun(@isempty,A));
    if summary.numeric_outputs_finite && summary.finite_partition_retained && summary.output_shapes_valid
        Q=[P0,P];N=size(Q,1)+size(Q,2);
        S=sparse(N,N);S(1:n,n+1:end)=Q;S(n+1:end,1:n)=Q';
        [comp,sz]=conncomp(graph(S));
        summary.graph_component_count=numel(sz);
        summary.sample_partition_count=numel(unique(label));
        summary.native_label_partition_matches_graph=same_partition(label,comp(1:n)');
        summary.strict_c=(numel(sz)==c && numel(unique(label))==c);
        summary.minimum_P=min(P(:));
        summary.P_total_mass=sum(P(:));
        summary.row_simplex_residual=max(abs(sum(P,2)-1));
        summary.row_simplex_mean_mass=mean(sum(P,2));
        summary.alpha_simplex_residual=max(abs(sum(alpha,1)-1));
        summary.minimum_alpha=min(alpha(:));
        summary.must_link_violations=sum(label(ML(:,1))~=label(ML(:,2)));
        summary.cannot_link_violations=sum(label(CL(:,1))==label(CL(:,2)));
        L=spdiags(sum(S,2),0,N,N)-S;
        harmonic=L*Y;freeRes=0;endpointRes=0;
        assert(size(Y,2)==size(organizedCL,1),'PCBGQ:YShape','CL/Y dimension');
        for j=1:size(organizedCL,1)
            endpoints=organizedCL(j,:);
            free=setdiff(1:N,endpoints);
            freeRes=max(freeRes,max(abs(harmonic(free,j)),[],'all'));
            endpointRes=max(endpointRes,max(abs(sort(Y(endpoints,j))-[-1;1])));
        end
        summary.returned_Y_current_graph_free_residual=freeRes;
        summary.returned_Y_endpoint_residual=endpointRes;
        summary.harmonic_residual_scope='Returned Y precedes final P update; diagnostic, not a stationarity claim';
        summary.benchmark_contract_met=summary.strict_c && summary.minimum_P>=-1e-12 ...
          &&summary.minimum_alpha>=-1e-12&&summary.alpha_simplex_residual<=1e-8 ...
          &&summary.native_label_partition_matches_graph&&summary.initializer_repeated_B_exact;
        summary.status='RETURNED_FINITE';
        if ~summary.benchmark_contract_met;summary.status='RETURNED_FINITE_CONTRACT_FAILED';end
    else
        summary.benchmark_contract_met=false;summary.status='RETURNED_NONFINITE_OR_INVALID';
    end
catch err
    summary.status='EXCEPTION';
    summary.exception_identifier=err.identifier;
    summary.exception_message=err.message;
    summary.exception_stack=err.stack;
end
summary.wall_seconds_inside_worker=toc(wall);
summary.rng_final=rng;
save(fullfile(outputDir,'native_outputs.mat'),'-struct','fitRaw','-v7');
writejson(fullfile(outputDir,'worker_summary.json'),summary);
fprintf('PCBG_QUALIFICATION_STATUS: %s\n',summary.status);
end

function out=normalpath(p)
out=strrep(char(p),'\','/');
end
function tf=same_partition(a,b)
a=a(:);b=b(:);[~,~,aa]=unique(a);[~,~,bb]=unique(b);
tf=true;
for j=1:max(aa);if numel(unique(bb(aa==j)))~=1;tf=false;return;end;end
for j=1:max(bb);if numel(unique(aa(bb==j)))~=1;tf=false;return;end;end
end
function checkpairs(ML,CL,n)
for item={ML,CL}
    a=item{1};assert(size(a,2)==2&&all(a(:)==fix(a(:))) ...
      &&all(a(:)>=1)&&all(a(:)<=n)&&all(a(:,1)~=a(:,2)), ...
      'PCBGQ:Pairs','Invalid pair list');
    assert(size(unique(sort(a,2),'rows'),1)==size(a,1), ...
      'PCBGQ:Pairs','Repeated unordered pair within sign');
end
assert(isempty(intersect(sort(ML,2),sort(CL,2),'rows')),'PCBGQ:Pairs','Conflicting signed pair');
parent=1:n;
for j=1:size(ML,1)
    a=ML(j,1);while parent(a)~=a;a=parent(a);end
    b=ML(j,2);while parent(b)~=b;b=parent(b);end
    parent(a)=b;
end
for j=1:size(CL,1)
    a=CL(j,1);while parent(a)~=a;a=parent(a);end
    b=CL(j,2);while parent(b)~=b;b=parent(b);end
    assert(a~=b,'PCBGQ:Pairs','Contradictory closure');
end
end
function writejson(path,obj)
f=fopen(path,'w','n','UTF-8');
assert(f>=0);
cleanup=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(obj));
end

