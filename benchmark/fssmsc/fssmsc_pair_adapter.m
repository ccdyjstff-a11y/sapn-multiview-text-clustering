function result = fssmsc_pair_adapter(input_path,output_path)
% Only X, purchased P/N (one-based unordered edges), k, seed and parameters
% are admitted. No class labels or final evaluation occur in this function.
here=fileparts(mfilename('fullpath'));
author=fullfile(here,'..','literature','FSSMSC','master');
addpath(author);addpath(here);
payload=load(input_path);
allowed={'X','common_path','P','N','k','seed','m','para','input_id'};
assert(all(ismember(fieldnames(payload),allowed)),'Unexpected input field; class labels are prohibited.');
required={'P','N','k','seed','m','para'};
assert(all(isfield(payload,required)),'Missing required input.');
k=double(payload.k);
assert(xor(isfield(payload,'X'),isfield(payload,'common_path')),'Provide X or a frozen common cache.');
if isfield(payload,'common_path')
    common=load(payload.common_path);
    assert(common.seed==payload.seed && common.k==k && common.m==payload.m);
else
    common=fssmsc_make_common(payload.X,k,payload.m,payload.seed);
end
X=common.X;n=common.n;
assert(k>=2 && k==fix(k) && k<=n);
assert(all(cellfun(@(x)size(x,1)==n,X)));
assert(all(cellfun(@(x)all(isfinite(x(:))) && isreal(x),X)));
P=validate_edges(payload.P,n);N=validate_edges(payload.N,n);
assert(~isempty(P) && ~isempty(N),'Both constraint signs required by the normalized trace-ratio protocol.');
assert(isempty(intersect(P,N,'rows')),'Contradictory duplicate queried edge.');
assert(payload.m>=k && payload.m<=n && payload.m==fix(payload.m),'Anchor count must be frozen within k..n.');
p=payload.para;p.k=k;p.C_q=n;
fields={'maxiter','tol','lambda','lambda0','epsilon','epsilon_q','mZ','lambda_Z','lambda_M','beta'};
assert(all(isfield(p,fields)),'Parameters must be declared, not inferred from accuracy.');
for j=1:numel(fields)
    v=p.(fields{j});assert(isscalar(v)&&isfinite(v)&&v>0);
end
assert(p.mZ==1,'This adapter preserves the released mZ=1 normalization domain.');
query_vertices=unique([P(:);N(:)]);nl=numel(query_vertices);
permutation=[query_vertices;setdiff((1:n)',query_vertices)];
inverse_permutation=zeros(n,1);inverse_permutation(permutation)=1:n;
Pt=reshape(inverse_permutation(P),size(P));Nt=reshape(inverse_permutation(N),size(N));
WP=sparse([Pt(:,1);Pt(:,2)],[Pt(:,2);Pt(:,1)],1,nl,nl);
WN=sparse([Nt(:,1);Nt(:,2)],[Nt(:,2);Nt(:,1)],1,nl,nl);
% Normalization uses the explicitly supplied directed entries, with no
% self-pairs or inferred labels added. Unknown pairs remain exactly zero.
LM=(spdiags(sum(WP,2),0,nl,nl)-WP)/nnz(WP);
LC=(spdiags(sum(WN,2),0,nl,nl)-WN)/nnz(WN);
Xp=cell(size(X));U=common.U;
for v=1:numel(X)
    Xp{v}=X{v}(permutation,:)';
end
core_start=tic;
[A,B,Z,diagnostics]=fssmsc_author_diagnostic(Xp,U,LM,LC,nl,p);
core_seconds=toc(core_start);
assert(diagnostics.history(end).finite,'Nonfinite author terminal state.');
A_for_clustering=normr(A')';
F=normr((A_for_clustering*Z)');
result=struct('method','FSSMSC (author optimization, pair-input clustering adapter)','A',A,'B',B,'Z',Z, ...
    'F',F,'L_M',LM,'L_C',LC,'permutation',permutation, ...
    'P',P,'N',N,'k',k,'seed',payload.seed,'m',payload.m,'nl',nl,'para',p, ...
    'diagnostics',diagnostics,'anchors_converged',common.anchors_converged, ...
    'timing',struct('anchors_cached',common.anchor_seconds,'core',core_seconds), ...
    'status','embedding_completed','test_metrics_computed',false);
if isfield(payload,'common_path'),result.common_path=payload.common_path;end
if isfield(payload,'input_id'),result.input_id=payload.input_id;end
if nargin>1 && ~isempty(output_path)
    save(output_path,'-struct','result','-v7');
end
end

function pairs=validate_edges(raw,n)
pairs=double(raw);
assert(ismatrix(pairs)&&size(pairs,2)==2,'Edges must be Q by 2.');
assert(all(isfinite(pairs(:)))&&all(pairs(:)==fix(pairs(:)))&&all(pairs(:)>=1)&all(pairs(:)<=n));
pairs=sort(pairs,2);
assert(all(pairs(:,1)<pairs(:,2)),'No self-query is allowed.');
assert(size(unique(pairs,'rows'),1)==size(pairs,1),'Duplicate queried pairs.');
pairs=sortrows(pairs);
end
