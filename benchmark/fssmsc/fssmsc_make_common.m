function common=fssmsc_make_common(X,k,m,seed)
% Feature-only author preprocessing and anchor construction, reusable for
% every constraint fold/candidate because neither stage reads supervision.
X=reshape(X,1,[]);n=size(X{1},1);
assert(m>=k && m<=n && m==fix(m));
row_norms=cell(size(X));
for v=1:numel(X)
    assert(size(X{v},1)==n && all(isfinite(X{v}(:))) && isreal(X{v}));
    X{v}=double(X{v});row_norms{v}=sqrt(sum(X{v}.^2,2));
    X{v}=normr(X{v}); % Original author normalization; zero rows remain zero.
end
rng(double(seed),'twister');
anchor_indices=randperm(n,double(m))';
start=tic;
[~,centers,anchors_converged]=litekmeans(cell2mat(X),double(m), ...
    'MaxIter',100,'Replicates',1,'Start',anchor_indices);
widths=cellfun(@(x)size(x,2),X);
Uc=mat2cell(centers,double(m),widths);U=cellfun(@transpose,Uc,'UniformOutput',false);
common=struct('X',{X},'U',{U},'row_norms',{row_norms},'k',k,'m',m,'seed',seed, ...
    'n',n,'anchor_indices',anchor_indices,'anchors_converged',anchors_converged, ...
    'anchor_seconds',toc(start),'normalization','author row-L2; zero rows retained', ...
    'anchor_initialization','uniform randperm without replacement supplied as explicit litekmeans Start');
end
