function out = fssmsc_objective_diagnostics(X,U,LM,LC,nl,p,A,B,Z,q,Lambda,cache)
% Read-only diagnostics. These do not define an alternative optimizer.
m=size(Z,1);
D=diag(1./sqrt(q));
AZl=A*Z(:,1:nl);
smooth=trace(A*(eye(m)-D*(Z*Z')*D)*A');
ml=trace(AZl*LM*AZl');
cl=trace(AZl*LC*AZl');
if nargin<12
    recZ=0;recB=0;
    for v=1:numel(X)
        recZ=recZ+0.5*norm(X{v}-U{v}*Z,'fro')^2;
        recB=recB+0.5*norm(X{v}-U{v}*B,'fro')^2;
    end
else
    recZ=.5*cache.Xnorm;recB=recZ;
    for v=1:numel(X)
        recZ=recZ-sum(sum(Z.*cache.UtX{v}))+.5*sum(sum(Z.*(cache.UtU{v}*Z)));
        recB=recB-sum(sum(B.*cache.UtX{v}))+.5*sum(sum(B.*(cache.UtU{v}*B)));
    end
end
degree=0.5*p.lambda0*norm(q-Z*Z'*ones(m,1))^2;
ratio=p.beta*(smooth+p.lambda_M*ml)/(cl+p.epsilon);
regularized=ratio+p.lambda_Z*sum(abs(Z(:)))+degree;
out=struct('paper_objective_at_Z',recZ+regularized, ...
    'augmented_lagrangian',recB+regularized+sum(Lambda(:).*(B(:)-Z(:)))+0.5*p.lambda*norm(B-Z,'fro')^2, ...
    'reconstruction_Z',recZ,'reconstruction_B',recB, ...
    'smooth',smooth,'must_link',ml,'cannot_link',cl,'ratio',ratio, ...
    'degree_penalty',degree,'primal_BZ_max',max(abs(B(:)-Z(:))), ...
    'orthogonality_F',norm(A*A'-eye(size(A,1)),'fro'), ...
    'column_sum_error',max(abs(sum(Z,1)-1)), ...
    'Z_min',min(Z(:)),'Z_max',max(Z(:)), ...
    'q_min',min(q),'q_max',max(q), ...
    'finite',all(isfinite([A(:);B(:);Z(:);q(:);Lambda(:)])));
end
