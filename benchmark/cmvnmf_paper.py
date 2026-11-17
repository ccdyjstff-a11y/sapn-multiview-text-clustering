"""CMVNMF (Zhang et al., AAAI 2015), Eqs. (2)--(4).

Paper-based Python implementation, NOT author code. For aligned datasets the
adapter copies observed object pairs to each cross-view pair in both directions,
and supplies known identity correspondences as must-links. No extra class labels.
The paper returns per-view clusters; normalized mean membership is an explicitly
declared aligned-data output adapter, with per-view outputs retained.
No simplex/ridge/reference penalties are added to the published objective.
"""
import numpy as np
from scipy import sparse

class CMVNMF:
    def __init__(self,beta=1.,max_iter=3000,tol=1e-6,tol_kkt=1e-3,seed=0,identity_links=True):
        if beta<0 or not np.isfinite(beta):raise ValueError('beta must be finite and nonnegative')
        self.beta=beta;self.max_iter=max_iter;self.tol=tol;self.tol_kkt=tol_kkt;self.seed=seed;self.identity_links=identity_links

    def objective(self):
        value=0.
        for X,U,H in zip(self.X,self.U,self.H):
            value+=float(X.multiply(X).sum())+np.sum((U.T@U)*(H@H.T))-2*np.sum(U*np.asarray(X@H.T))
        for a in range(self.V):
            for b in range(a):
                Ha,Hb=self.H[a],self.H[b]
                value+=self.beta*(np.sum((Ha**2+Hb**2)*self.degree[None,:])-2*np.sum(Ha*(self.M@Hb.T).T)+2*np.sum(Ha*(self.C@Hb.T).T))
        return float(value)

    def terms(self,v):
        U,H,X=self.U[v],self.H[v],self.X[v]
        un=np.asarray(X@H.T);ud=U@(H@H.T)
        hn=np.asarray(X.T@U).T;hd=(U.T@U)@H
        for b in range(self.V):
            if b==v:continue
            hn+=self.beta*(self.M@self.H[b].T).T
            hd+=self.beta*(H*self.degree[None,:]+(self.C@self.H[b].T).T)
        return un,ud,hn,hd

    def stationarity(self):
        residuals=[]
        for v in range(self.V):
            un,ud,hn,hd=self.terms(v)
            for A,num,den in ((self.U[v],un,ud),(self.H[v],hn,hd)):
                g=2*(den-num);pg=np.where(A>1e-12,g,np.minimum(g,0))
                residuals.append(float(np.linalg.norm(pg)/max(np.linalg.norm(2*num),1e-12)))
        return max(residuals)

    def fit_predict(self,Xs,P,N,k,H0,Hv0=None):
        self.n=Xs[0].shape[0];self.V=len(Xs);self.k=k
        if self.V<2:raise ValueError('CMVNMF requires at least two views')
        self.X=[]
        for x in Xs:
            x=sparse.csr_matrix(x,dtype=float);energy=float(x.multiply(x).sum())
            if x.shape[0]!=self.n or energy<=0 or np.any(x.data<0) or not np.isfinite(x.data).all():raise ValueError('Invalid views')
            self.X.append((x*np.sqrt(self.n/energy)).T.tocsr())
        def adj(pairs):
            p=np.asarray(pairs,int).reshape(-1,2)
            if len(p) and (p.min()<0 or p.max()>=self.n or np.any(p[:,0]==p[:,1])):raise ValueError('Invalid pairs')
            return sparse.csr_matrix((np.ones(2*len(p)),(np.r_[p[:,0],p[:,1]],np.r_[p[:,1],p[:,0]])),shape=(self.n,self.n))
        self.M=adj(P);self.C=adj(N)
        if self.M.multiply(self.C).nnz:raise ValueError('Contradictory pairs')
        if self.identity_links:self.M=self.M+sparse.eye(self.n,format='csr')
        self.degree=np.asarray(self.M.sum(axis=1)).ravel()
        self.H=[np.maximum(np.asarray(h,float),1e-8).copy() for h in (Hv0 if Hv0 is not None else [H0]*self.V)]
        rng=np.random.RandomState(self.seed);self.U=[]
        for X,H in zip(self.X,self.H):
            if H.shape!=(k,self.n):raise ValueError('Bad initialization')
            U=np.maximum(np.asarray(X@H.T)/np.maximum(H.sum(axis=1),1e-12),1e-10)*(1+1e-4*rng.rand(X.shape[0],k))
            for _ in range(20):U*=np.asarray(X@H.T)/np.maximum(U@(H@H.T),1e-30)
            self.U.append(U)
        hist=[self.objective()];kh=[];quiet=0;converged=False;res=self.stationarity()
        for it in range(self.max_iter):
            for v in range(self.V):
                U,H,X=self.U[v],self.H[v],self.X[v]
                U*=np.asarray(X@H.T)/np.maximum(U@(H@H.T),1e-30)
                _,_,hn,hd=self.terms(v);self.H[v]*=hn/np.maximum(hd,1e-30)
            J=self.objective()
            if not np.isfinite(J) or J>hist[-1]+1e-9*max(1,abs(hist[-1])):raise RuntimeError('CMVNMF objective increased/nonfinite')
            change=abs(J-hist[-1])/max(1,abs(hist[-1]));hist.append(J)
            if (it+1)%10==0 or it+1==self.max_iter:
                res=self.stationarity();kh.append(dict(iteration=it+1,relative_KKT=res))
                quiet=quiet+1 if change<self.tol and res<self.tol_kkt else 0
                if quiet>=2:converged=True;break
        normalized=[h/np.maximum(h.sum(axis=0,keepdims=True),1e-30) for h in self.H]
        consensus=np.mean(normalized,axis=0)
        info=dict(implementation='paper equations; aligned identity/pair/output adapters documented',
            converged=converged,iterations=len(hist)-1,J_hist=hist,kkt=res,kkt_history=kh,
            U_norms=[float(np.linalg.norm(u)) for u in self.U],H_norms=[float(np.linalg.norm(h)) for h in self.H],
            view_labels=[h.argmax(axis=0) for h in self.H],beta=self.beta,identity_links=self.identity_links)
        return consensus.argmax(axis=0),consensus,info
