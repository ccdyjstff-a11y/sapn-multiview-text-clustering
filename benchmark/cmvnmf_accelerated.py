"""Coordinate descent for the UNCHANGED CMVNMF objective.

Solver sensitivity, not an additional published baseline. Resumes saved paper-MU
factors and reduces the same objective without adding regularizers/constraints.
"""
import numpy as np
from cmvnmf_paper import CMVNMF

class CMVNMFCoordinate(CMVNMF):
    def continue_fit(self,max_iter=10000,sweeps=3):
        hist=[self.objective()];quiet=0;converged=False;res=self.stationarity();kh=[]
        for it in range(max_iter):
            for v in range(self.V):
                U,H,X=self.U[v],self.H[v],self.X[v]
                gram=H@H.T;cross=np.asarray(X@H.T)
                for _ in range(sweeps):
                    for j in range(self.k):
                        if gram[j,j]>1e-30:U[:,j]=np.maximum(U[:,j]+(cross[:,j]-U@gram[:,j])/gram[j,j],0)
                gram=U.T@U;cross=np.asarray(X.T@U).T
                for b in range(self.V):
                    if b!=v:cross+=self.beta*((self.M-self.C)@self.H[b].T).T
                degree=self.beta*(self.V-1)*self.degree
                for _ in range(sweeps):
                    for j in range(self.k):
                        denom=gram[j,j]+degree
                        H[j]=np.maximum(H[j]+(cross[j]-gram[j]@H-degree*H[j])/np.maximum(denom,1e-30),0)
            J=self.objective()
            if not np.isfinite(J) or J>hist[-1]+1e-9*max(1,abs(hist[-1])):raise RuntimeError('CMVNMF CD increased objective')
            rel=abs(J-hist[-1])/max(1,abs(hist[-1]));hist.append(J)
            if (it+1)%10==0 or it+1==max_iter:
                res=self.stationarity();kh.append(dict(iteration=it+1,kkt=res))
                quiet=quiet+1 if rel<self.tol and res<=self.tol_kkt else 0
                if quiet>=2:converged=True;break
        Hmean=np.mean([h/np.maximum(h.sum(0,keepdims=True),1e-30) for h in self.H],axis=0)
        return Hmean.argmax(0),Hmean,dict(converged=converged,iterations=len(hist)-1,kkt=res,kkt_history=kh,J_hist=hist,
            U_norms=[float(np.linalg.norm(u)) for u in self.U],H_norms=[float(np.linalg.norm(h)) for h in self.H],
            note='same CMVNMF objective; coordinate-descent continuation from paper MU state')
