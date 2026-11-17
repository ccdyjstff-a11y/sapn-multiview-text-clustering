"""Exact controlled variants of the delivered SAPN paper solver.

Private coupling changes only V**(-gamma) to a[v]**gamma in PRIVATE
ML/CL penalties. Consensus supervision remains fixed. The original legacy
branch includes private penalties in E[v], hence in the weight minimizer.
"""
from pathlib import Path
import importlib.util
import numpy as np

MAIN = Path(__file__).resolve().parents[2]
CORE = Path(__file__).resolve().parent/'sapn_text_paper.py'
spec = importlib.util.spec_from_file_location('frozen_paper_core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
SAPNText = core.SAPNText


class NoCorrectionSAPN(SAPNText):
    """Remove only cg*u*tr((H-R)L(H-R)^T), preserving anchor and graph."""
    def removed_value(self, H):
        d = H-self.H_ref
        return float(self.cg*self._reference_strength*np.sum(d*(self.Ggeom@d.T).T))

    def _objective(self, H):
        return super()._objective(H)-self.removed_value(H)

    def _h_terms(self):
        K,T = super()._h_terms()
        return K,T-self.cg*self._reference_strength*(self.Ggeom@self.H_ref.T).T

    def _grad_H(self, H):
        return super()._grad_H(H)-2*self.cg*self._reference_strength*(self.Ggeom@H.T).T

    def _consensus_graph(self, graph, graph_scale):
        if graph is None and self._active_private is None:
            return self.Ggeom,self.cg*(1-self._reference_strength)
        return graph,graph_scale

    def _h_loss(self,H,K,T,coefficients=None,graph=None,graph_scale=0.):
        graph,graph_scale=self._consensus_graph(graph,graph_scale)
        return super()._h_loss(H,K,T,coefficients,graph,graph_scale)

    def _h_block(self,H,K,T,selected=None,coefficients=None,graph=None,graph_scale=0.):
        graph,graph_scale=self._consensus_graph(graph,graph_scale)
        return super()._h_block(H,K,T,selected,coefficients,graph,graph_scale)

    def _endpoint_update(self,H,K,T,coefficients=None,graph=None,graph_scale=0.):
        graph,graph_scale=self._consensus_graph(graph,graph_scale)
        return super()._endpoint_update(H,K,T,coefficients,graph,graph_scale)

    def fit_predict(self,*args,**kwargs):
        state=kwargs.get('state_init')
        if state is not None and state.get('correction_graph_removed') is not True:
            raise ValueError('Continuation must preserve the removed correction graph')
        z,H,q=super().fit_predict(*args,**kwargs)
        q['objective_terms']['geometry']-=self.removed_value(H)
        q['correction_graph_removed']=True
        return z,H,q

    def get_state(self):
        state=super().get_state();state['correction_graph_removed']=True
        return state

    def gradient_diagnostics(self,H):
        K,T=self._h_terms();B=H@H.T;u=self._reference_strength
        gradients=dict(data=2*((K-self.cg*u*np.eye(self.k))@H-(T-self.cg*u*self.H_ref)),
            positive=2*self.cp*(self.Lp@H.T).T,negative=self.cn*self._negative(H)[1],
            orthogonal=4*self.co*(B-np.diag(np.diag(B)))@H,
            geometry=2*self.cg*((1-u)*(self.Ggeom@H.T).T+u*(H-self.H_ref)))
        return dict(tangent_norm={key:float(np.linalg.norm(g-g.mean(axis=0,keepdims=True))) for key,g in gradients.items()},
            query_coverage=float(len(self._endpoints)/self.n),
            entropy=float(-np.mean(np.sum(H*np.log(np.maximum(H,1e-300)),axis=0))),
            mean_max_probability=float(H.max(axis=0).mean()),
            total_pair_coefficient_positive=float(self.cp+sum(self.a[v]**self.gamma*self._view_coefficients(v)[0] for v in range(self.V))),
            total_pair_coefficient_negative=float(self.cn+sum(self.a[v]**self.gamma*self._view_coefficients(v)[1] for v in range(self.V))))


CONFIG=dict(mu=.001,lambda_c=.2,lambda_p=20.,lambda_n=20.,lambda_r='auto',lambda_g=0.,
    gamma=2.,negative_loss='overlap',label_to_views=.5,private_simplex='endpoints',
    geometry_weight=.1,reference_strength='coverage',supervision_weighting='fixed',
    reliability='none',endpoint_interval=5,tol_J=1e-5,tol_kkt=1e-3)

def make_model(method,parameter,seed,graph,max_iter,lambda_o):
    config=dict(CONFIG,lambda_p=parameter,lambda_n=parameter,seed=seed,
                geometry_graph=graph,max_iter=max_iter,lambda_o=lambda_o)
    if method=='private-coupled':config['supervision_weighting']='legacy'
    elif method not in ['full','no-correction']:raise ValueError(method)
    cls=NoCorrectionSAPN if method=='no-correction' else SAPNText
    return cls(**config)
