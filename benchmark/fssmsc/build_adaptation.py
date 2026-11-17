"""Generate diagnostic copies; author files remain byte-for-byte untouched."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT.parent / "literature" / "FSSMSC" / "master"


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


original = (AUTHOR / "FSSMSC_bound.m").read_text(encoding="utf-8")
s = replace_once(original, "function [A,B,Z] = FSSMSC_bound(Xp, U, L_M, L_C,nl,para)",
                 "function [A,B,Z,diagnostics] = fssmsc_author_diagnostic(Xp, U, L_M, L_C,nl,para)")
s = replace_once(s, "%%%%%%Initialization", """% Cache constant view products; keep original per-view addition order.
cached_UtX=cell(1,nv);cached_UtU=cell(1,nv);diagnostic_Xnorm=0;
for v=1:nv
    cached_UtX{v}=U{v}'*Xp{v};cached_UtU{v}=U{v}'*U{v};
    diagnostic_Xnorm=diagnostic_Xnorm+sum(Xp{v}(:).^2);
end
diagnostic_cache=struct('UtX',{cached_UtX},'UtU',{cached_UtU},'Xnorm',diagnostic_Xnorm);
%%%%%%Initialization""")
assert s.count("U{v}'*U{v}") == 3
assert s.count("U{v}'*Xp{v}") == 3
# Keep the first occurrence (cache construction), replace the two author uses.
s = s.replace("LHS = LHS + U{v}'*U{v};", "LHS = LHS + cached_UtU{v};")
s = s.replace("RHS = RHS + U{v}'*Xp{v};", "RHS = RHS + cached_UtX{v};")
s = replace_once(s, "iter = 0;", """initial_state = struct('A',A,'B',B,'Z',Z,'q',q,'Lambda',Lambda);
history = struct([]);
iter = 0;""")
s = replace_once(s, "iter = iter + 1;", """iter = iter + 1;
    previous_A = A;""")
s = replace_once(s, "[A,~] = v1_trace_ratio_optim(LU, LL,k,20);",
                 "[A,~,inner_diagnostics] = fssmsc_trace_ratio_diagnostic(LU, LL,k,20);")
s = replace_once(s, "stopC = max([diffZ,diffB]);", """stopC = max([diffZ,diffB]);
    entry = fssmsc_objective_diagnostics(Xp,U,L_M,L_C,nl,para,A,B,Z,q,Lambda,diagnostic_cache);
    entry.iteration = iter;
    entry.delta_Z = diffZ;
    entry.delta_B = diffB;
    entry.delta_A = max(abs(A(:)-previous_A(:)));
    entry.delta_q = max(abs(q(:)-qk(:)));
    entry.stopC = stopC;
    entry.inner = inner_diagnostics;
    if iter==1
        history = entry;
    else
        history(iter) = entry;
    end""")
s += """
diagnostics = struct('initial_state',initial_state,'history',history, ...
    'q',q,'Lambda',Lambda,'iterations',iter,'legacy_stop_reached',stopC<tol, ...
    'method_type','author-operational pair-input adaptation', ...
    'paper_stationarity_claimed',false);
if stopC < tol
    diagnostics.stop_reason = 'author_ZB_change_tolerance';
else
    diagnostics.stop_reason = 'author_iteration_cap';
end
"""
(ROOT / "fssmsc_author_diagnostic.m").write_text(s, encoding="utf-8", newline="\n")

original_t = (AUTHOR / "v1_trace_ratio_optim.m").read_text(encoding="utf-8")
t = replace_once(original_t, "function [Vs,rho] = v1_trace_ratio_optim(A,B,k,max_iter)",
                 "function [Vs,rho,diagnostics] = fssmsc_trace_ratio_diagnostic(A,B,k,max_iter)")
t = replace_once(t, "if iter >= max_iter", """diagnostics = struct('iterations',iter,'ratio',rho, ...
    'ratio_change',abs(rho-rho_before),'denominator',trace(Vs'*B*Vs), ...
    'author_failure_message',iter>=max_iter,'finite',all(isfinite(Vs(:))) && isfinite(rho));
if iter >= max_iter""")
(ROOT / "fssmsc_trace_ratio_diagnostic.m").write_text(t, encoding="utf-8", newline="\n")

files = list(AUTHOR.glob("*.m")) + list(ROOT.glob("*.m"))
manifest = {
    "author_commit": "064f9cba06b30d7b152eb046e909a028e386ded7",
    "branch": "master",
    "author_url": "https://github.com/linghm/FSSMSC/tree/master",
    "changes": ["new function names", "cache constant per-view UtX/UtU products with original addition order", "read-only history and final-state diagnostics",
                "pair-input wrapper replaces class-label demo only; no optimization formulas changed"],
    "files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
}
(ROOT / "ADAPTATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"generated": 2, "manifest": str(ROOT / "ADAPTATION_MANIFEST.json")}))
