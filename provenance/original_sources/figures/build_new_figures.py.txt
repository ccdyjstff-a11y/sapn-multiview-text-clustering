from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent;MAIN=ROOT.parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
COLORS=['#0072B2','#D55E00','#009E73','#7A5195','#AA4499'];DATA=['3Sources','BBC','Reuters','Cora','CiteSeer']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={}
def save(fig,name,source,description):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
    manifest[name]=dict(description=description,source_hashes={str(p.relative_to(MAIN)):sha(p) for p in source})
    plt.close(fig)

def model():
    fig,ax=plt.subplots(figsize=(7,3.8));ax.set_xlim(0,10);ax.set_ylim(0,5.5);ax.axis('off')
    def box(x,y,w,h,title,sub,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.08',facecolor=color,edgecolor='#64748B',lw=.7))
        ax.text(x+w/2,y+h*.67,title,ha='center',va='center',fontsize=8,fontweight='bold');ax.text(x+w/2,y+h*.27,sub,ha='center',va='center',fontsize=6.5)
    def arrow(a,b,color='#475569',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=color,lw=1,linestyle=style,connectionstyle='arc3'))
    box(.15,2.65,1.65,1.0,'Views $X_v$','Equal energy','#F1F5F9')
    box(2.65,2.65,2.3,1.0,'View-specific factors','Basis $U_v$; codes $H_v$','#E7F3EC')
    box(6.15,2.65,1.75,1.0,'Consensus $H$','Simplex columns','#E5F0FA')
    box(8.65,2.75,1.1,.8,'Clusters',r'$\arg\max H$','#F1F5F9')
    arrow((1.9,3.15),(2.55,3.15));arrow((5.05,3.15),(6.05,3.15));arrow((8,3.15),(8.55,3.15))
    ax.text(5.55,3.48,'Coupling',ha='center',fontsize=7)
    box(3.25,4.45,4.0,.75,'Must-link / cannot-link queries','Fixed supervision shares; $\eta=0.5$','#FFF1DE')
    arrow((4.1,4.35),(3.8,3.75),'#B36B00');arrow((6.5,4.35),(6.9,3.75),'#B36B00')
    box(.25,.55,2.45,1.0,'Initialize & align','Frozen reference $R$','#F1F5F9')
    box(3.35,.55,2.3,1.0,'Fixed graph $L$','Query coverage $u$','#F1F5F9')
    box(6.35,.55,3.15,1.0,'Reference geometry','Consensus / correction smoothing','#EEEAF7')
    arrow((1,2.55),(1.4,1.55),style='--');arrow((1.8,2.55),(4,.55+1.0),style='--')
    ax.plot([1.4,1.4,7.9],[.45,.15,.15],color='#7A5195',lw=1)
    arrow((7.9,.15),(7.9,.45),'#7A5195');arrow((5.75,1.1),(6.25,1.1),'#7A5195');arrow((7.1,1.65),(7.1,2.55),'#7A5195')
    save(fig,'fig00_model',[],'Model schematic; all edges show model dependencies, not empirical effect sizes.')

def location():
    folder=MAIN/'coverage_location_control_v2'
    if not (folder/'INDEPENDENT_FULL_AUDIT.json').exists():return
    assert read(folder/'INDEPENDENT_FULL_AUDIT.json')['status']=='PASS'
    s=read(folder/'summary.json');fig,axs=plt.subplots(1,2,figsize=(7,3.3),gridspec_kw={'width_ratios':[1,1.7]})
    ax=axs[0]
    for i,ds in enumerate(DATA):
        rows=[r for r in s['cells'] if r['method']=='coverage' and r['dataset']==ds]
        x=np.array([v for r in rows for v in r['u_expected']]);y=np.array([v for r in rows for v in r['u_actual']])
        ax.scatter(x,y,s=18,color=COLORS[i],label='Reuters-local' if ds=='Reuters' else ds,zorder=3)
    ax.plot([0,.5],[0,.5],color='#777777',ls='--',lw=.8);ax.set(xlabel='Location-blind expected $u$',ylabel='Observed $u$',xlim=(-.02,.5),ylim=(-.02,.5));ax.set_title('(a) Reference strengths',loc='left');ax.legend(frameon=False,fontsize=6.5)
    ax=axs[1];rows=s['paired'];order=[(d,q) for d in DATA for q in [160,400]]
    for i,key in enumerate(order):
        r=next(r for r in rows if (r['dataset'],r['Q'])==key);a=r['ACC']['deltas_pp']
        ax.scatter(a,np.array([i]*3)+[-.12,0,.12],s=17,color=COLORS[i//2],alpha=.6)
        ax.scatter(r['ACC']['mean_pp'],i,s=38,marker='D',facecolor='white',edgecolor='black',zorder=4)
    ax.axvline(0,color='#555555',lw=.8);ax.set_yticks(range(10),[f'{"Reuters-local" if d=="Reuters" else d} | {q}' for d,q in order],fontsize=7);ax.invert_yaxis();ax.set_xlabel('Observed minus expected rule ACC (pp)');ax.set_title('(b) Paired accuracy differences',loc='left');ax.grid(axis='x',alpha=.18)
    fig.tight_layout(w_pad=1.5);save(fig,'fig08_location_control',[folder/'summary.json',folder/'INDEPENDENT_FULL_AUDIT.json'],'Corrected degenerate-null control; all 30 paired seeds, unselected mean summaries.')

def ablation():
    folder=MAIN/'isolated_pair_ablation_v1'
    if not (folder/'INDEPENDENT_FULL_AUDIT.json').exists():return
    assert read(folder/'INDEPENDENT_FULL_AUDIT.json')['status']=='PASS'
    s=read(folder/'summary.json');fig,axs=plt.subplots(1,5,figsize=(7,3.25),sharey=True)
    masks=['without_consensus_pair','without_private_pair','without_all_pair','without_ML','without_CL'];labels=['No consensus\npairs','No view-specific\npairs','No pair\nlosses','No must-link\nlosses','No cannot-link\nlosses']
    for j,(mask,title) in enumerate(zip(masks,labels)):
        ax=axs[j]
        for i,ds in enumerate(DATA):
            r=next(r for r in s['paired'] if (r['dataset'],r['mask'])==(ds,mask));a=r['ACC_deltas_pp']
            ax.scatter(a,np.array([i]*3)+[-.12,0,.12],s=15,color=COLORS[i],alpha=.6)
            if r['eligible']:ax.scatter(r['ACC_mean_pp'],i,s=28,marker='D',facecolor='white',edgecolor='black',zorder=4)
        ax.axvline(0,color='#555555',lw=.8);ax.set_title(title,fontsize=8);ax.set_yticks(range(5),['Reuters-local' if d=='Reuters' else d for d in DATA],fontsize=7);ax.grid(axis='x',alpha=.2);ax.set_xlabel('$\Delta$ACC (pp)')
    axs[0].invert_yaxis();fig.tight_layout(w_pad=.7);save(fig,'fig09_isolated_pair_ablations',[folder/'summary.json',folder/'INDEPENDENT_FULL_AUDIT.json'],'Five independent loss masks; original endpoint domains, coverage, and surviving coefficients retained.')

def reference_component():
    folder=MAIN/'reference_fair_selection_expanded_v2'
    assert read(folder/'INDEPENDENT_FULL_AUDIT.json')['status']=='PASS'
    rows=read(folder/'evaluation.json');assert len(rows)==120 and all(r['converged'] for r in rows)
    fig,axs=plt.subplots(2,3,figsize=(7,6.2),sharey=True)
    order=[(d,q) for d in DATA for q in [160,400]];values=[]
    for j,(method,title) in enumerate([('coverage','Coverage reference'),('fixed-half-CV','Fixed reference ($u=0.5$)')]):
        for k,metric in enumerate(['ACC','NMI','ARI']):
            ax=axs[j,k]
            for i,(ds,q) in enumerate(order):
                a=[]
                for seed in [23,24,25]:
                    one=lambda m:next(r for r in rows if (r['dataset'],r['total_queries'],r['seed'],r['method'])==(ds,q,seed,m))
                    a.append(100*(one(method)[metric]-one('graph-only-CV')[metric]))
                values.append(dict(method=method,dataset=ds,Q=q,metric=metric,deltas_pp=a,mean_pp=float(np.mean(a))))
                ax.scatter(a,np.array([i]*3)+[-.12,0,.12],s=12,color=COLORS[i//2],alpha=.6)
                ax.scatter(np.mean(a),i,s=26,marker='D',facecolor='white',edgecolor='black',zorder=4)
            ax.axvline(0,color='#555555',lw=.8);ax.set_yticks(range(10),[f'{"Reuters-local" if d=="Reuters" else d} | {q}' for d,q in order],fontsize=6.7)
            ax.set_title(f'({chr(97+3*j+k)}) {title}\n{metric}',fontsize=8)
            ax.set_xlabel(f'{metric} difference from graph-only (pp)',fontsize=6.6);ax.grid(axis='x',alpha=.18)
    axs[0,0].invert_yaxis();fig.tight_layout(w_pad=.8,h_pad=1.6)
    save(fig,'fig11_reference_component',[folder/'evaluation.json',folder/'INDEPENDENT_FULL_AUDIT.json'],'Reference-containing configurations minus graph-only; all 30 paired seeds per metric; each configuration selects its pair coefficient within budget.')
    manifest['fig11_reference_component']['plotted_values']=values

if __name__=='__main__':
    model();location();ablation();reference_component()
    p=OUT/'new_figure_sources.json';prior=read(p) if p.exists() else {};prior.update(manifest);p.write_text(json.dumps(prior,indent=2),encoding='utf-8')
    print(list(manifest))
