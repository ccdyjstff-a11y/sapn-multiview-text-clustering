"""Author-owned function snapshot; consumed by the portable replay entry.
Original function ASTs and their immutable public origins are bound in SOURCE_PROVENANCE.json.
This file is not a standalone entry point.
"""

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
