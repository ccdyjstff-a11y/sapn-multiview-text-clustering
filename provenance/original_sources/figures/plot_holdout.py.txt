"""Four-panel document-generalization plot from qualified, complete source data."""
from pathlib import Path
import json,sys,hashlib,csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
SKILL=Path('C:/Users/LENOVO/.codex/skills/nature-figure')
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

METHODS=['Full','SAPN-EQ','CMVNMF','NoCorr']
COLORS={'Full':'#3268A8','SAPN-EQ':'#9D6E35','CMVNMF':'#698D7E','NoCorr':'#8B7DA7'}
MARKERS={'Full':'o','SAPN-EQ':'s','CMVNMF':'^','NoCorr':'D'}

def verify():
    for name in ['INPUT_BOUND_RESULTS_READY.json','RESULTS_READY.json']:
        ready=read(ROOT/name);assert ready['status']=='PASS'
        for file,digest in ready['files'].items():assert sha(ROOT/file)==digest

def draw_group(ax,x,values,eligible,color,marker):
    vals=np.asarray(values,float);eligible=np.asarray(eligible,bool)
    for j,(value,ok) in enumerate(zip(vals,eligible)):
        if not np.isfinite(value):continue
        ax.scatter(x+[-.035,0,.035][j],value,s=9,marker=marker,edgecolor=color if ok else '#777777',
                   facecolor=color if ok else 'none',linewidth=.45,alpha=.55,zorder=3)
    if eligible.all() and np.isfinite(vals).all():
        ax.errorbar([x],[vals.mean()],yerr=[vals.std(ddof=1)],fmt=marker,color=color,markersize=3.7,
                    markerfacecolor='white',markeredgewidth=.9,elinewidth=.85,capsize=2,capthick=.75,zorder=4)

def main():
    verify();rows=read(ROOT/'evaluation.json');assert len(rows)==120
    index={(r['sample'],r['seed'],r['budget'],r['method']):r for r in rows}
    assert len(index)==120
    source=[]
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7.5,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,155/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.095,top=.865,wspace=.30,hspace=.42)
    titles=['No queries (Q = 0)','After 400 queries','Full relative to controls (Q = 400)','Change from Q = 0 to Q = 400']
    for i,ax in enumerate(axes.ravel()):
        ax.set_title(titles[i],loc='left',pad=12)
        ax.text(-.16,1.10,'abcd'[i],transform=ax.transAxes,weight='bold',fontsize=9,va='bottom')
        ax.set_xlim(.5,5.5);ax.set_xticks(range(1,6),[f'S{i}' for i in range(1,6)])
        ax.set_xlabel('Published Reuters sample');ax.tick_params(length=2.5,width=.65,pad=2)
        ax.grid(axis='y',color='#E9E9E9',linewidth=.4,zorder=0)
    for axis,Q in zip(axes[0],[0,400]):
        axis.set_ylim(0,100);axis.set_ylabel('ACC (%)')
        for sample in range(1,6):
            for mi,method in enumerate(METHODS):
                rs=[index[sample,seed,Q,method] for seed in [401,402,403]]
                values=[100*r['ACC'] if r['ACC'] is not None else np.nan for r in rs]
                draw_group(axis,sample+(mi-1.5)*.17,values,[r['eligible'] for r in rs],COLORS[method],MARKERS[method])
                for r,v in zip(rs,values):source.append(dict(panel='a' if Q==0 else 'b',sample=sample,seed=r['seed'],series=method,value=v,units='ACC percent',eligible=r['eligible']))
    ax=axes[1,0];ax.set_ylabel('Full − control ACC (pp)');ax.axhline(0,color='#555555',lw=.7,zorder=1)
    for sample in range(1,6):
        for mi,method in enumerate(METHODS[1:]):
            values=[];eligible=[]
            for seed in [401,402,403]:
                a=index[sample,seed,400,'Full'];b=index[sample,seed,400,method]
                value=100*(a['ACC']-b['ACC']) if a['ACC'] is not None and b['ACC'] is not None else np.nan
                ok=a['eligible'] and b['eligible'];values.append(value);eligible.append(ok)
                source.append(dict(panel='c',sample=sample,seed=seed,series='Full minus '+method,value=value,units='percentage points',eligible=ok))
            draw_group(ax,sample+(mi-1)*.22,values,eligible,COLORS[method],MARKERS[method])
    ax=axes[1,1];ax.set_ylabel('Q400 − Q0 ACC (pp)');ax.axhline(0,color='#555555',lw=.7,zorder=1)
    for sample in range(1,6):
        for mi,method in enumerate(METHODS):
            values=[];eligible=[]
            for seed in [401,402,403]:
                a=index[sample,seed,400,method];b=index[sample,seed,0,method]
                value=100*(a['ACC']-b['ACC']) if a['ACC'] is not None and b['ACC'] is not None else np.nan
                ok=a['eligible'] and b['eligible'];values.append(value);eligible.append(ok)
                source.append(dict(panel='d',sample=sample,seed=seed,series=method,value=value,units='percentage points',eligible=ok))
            draw_group(ax,sample+(mi-1.5)*.17,values,eligible,COLORS[method],MARKERS[method])
    for ax in axes[1]:ax.margins(y=.17)
    handles=[Line2D([],[],marker=MARKERS[m],color=COLORS[m],linestyle='none',markersize=4,label=m) for m in METHODS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.985),ncol=4,columnspacing=2.4,handletextpad=.6)
    fig.text(.54,.940,'Small markers: three seeds; outlined markers and bars: mean ± SD',ha='center',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=out/'reuters_holdout.alignment.json',
        overlay_svg=out/'reuters_holdout.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(out/'reuters_holdout.pdf')
    fig.savefig(out/'reuters_holdout.svg')
    fig.savefig(out/'reuters_holdout.png',dpi=600)
    fig.savefig(out/'reuters_holdout.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    with (out/'reuters_holdout_source_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(source[0]));writer.writeheader();writer.writerows(source)
    assert len(source)==120+45+60
    plt.close(fig)
    print('EXPORTED',out,'source observations',len(source),flush=True)

if __name__=='__main__':main()
