"""Typography-only redraw of four existing SAPN supplementary figures.

All coordinates come from the public release. No fitting or selection occurs.
The inherited scientific marks, limits, and plot rectangles are retained;
rendered body text is at least 8 pt when the figure is placed at 166 mm width.
Usage: python plot_final_figure_fonts.py --release-root REPOSITORY --output OUT
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FormatStrFormatter

DATA=['3Sources','BBC','Reuters','Cora','CiteSeer']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def name(d):return 'Reuters-local' if d=='Reuters' else d
def base_style(font='Arial'):
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font,'DejaVu Sans'],
        'font.size':9,'axes.labelsize':9,'axes.titlesize':9.4,'xtick.labelsize':9,
        'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.linewidth':.6,'legend.frameon':False,
        'axes.unicode_minus':True,'savefig.facecolor':'white'})

def save(fig,stem,out,require_matplotlib_panel_alignment,expected_rows,palette):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=out/(stem+'.alignment.json'),
        overlay_svg=out/(stem+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    height=fig.get_figheight()*72
    rects=[]
    for ax in fig.axes:
        bb=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        rects.append([bb.x0*72,height-bb.y1*72,bb.x1*72,height-bb.y0*72])
    # Plot only the inherited coordinates. Preserve a machine-readable artist
    # trace to separate scientific geometry from text-layout changes.
    traces=[]
    for ax in fig.axes:
        traces.append(dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),yscale=ax.get_yscale(),
            lines=[np.asarray(line.get_xydata()).tolist() for line in ax.lines],
            scatter=[np.asarray(c.get_offsets()).tolist() for c in ax.collections if hasattr(c,'get_offsets')],
            error_segments=[np.asarray(c.get_segments()).tolist() for c in ax.collections if hasattr(c,'get_segments')]))
    fig.savefig(out/(stem+'.pdf'),dpi=600)
    fig.savefig(out/(stem+'.svg'),dpi=600)
    fig.savefig(out/(stem+'.png'),dpi=600)
    dump(out/(stem+'.geometry.json'),dict(width_pt=fig.get_figwidth()*72,height_pt=height,
        axes_bbox_pt=rects,scientific_artist_traces=traces,source_rows=expected_rows,palette=palette,
        body_font_at_166mm_pt=9*166/(fig.get_figwidth()*25.4)))
    plt.close(fig)

def numerics(repo,out,alignment):
    src=repo/'figures/source_data/fig05_source_data.json';records=read(src)['records']
    assert [r['dataset'] for r in records]==DATA and len(records)==5
    base_style('DejaVu Sans')
    colors=['#0072B2','#D55E00','#007A5E','#8A4D91','#555555']
    marks=['o','s','^','D','v'];patterns=['-','--','-.',':',(0,(5,1))]
    fig,axes=plt.subplots(1,2,figsize=(178/25.4,91/25.4))
    count=0
    for i,(r,c,m) in enumerate(zip(records,colors,marks)):
        assert r['seed']==23 and r['Q']==400
        x,y=r['iteration'],r['J_relative_to_stage2_initial'];count+=len(x)
        line,=axes[0].plot(x,y,color=c,ls=patterns[i],lw=1.1)
        axes[0].plot(x[-1],y[-1],color=c,marker=m,ms=4)
        hx=[q['iteration'] for q in r['maximum_block_residual']]
        hy=[q['residual'] for q in r['maximum_block_residual']];count+=len(hx)
        assert np.isfinite(y).all() and np.min(hy)>0
        line2,=axes[1].semilogy(hx,hy,color=c,marker=m,ls=patterns[i],ms=3,lw=1)
        assert np.array_equal(line.get_ydata(),y) and np.array_equal(line2.get_ydata(),hy)
    axes[0].set_title('(a) Second-stage objective',loc='left')
    axes[0].set_ylabel(r'$J_t / J_0$ (stage 2)');axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[1].set_title('(b) First-order stopping residual',loc='left')
    axes[1].axhline(1e-3,color='#222222',lw=.9,ls='--')
    axes[1].text(.80,8.50e-4,r'tolerance $10^{-3}$',transform=axes[1].get_yaxis_transform(),ha='right',va='top',fontsize=9)
    axes[1].set_ylabel('Maximum normalized block residual')
    for ax in axes:
        ax.set_xlabel('Iteration within stage 2');ax.set_xlim(left=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#DDDDDD',lw=.45)
        ax.set_axisbelow(True);ax.tick_params(length=2.5,width=.6)
    handles=[Line2D([],[],color=c,marker=m,ls=ls,ms=3.5,label=name(ds)) for ds,c,m,ls in zip(DATA,colors,marks,patterns)]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,.005),fontsize=9)
    fig.subplots_adjust(left=.12,right=.985,bottom=.25,top=.91,wspace=.36)
    save(fig,'fig05_second_stage_numerics',out,alignment,count,colors)
    return [src]

def mechanisms(repo,out,alignment):
    src=repo/'evidence/v5/mechanisms/per_run.csv';rows=csvread(src)
    index={(r['dataset'],int(r['seed']),int(r['budget']),r['method']):r for r in rows}
    assert len(rows)==len(index)==120
    controls=['private-coupled-CV','no-correction-CV'];records=[]
    for ds in DATA:
        for q in [160,400]:
            for control in controls:
                for seed in [301,302,303]:
                    a,b=index[ds,seed,q,'full-CV'],index[ds,seed,q,control]
                    assert a['converged'].lower()==b['converged'].lower()=='true'
                    records.append(dict(dataset=ds,budget=q,control=control,seed=seed,delta=100*(float(a['ACC'])-float(b['ACC']))))
    assert len(records)==60
    base_style();fig,axes=plt.subplots(2,2,figsize=(183/25.4,110/25.4),sharey=True)
    fig.subplots_adjust(left=.17,right=.985,bottom=.13,top=.89,wspace=.24,hspace=.60)
    colors=['#346A8A','#407B6B'];spans={}
    for control in controls:
        vals=[r['delta'] for r in records if r['control']==control]
        extents=[]
        for q in [160,400]:
            for ds in DATA:
                v=[r['delta'] for r in records if (r['dataset'],r['budget'],r['control'])==(ds,q,control)]
                extents.append(abs(np.mean(v))+np.std(v,ddof=1))
        spans[control]=max(1.,max(abs(v) for v in vals),max(extents))*1.18
    for i,q in enumerate([160,400]):
        for j,control in enumerate(controls):
            ax=axes[i,j];ax.axvline(0,color='#A5A5A5',lw=.7,zorder=0)
            for y,ds in enumerate(DATA):
                vals=[r['delta'] for r in records if (r['dataset'],r['budget'],r['control'])==(ds,q,control)]
                assert len(vals)==3
                for z,val in enumerate(vals):ax.scatter(val,y+(z-1)*.12,s=11,marker='o',color='#9CA4AA',linewidths=.6,alpha=.85,zorder=2)
                ax.errorbar(np.mean(vals),y,xerr=np.std(vals,ddof=1),fmt='D',markersize=3.5,color=colors[j],capsize=2,elinewidth=.8,zorder=3)
            ax.set_xlim(-spans[control],spans[control]);ax.set_ylim(4.55,-.55)
            ax.set_yticks(range(5),[name(ds) for ds in DATA]);ax.tick_params(axis='y',length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.grid(axis='y',color='#ECEFF1',lw=.5,zorder=-1)
            ax.spines[['top','right','left']].set_visible(False)
            ax.text(-.08,1.08,'abcd'[i*2+j],transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom')
            ax.set_title(['Private coupling\nindependent selection','No correction smoothing\nindependent selection'][j],pad=10 if i==0 else 6)
            ax.set_xlabel('Full minus control ACC (pp)')
            if j==0:ax.text(-.36,.5,f'Q = {q}',transform=ax.transAxes,fontsize=9,rotation=90,rotation_mode='anchor',ha='center',va='center')
    save(fig,'fig18_mechanism_controls',out,alignment,len(records),colors+['#9CA4AA'])
    dump(out/'fig18_plotted_values.json',records)
    return [src]

def holdout(repo,out,alignment):
    src=repo/'evidence/v6/multilingual_holdout/figures/reuters_holdout_source_data.csv';rows=csvread(src)
    assert len(rows)==225 and all(r['eligible'].lower()=='true' for r in rows)
    methods=['Full','SAPN-EQ','CMVNMF','NoCorr'];colors=['#3268A8','#9D6E35','#698D7E','#8B7DA7'];marks=['o','s','^','D']
    base_style();fig,axes=plt.subplots(2,2,figsize=(183/25.4,155/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.095,top=.865,wspace=.30,hspace=.42)
    titles=['No queries (Q = 0)','After 400 queries','Full relative to controls (Q = 400)','Change from Q = 0 to Q = 400']
    for i,ax in enumerate(axes.ravel()):
        ax.set_title(titles[i],loc='left',pad=12);ax.text(-.16,1.10,'abcd'[i],transform=ax.transAxes,weight='bold',fontsize=9,va='bottom')
        ax.set_xlim(.5,5.5);ax.set_xticks(range(1,6),[f'S{i}' for i in range(1,6)])
        ax.set_xlabel('Published Reuters sample');ax.tick_params(length=2.5,width=.65,pad=2)
        ax.grid(axis='y',color='#E9E9E9',lw=.4,zorder=0);ax.spines[['top','right']].set_visible(False)
    count=0
    for panel,ax in zip('abcd',axes.ravel()):
        if panel in 'ab':ax.set_ylim(0,100);ax.set_ylabel('ACC (%)')
        else:ax.set_ylabel('Full − control ACC (pp)' if panel=='c' else 'Q400 − Q0 ACC (pp)');ax.axhline(0,color='#555555',lw=.7,zorder=1)
        use=methods[1:] if panel=='c' else methods
        for sample in range(1,6):
            for mi,method in enumerate(use):
                series='Full minus '+method if panel=='c' else method
                rr=sorted([r for r in rows if r['panel']==panel and int(r['sample'])==sample and r['series']==series],key=lambda r:int(r['seed']))
                assert [int(r['seed']) for r in rr]==[401,402,403]
                vals=[float(r['value']) for r in rr];count+=len(vals)
                idx=methods.index(method);color,marker=colors[idx],marks[idx]
                x=sample+(mi-1)*.22 if panel=='c' else sample+(mi-1.5)*.17
                for j,val in enumerate(vals):ax.scatter(x+[-.035,0,.035][j],val,s=9,marker=marker,edgecolor=color,facecolor=color,lw=.45,alpha=.55,zorder=3)
                ax.errorbar([x],[np.mean(vals)],yerr=[np.std(vals,ddof=1)],fmt=marker,color=color,markersize=3.7,markerfacecolor='white',markeredgewidth=.9,elinewidth=.85,capsize=2,capthick=.75,zorder=4)
        if panel in 'cd':ax.margins(y=.17)
    handles=[Line2D([],[],marker=m,color=c,linestyle='none',markersize=4,label=method) for method,c,m in zip(methods,colors,marks)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.997),ncol=4,columnspacing=2.4,handletextpad=.6)
    fig.text(.54,.928,'Small markers: three seeds; outlined markers and bars: mean ± SD',ha='center',fontsize=9)
    save(fig,'fig19_reuters_holdout',out,alignment,count,colors)
    return [src]

def mass(repo,out,alignment):
    src=repo/'evidence/v6/selection_mass_diagnostic/figure_delta_source.csv';rows=csvread(src)
    assert len(rows)==60 and all(r['paired_eligible'].lower()=='true' for r in rows)
    conditions=[(d,q) for d in DATA for q in [160,400]]
    positions=np.array([i+(i//2)*.4 for i in range(10)],float);colors={160:'#377E9A',400:'#328467'}
    extents=[abs(float(r['new_minus_old_ACC_pp'])) for r in rows]
    for method in ['full','no-correction']:
        for ds,q in conditions:
            vals=[float(r['new_minus_old_ACC_pp']) for r in rows if (r['dataset'],int(r['budget']),r['method'])==(ds,q,method)]
            assert len(vals)==3;extents.append(abs(np.mean(vals))+np.std(vals,ddof=1))
    lim=max([1.]+extents)*1.12
    base_style();fig,axes=plt.subplots(1,2,figsize=(183/25.4,145/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.19,right=.975,bottom=.13,top=.87,wspace=.19)
    for ci,(method,title) in enumerate([('full','Full'),('no-correction','No correction smoothing')]):
        ax=axes[ci];ax.axvline(0,color='#777777',lw=.65,zorder=1)
        for i,(ds,q) in enumerate(conditions):
            rr=sorted([r for r in rows if (r['dataset'],int(r['budget']),r['method'])==(ds,q,method)],key=lambda r:int(r['seed']))
            vals=[float(r['new_minus_old_ACC_pp']) for r in rr]
            for off,val in zip([-.24,-.08,.08],vals):ax.scatter([val],[positions[i]+off],s=11,marker='o',facecolor='#92989D',edgecolor='#92989D',lw=.6,alpha=.75,zorder=2)
            ax.errorbar(np.mean(vals),positions[i]+.26,xerr=np.std(vals,ddof=1),fmt='D',markersize=3.4,color=colors[q],elinewidth=.8,capsize=2,capthick=.7,zorder=3)
            ax.axhline(positions[i],color='#ECEFF1',lw=.45,zorder=0)
        ax.set_xlim(-lim,lim);ax.set_ylim(positions[-1]+.55,-.55);ax.set_yticks(positions)
        ax.set_yticklabels([f'{name(ds)} | {q}' for ds,q in conditions]);ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Scaled CV − original CV ACC (pp)');ax.set_title(title,pad=10)
        ax.text(-.065,1.025,'ab'[ci],transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#92989D',markeredgecolor='#92989D',markersize=3,label='Paired seed'),
        Line2D([0],[0],marker='D',color=colors[160],markersize=3,label='Q = 160: mean ± sample SD'),
        Line2D([0],[0],marker='D',color=colors[400],markersize=3,label='Q = 400: mean ± sample SD')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.97),ncol=3,fontsize=9,handlelength=1.4,columnspacing=1.2)
    save(fig,'fig20_mass_transfer',out,alignment,len(rows),list(colors.values())+['#92989D'])
    return [src]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--release-root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();repo=args.release_root.resolve();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    # Study-authored, standalone helper already included in the public release.
    sys.path.insert(0,str(repo/'figures'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    sources=[]
    for fn in [numerics,mechanisms,holdout,mass]:sources+=fn(repo,out,require_matplotlib_panel_alignment)
    dump(out/'FONT_POLISH_PROVENANCE.json',dict(status='RENDERED_QA_PENDING',
        script_sha256=sha(__file__),purpose='Typography-only revision of four unchanged supplementary quantitative figures',
        backend='Python Matplotlib',matplotlib=matplotlib.__version__,numpy=np.__version__,
        data_sources=[{'path':str(p.relative_to(repo)).replace('\\','/'),'sha256':sha(p)} for p in sources],
        no_fitting_or_model_selection=True,exclusions=0,uncertainty='Original sample SD; original per-run trajectories where applicable',
        original_axes_dimensions_and_data_ranges_retained=True,body_text_at_166mm='At least 8 pt; mathematical scripts can be smaller',
        outputs=[{'path':p.name,'sha256':sha(p)} for p in sorted(out.glob('*.pdf'))]))
    print('Rendered four figures with unchanged scientific coordinates; visual and geometry QA required.')

if __name__=='__main__':main()
