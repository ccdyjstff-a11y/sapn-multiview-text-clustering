"""Author-owned function snapshot; consumed by the portable replay entry.
Original function ASTs and their immutable public origins are bound in SOURCE_PROVENANCE.json.
This file is not a standalone entry point.
"""

def setup():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':8, 'axes.labelsize':8, 'axes.titlesize':9, 'xtick.labelsize':8,
        'ytick.labelsize':8, 'legend.fontsize':8, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6,
        'lines.linewidth':1.1, 'figure.facecolor':'white', 'axes.facecolor':'white'})

def panel_label(ax, letter):
    ax.annotate(letter, xy=(0,1), xycoords='axes fraction', xytext=(-19,5),
        textcoords='offset points', ha='left', va='bottom', fontsize=9, fontweight='bold')

def statistics_legend(fig, y=.994):
    handles = [Line2D([],[], marker='o', color='#6A727C', linestyle='none', markersize=3.2, label='Paired seed'),
        Line2D([],[], marker='D', color='#171A1E', linestyle='-', markersize=4, label='Mean ± sample SD'),
        Line2D([],[], marker='x', color='#777777', linestyle='none', markersize=4, label='Ineligible terminal')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56,y), ncol=3,
        frameon=False, columnspacing=1.5, handlelength=1.5)

def record_lookup(group):
    records = group['records']
    lookup = {(r['dataset'],r['seed'],r['Q'],r['method']):r for r in records}
    assert len(lookup) == len(records)
    return lookup

def effect_extent(cases, padding_floor=1.):
    all_values = [0.]
    for case in cases:
        all_values.extend(p['delta_pp'] for p in case['points'] if p['delta_pp'] is not None)
        if case['mean_pp'] is not None:
            all_values.extend([case['mean_pp']-case['sd_pp'], case['mean_pp']+case['sd_pp']])
    lo, hi = min(all_values), max(all_values)
    padding = max(padding_floor, .09*(hi-lo))
    return lo-padding, hi+padding

def effect_marks(ax, case, index, color, horizontal=True):
    offsets = np.linspace(-.19,.19,case['planned'])
    for p, offset in zip(case['points'], offsets):
        if p['delta_pp'] is None:
            continue
        x, y = (p['delta_pp'], index+offset) if horizontal else (index+offset, p['delta_pp'])
        ax.scatter([x],[y], s=10 if p['eligible'] else 15,
            marker='o' if p['eligible'] else 'x', color=color if p['eligible'] else '#777777',
            alpha=.72, linewidths=.6, zorder=3)
    if case['mean_pp'] is not None:
        x, y = (case['mean_pp'],index) if horizontal else (index,case['mean_pp'])
        args = {'xerr':case['sd_pp']} if horizontal else {'yerr':case['sd_pp']}
        ax.errorbar(x,y,**args,fmt='D',color='#171A1E',markersize=3.7,
            markerfacecolor='white',markeredgewidth=.8,elinewidth=.8,capsize=2,zorder=4)

def effects_source(cases):
    points = [p for case in cases for p in case['points']]
    summaries = [{key:case[key] for key in ['dataset','Q','comparison','metric','planned','eligible','mean_pp','sd_pp']} for case in cases]
    for case, summary in zip(cases, summaries):
        summary['direction'] = case['points'][0]['direction']
    return points, summaries

def draw15(evidence):
    group=evidence['new_repetition'];lookup=record_lookup(group);points=[];summaries=[]
    fig,axes=plt.subplots(3,2,figsize=(WIDTH_MM/25.4,180/25.4))
    fig.subplots_adjust(left=.095,right=.985,bottom=.075,top=.92,wspace=.31,hspace=.65)
    markers=['o','s','^'];handles=[Line2D([],[],color=COLORS[m],marker=s,markersize=4,label=m) for m,s in zip(METHODS,markers)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1),ncol=3,frameon=False)
    for i,ds in enumerate(DEVELOPMENT):
        ax=axes.flat[i]
        for mi,(method,marker) in enumerate(zip(METHODS,markers)):
            means=[];sds=[]
            for Q in [0,160,400]:
                cell=next(c for c in group['cells'] if (c['dataset'],c['Q'],c['method'])==(ds,Q,method))
                assert cell['planned']==7 and cell['seeds']==list(range(201,208))
                metric=cell['metrics']['ACC'];means.append(metric['primary_mean_percent']);sds.append(metric['primary_sd_percent'])
                summaries.append(dict(dataset=ds,Q=Q,method=method,metric='ACC',planned=cell['planned'],eligible=cell['eligible'],
                    mean_percent=metric['primary_mean_percent'],sd_percent=metric['primary_sd_percent']))
                for seed,offset in zip(cell['seeds'],np.linspace(-6,6,cell['planned'])):
                    record=lookup[(ds,seed,Q,method)];value=record['metrics_raw']['ACC'];value=None if value is None else 100*value
                    point=dict(dataset=ds,seed=seed,Q=Q,method=method,metric='ACC',value_percent=value,
                        eligible=record['eligible'],status=record['status'],plotted=value is not None,
                        prediction_sha256=record['prediction_sha256'])
                    points.append(point)
                    if value is not None:
                        ax.scatter(Q+(mi-1)*13+offset,value,s=8,marker='o' if record['eligible'] else 'x',
                            color=COLORS[method] if record['eligible'] else '#777777',alpha=.5,linewidths=.5,zorder=2)
                if metric['primary_mean_percent'] is None:
                    # Missing complete aggregates remain explicit and are never assigned zero.
                    ax.annotate(f"{Q}: {cell['eligible']}/{cell['planned']}",xy=(Q,1),xycoords=('data','axes fraction'),
                        xytext=(0,-4-10*mi),textcoords='offset points',ha='center',va='top',fontsize=8,color=COLORS[method])
            yy=np.array([np.nan if v is None else v for v in means])
            sd=np.array([np.nan if v is None else v for v in sds])
            ax.errorbar([0,160,400],yy,yerr=sd,color=COLORS[method],marker=marker,markersize=3.5,
                linewidth=1.1,elinewidth=.7,capsize=2,zorder=4)
        ax.set_xticks([0,160,400]);ax.set_xlim(-30,430);ax.set_xlabel('Simulated budget Q')
        ax.set_ylabel('ACC (%)');ax.set_title(NAMES[ds],loc='left',pad=8)
        ax.grid(axis='y',color='#E0E3E8',linewidth=.35);ax.yaxis.set_major_locator(MaxNLocator(4))
        panel_label(ax,chr(97+i))
    axes.flat[5].set_axis_off()
    # An explicit non-data sixth cell explains the common uncertainty and unit.
    axes.flat[5].text(.04,.78,'Seven new seeds per corpus\n\nSmall points: individual runs\nMarkers: mean ± sample SD',
        transform=axes.flat[5].transAxes,fontsize=8,va='top',linespacing=1.5)
    return export_and_audit(fig,'fig15_supervision_repetition',points,summaries,evidence)

def draw16(evidence):
    group=evidence['new_repetition'];lookup=record_lookup(group);cases=[];mapped={}
    for ds in DEVELOPMENT:
        for metric in METRICS:
            mapped[(ds,metric)]=[paired_case(group,lookup,ds,400,method,metric,direction=-1) for method in REMOVALS]
            cases.extend(mapped[(ds,metric)])
    fig,axes=plt.subplots(1,3,figsize=(WIDTH_MM/25.4,218/25.4))
    fig.subplots_adjust(left=.20,right=.985,bottom=.065,top=.925,wspace=.18)
    statistics_legend(fig)
    short_labels=['No cons. pairs','No view pairs','No pairs','No ML','No CL']
    for col,metric in enumerate(METRICS):
        ax=axes[col];ticks=[];labels=[]
        for row,ds in enumerate(DEVELOPMENT):
            for j,case in enumerate(mapped[(ds,metric)]):
                index=row*6+j;effect_marks(ax,case,index,'#0072B2')
                ticks.append(index)
                labels.append(f"{short_labels[j]} [{case['eligible']}/{case['planned']}]")
            if col==0:
                ax.text(-.02,row*6-.9,NAMES[ds],transform=ax.get_yaxis_transform(),
                    ha='right',va='center',fontsize=8,fontweight='bold',clip_on=False)
            if row<4:ax.axhline(row*6+4.6,color='#D9DDE3',linewidth=.4,zorder=0)
        ax.axvline(0,color='#8C949D',linewidth=.6,zorder=0)
        ax.set_xlim(*effect_extent([c for c in cases if c['metric']==metric]));ax.set_ylim(28.7,-1.6)
        ax.set_yticks(ticks,labels=labels if col==0 else ['']*25)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='x',color='#E0E3E8',linewidth=.35)
        ax.set_title(metric,pad=8);ax.set_xlabel('Removal − full (pp)')
        panel_label(ax,chr(97+col))
    return export_and_audit(fig,'fig16_signed_ablation_repetition',*effects_source(cases),evidence)

def draw17(evidence):
    group=evidence['new_repetition'];lookup=record_lookup(group);cases=[];mapped={}
    for method in CONTROLS:
        for metric in METRICS:
            mapped[(method,metric)]=[paired_case(group,lookup,ds,Q,method,metric) for ds in DEVELOPMENT for Q in [160,400]]
            cases.extend(mapped[(method,metric)])
    fig,axes=plt.subplots(3,3,figsize=(WIDTH_MM/25.4,220/25.4))
    fig.subplots_adjust(left=.215,right=.985,bottom=.065,top=.92,wspace=.18,hspace=.26)
    statistics_legend(fig)
    for row,(method,label) in enumerate(zip(CONTROLS,CONTROL_LABELS)):
        for col,metric in enumerate(METRICS):
            ax=axes[row,col];these=mapped[(method,metric)]
            for j,case in enumerate(these):effect_marks(ax,case,j,COLORS[method])
            ax.axvline(0,color='#8C949D',linewidth=.6,zorder=0)
            # These controls answer different questions and can have different effect scales.
            # Preserve every seed/SD extent and expose an independently labelled axis per panel.
            ax.set_xlim(*effect_extent(these,padding_floor=.05));ax.set_ylim(9.6,-.6)
            labels=[f"{NAMES[c['dataset']]} {c['Q']} [{c['eligible']}/{c['planned']}]" for c in these]
            ax.set_yticks(range(10),labels=labels if col==0 else ['']*10)
            ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='x',color='#E0E3E8',linewidth=.35)
            if row==0:ax.set_title(metric,pad=8)
            if row==2:ax.set_xlabel('Difference (pp)')
            panel_label(ax,chr(97+row*3+col))
        box=axes[row,0].get_position()
        fig.text(.012,(box.y0+box.y1)/2,label,rotation=90,rotation_mode='anchor',
            ha='center',va='center',fontsize=8)
    return export_and_audit(fig,'fig17_reference_repetition',*effects_source(cases),evidence)
