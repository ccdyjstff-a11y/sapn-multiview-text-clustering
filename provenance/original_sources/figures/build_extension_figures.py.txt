"""Figures14--17 from complete, independently audited SAPN evidence only.

Scientific questions, not predetermined positive conclusions:
14: How do complete pipelines differ under the same queries on seven corpora?
15: Are query-budget effects stable across seven new random runs?
16: Which signed losses affect the frozen full-model configuration?
17: When do reference preservation, coverage selection and correction matter?

All figures are quantitative grids. Python/Matplotlib is the established
backend; 180mm width, 8pt text, editable SVG/PDF and600dpiPNG/TIFF exports.
All planned seeds remain in source data. Failed pairs are not replaced by zero
or counted as wins; only complete eligible cells receive a mean and sampleSD.
Subsequent panel-wise visual review remains mandatory after automated checks.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import load_verified_evidence as evidence_loader
from build_extension_results import plot_source_data

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT/'figures'
QA = ROOT/'figure_qa'
TOOLS = Path('C:/Users/LENOVO/.codex/skills/nature-figure/scripts')
QA_DEPENDENCIES = ROOT/'qa_dependencies'
sys.path.insert(0, str(TOOLS))
from audit_panel_alignment import require_matplotlib_panel_alignment

METRICS = ['ACC', 'NMI', 'ARI']
DATASETS = evidence_loader.DATASETS
DEVELOPMENT = evidence_loader.DEVELOPMENT
NAMES = {'3Sources':'3Sources', 'BBC':'BBC', 'Reuters':'Reuters-local', 'Cora':'Cora',
    'CiteSeer':'CiteSeer', DATASETS[5]:'20NG', DATASETS[6]:'AG News'}
METHODS = ['SAPN-MV', 'SAPN-EQ', 'CMVNMF']
COMPARATORS = ['SAPN-EQ', 'CMVNMF', 'MVCHSS', 'FSSMSC']
COLORS = {'SAPN-MV':'#0072B2', 'SAPN-EQ':'#D55E00', 'CMVNMF':'#009E73',
    'MVCHSS':'#8762A3', 'FSSMSC':'#6A727C', 'graph-only-CV':'#0072B2',
    'fixed-CV':'#D55E00', 'without_correction_graph':'#009E73'}
REMOVALS = ['without_consensus_pair', 'without_private_pair', 'without_all_pair', 'without_ML', 'without_CL']
REMOVAL_LABELS = ['No cons.\npairs', 'No view\npairs', 'No\npairs', 'No\nML', 'No\nCL']
CONTROLS = ['graph-only-CV', 'fixed-CV', 'without_correction_graph']
CONTROL_LABELS = ['Full − graph only', 'Full − fixed-CV', 'Full − no correction graph']
WIDTH_MM = 180
DPI = 600

CONTRACTS = {
 'fig14_unified_comparison':dict(height_mm=210,
    question='How do paired pipeline differences vary across corpora and metrics at the same simulated400-query budget?',
    panels='Four comparator rows and three metric columns; each panel retains seven corpora and all three matched seeds.',
    group='main', unit='Random query/initialization run within a fixed corpus, three paired seeds per case.',
    direction='SAPN-MV minus comparator, percentage points',
    roles='ACC measures matching accuracy;NMI andARI check whether conclusions agree across non-identical clustering criteria. Corpus rows expose reversals.',
    caption='Unified comparison at simulated query budget Q =400. Points show three matched-seed differences; diamonds and bars show the mean ± sample SD for complete eligible cells. Rows compare full pipelines; positive values favor SAPN-MV. Brackets give eligible/planned counts. Crosses denote available ineligible terminal outcomes, and missing predictions have no point. Seeds repeat each fixed corpus and are not independent corpus samples. The disclosed baseline adaptations and native completion rules apply.'),
 'fig15_supervision_repetition':dict(height_mm=180,
    question='Do query-budget changes persist across the seven newly declared random runs on known development corpora?',
    panels='Five corpus panels; three method series at Q0/160/400; all seven new seeds retained.',
    group='new_repetition', unit='Seven new random runs201--207 on each previously used development corpus.',
    direction='ACC percent; within-method budget trajectories',
    roles='Separate corpus panels test randomization stability without pooling corpus identity or treating old development seeds as confirmation.',
    caption='Simulated query-budget effects for seven new seeds on five development corpora. Small points show individual runs, horizontally offset for visibility; method markers and bars give means ± sample SD for complete eligible cells. Missing complete means interrupt lines; crosses denote available ineligible terminal outcomes. The repetitions remain separate from the original baseline comparison. Budget changes affect the entire procedure, including pair-loss mass, endpoint domain and coverage.'),
 'fig16_signed_ablation_repetition':dict(height_mm=218,
    question='Which signed-supervision losses influence the frozen full-model configuration across new random runs?',
    panels='Three metric columns, each containing25 paired-effect rows grouped by five corpora and five isolated term removals.',
    group='new_repetition', unit='Seven paired new seeds per removal and development corpus at Q400.',
    direction='Removal minus SAPN-MV, percentage points',
    roles='Removing both paths, each path and each sign tests distinct component roles;NMI/ARI expose criterion disagreement.',
    caption='Isolated signed-loss removals at Q =400. Points show seven paired new seeds; diamonds and bars give the mean ± sample SD for complete eligible pairs. Negative removal-minus-full differences indicate deterioration. Cons./view denote consensus/view-specific losses;ML/CL denote must-link/cannot-link losses. Counts give eligible/planned pairs. Queries, endpoint domain, graph, reference, initialization and all remaining coefficients stay fixed at the full-model configuration. Reduced models are not retuned; removing pair losses is not Q =0.'),
 'fig17_reference_repetition':dict(height_mm=220,
    question='How do three distinct reference controls affect the full model across budgets and new random runs?',
    panels='Three control rows and three metric columns; ten dataset/budget cases per panel.',
    group='new_repetition', unit='Seven paired new seeds for each of five development corpora at Q160/400.',
    direction='SAPN-MV minus control, percentage points',
    roles='Graph-only isolates reference-containing selected configurations;fixed-CV challenges the coverage rule;no-correction deletes the precise graph correction term.',
    caption='Reference controls across seven new seeds: full SAPN-MV minus control. Points retain paired runs; diamonds and bars give means ± sample SD for complete eligible cells. Counts give eligible/planned pairs. Graph-only and fixed-CV select coefficients from the same within-budget query answers; fixed-CV also searches nine reference strengths. Correction removal keeps the full-model coefficient fixed. The first two rows compare selected configurations; the third is an isolated intervention. Horizontal scales vary by panel; seeds are not independent corpus replicates.')
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    assert rows
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


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


def paired_case(group, lookup, ds, Q, comparison, metric, direction=1):
    candidates = [p for p in group['paired'] if (p['dataset'],p['Q'],p['comparison']) == (ds,Q,comparison)]
    assert len(candidates) == 1
    pair = candidates[0]; values = pair['metrics'][metric]
    assert len(pair['seeds']) == len(values['deltas_pp']) == pair['planned']
    assert len(set(pair['seeds'])) == pair['planned']
    points = []
    for seed, delta in zip(pair['seeds'], values['deltas_pp']):
        left = lookup[(ds,seed,Q,'SAPN-MV')]; right = lookup[(ds,seed,Q,comparison)]
        a = left['metrics_raw'][metric]; b = right['metrics_raw'][metric]
        valid = bool(left['eligible'] and right['eligible'])
        expected = None if a is None or b is None else 100*(a-b)
        assert (delta is None) == (expected is None)
        if expected is not None:
            assert np.isclose(delta, expected, rtol=1e-12, atol=1e-12)
        points.append(dict(dataset=ds,seed=seed,Q=Q,comparison=comparison,metric=metric,
            direction='Full minus comparison' if direction==1 else 'Comparison minus full',
            full_metric_fraction=a, comparison_metric_fraction=b,
            delta_pp=None if delta is None else direction*delta,
            eligible=valid, full_status=left['status'], comparison_status=right['status'],
            plotted=delta is not None, missing_reason='No numeric prediction for one or both records' if delta is None else '',
            full_prediction_sha256=left['prediction_sha256'], comparison_prediction_sha256=right['prediction_sha256']))
    mean = None if values['mean_pp'] is None else direction*values['mean_pp']
    sd = values['sd_pp']
    if mean is not None:
        assert all(p['eligible'] and p['delta_pp'] is not None for p in points)
        assert np.isclose(mean, np.mean([p['delta_pp'] for p in points]), atol=1e-12)
        assert np.isclose(sd, np.std([p['delta_pp'] for p in points], ddof=1), atol=1e-12)
    return dict(dataset=ds,Q=Q,comparison=comparison,metric=metric,points=points,
        planned=len(points), eligible=sum(p['eligible'] for p in points), mean_pp=mean, sd_pp=sd)


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


def export_and_audit(fig, stem, points, summaries, evidence):
    contract = dict(CONTRACTS[stem], figure=stem, archetype='quantitative grid',
        width_mm=WIDTH_MM, base_font_pt=8, minimum_glyph_pt=5, dpi=DPI,
        uncertainty='Mean and one sample SD; only complete eligible cells. All available seed-level values retained.',
        inference='Descriptive matched random-run effects. No new significance test or independent-corpus replication claim.',
        adaptation='Style-only reuse of the existing analytic figure typography/palette; quantitative evidence logic implemented anew.',
        source_code_sha256=sha(Path(__file__)), loader_source_sha256=sha(Path(evidence_loader.__file__)),
        source_records=len(points), source_aggregates=len(summaries), missing_replaced_by_zero=False,
        missing_numeric_points=sum(not row.get('plotted',True) for row in points),
        empirical_data_status=evidence['status'])
    if stem=='fig14_unified_comparison':
        contract['caption'] += (' †: MVCHSS author-map domain extension. '
            '‡: FSSMSC author-map pair-input clustering adapter at its native cap. '
            'Eligibility is method-specific, not a shared stationarity certificate; '
            'FSSMSC paper orthogonality is not certified.')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=QA/(stem+'.alignment.json'),
        overlay_svg=QA/(stem+'.alignment.svg'), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        strict=True, require_panel_labels=True,
        exclude_axes=[ax for ax in fig.axes if not ax.axison])
    # No tight bounding box: preserve the declared180mm physical width exactly.
    for suffix in ['svg','pdf']:
        fig.savefig(FIGURES/(stem+'.'+suffix))
    fig.savefig(FIGURES/(stem+'.png'), dpi=600)
    fig.savefig(FIGURES/(stem+'.tiff'), dpi=600, pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    raster_metadata={}
    expected_size=(int(WIDTH_MM/25.4*DPI),int(contract['height_mm']/25.4*DPI))
    for suffix in ['png','tiff']:
        with Image.open(FIGURES/(stem+'.'+suffix)) as raster:
            reported_dpi=tuple(float(v) for v in raster.info['dpi'])
            assert all(abs(a-b)<=1 for a,b in zip(raster.size,expected_size)), 'Raster physical size mismatch'
            assert all(abs(v-DPI)<.1 for v in reported_dpi), 'Raster resolution metadata mismatch'
            raster_metadata[suffix]=dict(size_px=list(raster.size),dpi=list(reported_dpi),mode=raster.mode)
    contract['raster_metadata']=raster_metadata
    write_csv(FIGURES/(stem+'_source.csv'),points)
    write_csv(FIGURES/(stem+'_summary.csv'),summaries)
    contract['source_csv_sha256'] = sha(FIGURES/(stem+'_source.csv'))
    contract['summary_csv_sha256'] = sha(FIGURES/(stem+'_summary.csv'))
    commands = [
        ('text', [str(TOOLS/'audit_pdf_text.py'), str(FIGURES/(stem+'.pdf')), '--min-pt','5','--json']),
        ('collision', [str(TOOLS/'audit_figure_collisions.py'), str(FIGURES/(stem+'.pdf')),
            '--json-out',str(QA/(stem+'_collisions.json')), '--overlay-pdf',str(QA/(stem+'_collision_overlay.pdf'))])]
    reports = {}
    for name, arguments in commands:
        # PyMuPDF is isolated in the manuscript QA dependency directory.
        bootstrap = ('import runpy,sys;sys.path.insert(0,sys.argv.pop(1));'
                     'p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name="__main__")')
        run = subprocess.run([sys.executable,'-X','utf8','-c',bootstrap,str(QA_DEPENDENCIES),*arguments],
            capture_output=True,text=True,encoding='utf-8')
        (QA/(stem+'_'+name+'_stdout.txt')).write_text(run.stdout,encoding='utf-8')
        (QA/(stem+'_'+name+'_stderr.txt')).write_text(run.stderr,encoding='utf-8')
        if name == 'text':
            reports[name] = json.loads(run.stdout)
            write_json(QA/(stem+'_text_audit.json'), reports[name])
        else:
            reports[name] = json.loads((QA/(stem+'_collisions.json')).read_text(encoding='utf-8'))
        assert run.returncode == 0, f'{stem}: {name} audit blocked export approval; inspect saved report'
    contract['automated_audits'] = reports
    contract['visual_review_status'] = 'PENDING_PANEL_BY_PANEL_REVIEW'
    contract['outputs_sha256'] = {suffix:sha(FIGURES/(stem+'.'+suffix)) for suffix in ['svg','pdf','png','tiff']}
    write_json(QA/(stem+'_contract.json'),contract)
    return contract


def draw14(evidence):
    group=evidence['main'];lookup=record_lookup(group);cases=[]
    mapped={}
    for comparator in COMPARATORS:
        for metric in METRICS:
            mapped[(comparator,metric)]=[paired_case(group,lookup,ds,400,comparator,metric) for ds in DATASETS]
            cases.extend(mapped[(comparator,metric)])
    fig,axes=plt.subplots(4,3,figsize=(WIDTH_MM/25.4,210/25.4))
    fig.subplots_adjust(left=.18,right=.985,bottom=.065,top=.91,wspace=.18,hspace=.35)
    statistics_legend(fig)
    for row,comparator in enumerate(COMPARATORS):
        for col,metric in enumerate(METRICS):
            ax=axes[row,col];these=mapped[(comparator,metric)]
            limits=effect_extent([case for case in cases if case['metric']==metric])
            for j,case in enumerate(these):effect_marks(ax,case,j,COLORS[comparator])
            ax.axvline(0,color='#8C949D',linewidth=.6,zorder=0)
            ax.set_xlim(*limits);ax.set_ylim(6.6,-.6);ax.xaxis.set_major_locator(MaxNLocator(4))
            labels=[f"{NAMES[c['dataset']]} [{c['eligible']}/{c['planned']}]" for c in these]
            ax.set_yticks(range(7),labels=labels if col==0 else ['']*7)
            ax.grid(axis='x',color='#E0E3E8',linewidth=.35)
            if row==0:ax.set_title(metric,pad=8)
            if row==3:ax.set_xlabel('Difference (pp)')
            panel_label(ax,chr(97+row*3+col))
        box=axes[row,0].get_position()
        display_comparator={'MVCHSS':'MVCHSS†','FSSMSC':'FSSMSC‡'}.get(comparator,comparator)
        fig.text(.015,(box.y0+box.y1)/2,'Full − '+display_comparator,rotation=90,
            rotation_mode='anchor',ha='center',va='center',fontsize=8)
    return export_and_audit(fig,'fig14_unified_comparison',*effects_source(cases),evidence)


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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','render'],nargs='?',default='prepare')
    args=parser.parse_args();FIGURES.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
    setup()
    write_json(QA/'extension_figure_contracts.json',dict(backend='Python/Matplotlib',width_mm=WIDTH_MM,
        dpi=DPI,base_font_pt=8,minimum_glyph_pt=5,contracts=CONTRACTS,
        empirical_status='Requires one successful load_all_verified before any quantitative rendering',
        source_code_sha256=sha(Path(__file__))))
    source_audit=subprocess.run([sys.executable,'-X','utf8',str(TOOLS/'validate_figure.py'),str(Path(__file__)),'--json'],
        capture_output=True,text=True,encoding='utf-8')
    (QA/'extension_figures_source_audit.json').write_text(source_audit.stdout,encoding='utf-8')
    assert source_audit.returncode==0, 'Plot source audit failed; inspect saved report'
    if args.action=='prepare':
        print('PREPARED_NO_DATA_READ_NO_FIGURES_RENDERED');return
    try:
        evidence=evidence_loader.load_all_verified()
    except evidence_loader.PendingEvidence as exc:
        write_json(QA/'EXTENSION_FIGURES_STATUS.json',dict(status='PENDING_VERIFIED_EVIDENCE_NO_RENDER',reason=str(exc),
            source_code_sha256=sha(Path(__file__))))
        print('PENDING_VERIFIED_EVIDENCE_NO_RENDER');return
    assert evidence['status']=='ALL_REQUIRED_EVIDENCE_VERIFIED'
    shared_source=ROOT/'extension_plot_sources.json'
    expected_plot_data=plot_source_data(evidence)
    # The results builder owns the shared data file. Do not replace it silently.
    if not shared_source.exists():
        raise RuntimeError('Run build_extension_results.py first to create extension_plot_sources.json from verified evidence')
    assert json.loads(shared_source.read_text(encoding='utf-8'))==expected_plot_data, 'Shared source data do not match current verified evidence'
    write_json(QA/'extension_figure_evidence_sources.json',dict(source_hashes=evidence['source_hashes'],
        shared_source_data=str(shared_source), shared_source_data_sha256=sha(shared_source),
        fields=['records.metrics_raw','records.eligible','records.status','paired.metrics.deltas_pp',
                'paired.metrics.mean_pp','paired.metrics.sd_pp','cells.metrics.primary_mean_percent','cells.metrics.primary_sd_percent'],
        metric_conversion='Record fractions multiplied by100 exactly once. Paired/aggregate loader fields already inpercentage points/percent.',
        scope='Only requested panels selected; fixed-half and other metrics remain in full verified evidence/source data.',
        source_code_sha256=sha(Path(__file__))))
    try:
        reports=[function(evidence) for function in [draw14,draw15,draw16,draw17]]
    except Exception as exc:
        write_json(QA/'EXTENSION_FIGURES_STATUS.json',dict(status='RENDER_OR_AUTOMATED_AUDIT_FAILED',
            error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            source_code_sha256=sha(Path(__file__)),
            note='Any partial outputs are retained for diagnosis and have no delivery approval.'))
        raise
    write_json(QA/'EXTENSION_FIGURES_STATUS.json',dict(status='RENDERED_PENDING_VISUAL_REVIEW',
        figures=[r['figure'] for r in reports],source_code_sha256=sha(Path(__file__)),
        evidence_sources_sha256=sha(QA/'extension_figure_evidence_sources.json')))
    print('RENDERED_PENDING_VISUAL_REVIEW')


if __name__=='__main__':
    main()
