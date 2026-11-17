"""Plot the sealed two-corpus allocation extension from retained CSVs only.

The drawing block is derived from the executed post-seal report script. No
classes are read and no optimizer or scorer runs. All 40 ACC pairs are retained.
"""
from pathlib import Path
import argparse,csv,hashlib,json,statistics,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_panel_alignment import require_matplotlib_panel_alignment
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,default=HERE.parent/'evidence/allocation_extension/reporting')
parser.add_argument('--output-dir',type=Path,default=HERE/'generated')
a=parser.parse_args();OUTPUT=a.output_dir;OUTPUT.mkdir(parents=True,exist_ok=True)
def rows(name):
 with (a.source_dir/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
paired=rows('paired_values_200.csv');acc=rows('paired_ACC_8.csv');p={'datasets':['20NG','AGNews']}
assert len(paired)==200 and len(acc)==8
for r in paired:r['delta_pp']=float(r['delta_pp']);r['seed']=int(r['seed']);assert r['eligible']=='True'
for r in acc:
 for k in ['mean','sd','median']:r[k]=float(r[k])
 vals=[x['delta_pp'] for x in paired if x['dataset']==r['dataset'] and x['control']==r['control'] and x['metric']=='ACC']
 assert len(vals)==5 and abs(statistics.mean(vals)-r['mean'])<1e-12 and abs(statistics.stdev(vals)-r['sd'])<1e-12
 assert int(r['n_planned'])==int(r['n_eligible'])==5
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.05,3.4),sharey=True)
colors={'consensus-only':'#0072B2','views-only':'#D55E00'}
all_x=[r['delta_pp'] for r in paired if r['metric']=='ACC']
for row in acc:
 if row['mean'] is not None:all_x += [row['mean']-row['sd'],row['mean']+row['sd']]
lo,hi=min(all_x),max(all_x);pad=max((hi-lo)*.08,.3);lo=min(lo-pad,-.2);hi=max(hi+pad,.2)
for ax,policy,title in zip(axes,['CV','at-full-lambda'],['a  Independent query-only selection','b  Full-selected coefficient controls']):
 ax.axvline(0,color='.45',lw=.8,zorder=1)
 for di,ds in enumerate(p['datasets']):
  for ci,cm in enumerate(['consensus-only','views-only']):
   control=cm+'-'+policy;ypos=di*2.5+ci
   row=next(r for r in acc if r['dataset']==ds and r['control']==control)
   pts=[r for r in paired if r['dataset']==ds and r['control']==control and r['metric']=='ACC']
   for j,r in enumerate(pts):ax.scatter(r['delta_pp'],ypos+(j-2)*.055,s=20,marker='o' if ci==0 else '^',color=colors[cm],alpha=.55,zorder=3)
   if row['mean'] is not None:ax.errorbar(row['mean'],ypos,xerr=row['sd'],fmt='D',markersize=4,color=colors[cm],capsize=3,lw=1.25,zorder=4)
 ax.set_title(title,loc='left',pad=11);ax.set_xlim(lo,hi);ax.set_xlabel('Full − control ACC (percentage points)');ax.grid(axis='x',color='.9',lw=.6);ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks([0,1,2.5,3.5],['20NG / consensus','20NG / views','AG News / consensus','AG News / views']);axes[0].invert_yaxis()
fig.subplots_adjust(left=.23,right=.99,bottom=.19,top=.86,wspace=.13)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=['a','b'], row_groups=[['a','b']], json_out=OUTPUT/'fig22_alignment.json', tolerance_pt=1.5, strict=True)
figpath=OUTPUT/'fig22_allocation_extension';fig.savefig(str(figpath)+'.pdf');fig.savefig(str(figpath)+'.png',dpi=260);fig.savefig(str(figpath)+'.svg');plt.close(fig)

(OUTPUT/'fig22_plot_checks.json').write_text(json.dumps(dict(status='PASS',paired_ACC_points=40,groups=8,seeds_per_group=5,exclusions=0,source_sha256={n:hashlib.sha256((a.source_dir/n).read_bytes()).hexdigest() for n in ['paired_values_200.csv','paired_ACC_8.csv']},no_training_or_rescoring=True),indent=2)+'\n',encoding='utf-8')
