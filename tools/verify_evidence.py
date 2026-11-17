"""Verify supplied evidence hashes and independently aggregate paired summaries.

This does not retrain models or rescore original class labels.
"""
from pathlib import Path
import argparse, csv, hashlib, json, statistics
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):
    with (ROOT/'evidence/current'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):return abs(float(a)-float(b))<=1e-10*max(1,abs(float(a)),abs(float(b)))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    origin=json.loads((ROOT/'provenance/ARCHIVE_ORIGINS.json').read_text(encoding='utf-8'))
    for r in origin['members']:assert sha(ROOT/r['release_file'])==r['release_sha256'],r['release_file']
    primary=json.loads((ROOT/'primary_predictions/MANIFEST.json').read_text(encoding='utf-8'))
    assert len(primary['records'])==210
    for r in primary['records']:assert sha(ROOT/'primary_predictions'/r['file'])==r['sha256']
    allocations=json.loads((ROOT/'evidence/v9/allocation_repeat/PREDICTION_MAP.json').read_text(encoding='utf-8'))
    assert len(allocations['records'])==500
    for r in allocations['records']:assert sha(ROOT/r['prediction_member'])==r['prediction_sha256']
    paired=rows('five_metrics_paired_2000.csv');groups=defaultdict(list)
    assert len(paired)==2000
    for r in paired:groups[(r['dataset'],r['budget'],r['comparator'],r['metric'])].append(float(r['Full_minus_control_pp']))
    long=rows('five_metrics_200_long.csv');assert len(long)==200
    for r in long:
        v=groups[(r['dataset'],r['budget'],r['comparator'],r['metric'])]
        assert len(v)==10
        assert close(statistics.mean(v),r['mean_difference_pp'])
        assert close(statistics.stdev(v),r['sample_SD_pp'])
        assert close(statistics.median(v),r['median_difference_pp'])
    directions=rows('acc_directions_40_verified.csv');assert len(directions)==40
    for r in directions:
        v=groups[(r['dataset'],r['budget'],r['comparator'],'ACC')]
        tol=float(r['ACC_tie_tolerance_pp'])
        assert sum(x>tol for x in v)==int(r['wins'])
        assert sum(abs(x)<=tol for x in v)==int(r['ties'])
        assert sum(x<-tol for x in v)==int(r['losses'])
    selections=rows('selection_runs_300_verified.csv');assert len(selections)==300
    summary=rows('selection_summary_6_verified.csv')
    for r in summary:
        s=[x for x in selections if x['allocation']==r['allocation'] and x['budget']==r['budget']]
        assert len(s)==int(r['selections'])
        assert sum(x['best_BER_tie'].lower()=='true' for x in s)==int(r['best_BER_ties'])
        assert sum(x['default_fallback'].lower()=='true' for x in s)==int(r['default_fallbacks'])
    independent=[r for r in directions if r['control_code'].endswith('-I')]
    matched=[r for r in directions if r['control_code'].endswith('-F')]
    report=dict(status='PASS',archive_members_verified=len(origin['members']),primary_predictions_verified=210,
      logical_allocation_predictions_verified=500,unique_allocation_predictions=len({r['prediction_member'] for r in allocations['records']}),
      paired_values=len(paired),metric_groups=len(long),directions=len(directions),selections=len(selections),
      positive_mean_independent=sum(float(r['mean_ACC_difference_pp'])>0 for r in independent),
      positive_mean_matched=sum(float(r['mean_ACC_difference_pp'])>0 for r in matched),
      independent_abs_mean_below_one_pp=sum(abs(float(r['mean_ACC_difference_pp']))<1 for r in independent),
      best_BER_ties=sum(r['best_BER_tie'].lower()=='true' for r in selections),
      raw_label_rescoring=False,training_performed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise ValueError('Use a new output report')
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
