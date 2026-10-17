"""Evaluate threshold-free endpoints while independent inner calibration runs."""
import argparse
from .data import Dataset,config,dump
from .scoring import collect,pb_metrics,prm_metrics
from .uncertainty import bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();d=Dataset(config(args.config))
    methods,_=collect(d);pb={};pr={};within={}
    for n,m in methods.items():
        pb[n]=pb_metrics(d,m)
        if (d.prm&m['valid']).any():pr[n],within[n]=prm_metrics(d,m)
    dump(d.out/'PB_METRICS.json',pb);dump(d.out/'PRM_RANKING_METRICS.json',pr)
    bootstrap(d,methods,within)
    for n in ['ct7','token_lsml','token_equal','mindgap']+[x for x in methods if x.startswith('top5__all__')]:
        print(n,pb[n]['macro8'],pr.get(n,{}).get('within_auc'),flush=True)

if __name__=='__main__':main()
