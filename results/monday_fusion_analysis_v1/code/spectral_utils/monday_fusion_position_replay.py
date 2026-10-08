"""Replay unchanged position arms; the new grouped CF extension has no old reference."""
import numpy as np
from spectral_utils.monday_fusion_protocol import WORK,RUNS,dump
from spectral_utils.monday_fusion_execution import restore_content,code_seal,native
from spectral_utils.monday_fusion_position import variants

def replay_positions(data):
    code_seal(RUNS['level_reduction_v1'],'position_replay')
    reference=np.load(data.manifest['inputs']['step462']['path']); checks={}; strict_failures={}
    for ce in sorted(set(data.pop.cells)):
        for bk in ['B16','B23','B35','B54']:
            state=restore_content(data,bk,ce); rows=state['bank']['rows']; cache={}
            for recipe,legacy in [('DS_FILTER_EQ','BASE'),('MEAN_DS','GRP')]:
                scores,diag,failures=variants(data,state,recipe,cache)
                for variant in ['POS','PRIOR']+(['PRIOR_CF'] if legacy=='BASE' else []):
                    key=ce+'__'+bk+'__'+legacy+'_'+variant
                    if variant not in scores:
                        if variant=='PRIOR_CF':
                            strict_failures[key]={'reason':failures.get(variant),'registered_change':'No alternate-half or prior-only fallback'}
                            continue
                        raise RuntimeError('Historical position replay arm failed: '+key+' '+str(failures))
                    error=float(np.max(np.abs(scores[variant]-reference[bk+'__'+legacy+'_'+variant][rows])))
                    checks[key]={'max_abs_difference':error,'n_steps':len(rows)}
                    if error>1e-9:
                        dump(WORK/'POSITION_REPLAY_GATE.json',native({'status':'FAIL','checks':checks,'reason':key}))
                        raise RuntimeError(f'Position replay differs: {key}: {error}')
        print('Historical position replay PASS: '+ce,flush=True)
    dump(WORK/'POSITION_REPLAY_GATE.json',native({'status':'PASS','tolerance':1e-9,'comparisons':checks,
        'strict_CF_failures':strict_failures,'new_grouped_CF_has_no_historical_reference':True}))
