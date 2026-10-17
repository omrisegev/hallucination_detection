"""Small main-score bundles and a complete index to reconstruct other recipes."""
import json
import numpy as np
from spectral_utils.monday_fusion_protocol import RUNS,dump,sha
from spectral_utils.monday_fusion_execution import storage_guard

def package(data,experiment):
    from spectral_utils.monday_fusion_layers import input_state
    out=RUNS[experiment]; selected=json.loads((out/'SELECTION.json').read_text(encoding='utf8'))['winner']
    if selected is None: raise RuntimeError('No complete recipe to package')
    recipe=selected['recipe']; variant=selected['variant']
    if experiment=='estimator_input_v1' and variant=='original': variant='groups_answer'
    control=recipe.split('_')[0]+'_EQ' if recipe.startswith(('MEAN_','HEM_')) else 'DS_FILTER_EQ'
    arrays={'offsets':data.pop.off}; reference=[]
    for bk in data.env['BN']:
        a=np.full(data.pop.total,np.nan); b=a.copy(); written=np.zeros(data.pop.total,int)
        for ce in sorted(set(data.pop.cells)):
            state=input_state(data,experiment,bk,ce,variant); rows=state['bank']['rows']
            a[rows]=state['scores'][recipe]; b[rows]=state['scores'][control]; written[rows]+=1
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not(written==1).all(): raise RuntimeError('Main score coverage failed: '+bk)
        arrays[bk+'__selected']=a; arrays[bk+'__matched_EQ']=b
    highlighted=[]
    if experiment=='estimator_input_v1':
        # Omri explicitly requested this comparison as a main presentation finding.
        # Retain both sides, independently of which recipe wins the original criterion.
        for input_variant in ('groups_answer','channels_cell'):
            for bk in data.env['BN']:
                scores=np.full(data.pop.total,np.nan); written=np.zeros(data.pop.total,int)
                for ce in sorted(set(data.pop.cells)):
                    state=input_state(data,experiment,bk,ce,input_variant); rows=state['bank']['rows']
                    scores[rows]=state['scores']['HEM_MOM']; written[rows]+=1
                if not np.isfinite(scores).all() or not(written==1).all():
                    raise RuntimeError('Highlighted HEM_MOM score coverage failed: '+bk+'/'+input_variant)
                key=bk+'__'+input_variant+'__HEM_MOM'; arrays[key]=scores; highlighted.append(key)
    storage_guard(); np.savez_compressed(out/'STEP_SCORES.npz',**arrays)
    for path in sorted((out/'chunks').glob('*.stats.npz')):
        with np.load(path) as z:
            keys=[k for k in z.files if k.endswith('__step_score')]
        if keys: reference.append({'path':path.relative_to(out).as_posix(),'keys':keys,'sha256':sha(path)})
    dump(out/'STEP_SCORES_MANIFEST.json',{'status':'DEVELOPMENT_MAIN_COMPARISONS_NOT_FINAL_LOCK','recipe':recipe,'variant':variant,
        'matched_control':control,'sha256':sha(out/'STEP_SCORES.npz'),'other_main_scores':reference,
        'highlighted_original_bank_HEM_MOM_comparison':highlighted,
        'remaining_scores':'Reconstruct from bank variant plus chunk fit parameters, pinned code and hash-verified inputs'})
    diag=json.loads((out/'DIAGNOSTICS.json').read_text(encoding='utf8'))
    for stage,pattern in [('layers','*.diagnostics.json'),('lr','*.fits.json')]:
        diag[stage+'_artifact_index']={p.relative_to(out).as_posix():sha(p) for p in sorted((out/stage).glob(pattern))}
    dump(out/'DIAGNOSTICS.json',diag)
    dump(out/'RUN_STATUS.json',{'status':'EXPERIMENT_COMPLETE_REVIEW_PENDING','selected_content_recipe':selected,
        'position_decision':'COMPLETE_WITH_EXPLICIT_ARM_FAILURES','lr':'COMPLETE_WITH_EXPLICIT_ARM_FAILURES',
        'combined_recipe':'PENDING','decision_choice':'OMRI_PENDING','review':'CLAUDE_PENDING'})
