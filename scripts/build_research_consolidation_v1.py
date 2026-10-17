"""Build a source-linked Hebrew reflection. Never infer results from partial runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_consolidation_v1'
REPORT = ROOT / 'docs/reviews/research_consolidation_2026-09-08.html'
REQUIRED = {
    'historical_joint': 'results/historical_joint_refit_v3/REVIEW.json',
    'full_sampling': 'results/localization_full_sampling_v3/evaluation/REVIEW.json',
    'fixed_gate': 'results/fixed_gate_completion_review_v1/REVIEW.json',
}
SOURCES = [
    ('full_shortlist', 'results/localization_full_shortlist_v3/evaluation/METRICS.json', 'full_v3', 'answer_only_gmm'),
    ('historical_five', 'results/historical_fusion_refit_v3/METRICS.json', 'full_v3', 'explicit_per_method'),
    ('historical_joint', 'results/historical_joint_refit_v3/METRICS.json', 'full_v3', 'explicit_per_method'),
    ('full_sampling', 'results/localization_full_sampling_v3/evaluation/METRICS.json', 'full_v3', 'explicit_per_method'),
    ('pilot_corrected_inventory', 'results/fusion_entropy_sampling_v1/EVALUATION.json', 'pilot110_v3', 'answer_only_gmm'),
    ('pilot_trajectory', 'results/fusion_trajectory_imm_v1/EVALUATION.json', 'pilot110_v3', 'answer_only_gmm'),
    ('pilot_context_legacy', 'results/fusion_context_bank_pilot_v1/EVALUATION.json', 'legacy_pilot_not_matched', 'answer_only_gmm'),
    ('pilot_fallback_legacy', 'results/fusion_explicit_fallback_pilot_v1/EVALUATION.json', 'legacy_pilot_not_matched', 'answer_only_gmm'),
    ('pilot_pair_legacy', 'results/fusion_pair_quality_v1/EVALUATION.json', 'legacy_pilot_not_matched', 'answer_only_gmm'),
    ('pilot_conditioning_legacy', 'results/fusion_native_conditioning_v1/EVALUATION.json', 'legacy_pilot_not_matched', 'answer_only_gmm'),
    ('pilot_graph_legacy', 'results/fusion_graph_conditioning_v1/EVALUATION.json', 'legacy_pilot_not_matched', 'answer_only_gmm'),
    ('pilot_ar', 'results/fusion_prediction_quality_v1/EVALUATION.json', 'pilot_source_release', 'answer_only_gmm'),
    ('pilot_gap', 'results/fusion_token_gap_v1/EVALUATION.json', 'pilot_source_release', 'answer_only_gmm'),
]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def update_completed_registry():
    """Only update completed run metadata, after proving it is not a frozen input."""
    path=ROOT/'results/localization_full_benchmark_v3/METHOD_REGISTRY.json'
    for folder in ('localization_full_benchmark_v3','localization_full_shortlist_v3',
                   'localization_full_sampling_v3','historical_fusion_refit_v3','historical_joint_refit_v3'):
        manifest=load(ROOT/'results'/folder/'MANIFEST.json')
        assert all(Path(key).name!='METHOD_REGISTRY.json' for key in manifest.get('hashes',{})), 'Registry is frozen input'
    data=load(path)
    for item in data['methods']:
        if item.get('run')=='../historical_joint_refit_v3/RUN_STATE.json':
            item['status']='FULL_CORRECTED_REFIT_REVIEWED'
            if 'running_arms' in item:
                item['reviewed_arms']=item.pop('running_arms')
            item['report']='../historical_joint_refit_v3/REPORT.html'
            item['continuation']='All245 fits and automated review complete; remaining historical contracts are separate obligations.'
        elif item.get('run')=='../localization_full_sampling_v3/RUN_STATE.json':
            item['status']='FULL_SAMPLING_REVIEWED'
            item['report']='../localization_full_sampling_v3/evaluation/REPORT.html'
    save(path,data)


def obligations():
    result = {}
    for name, relative in REQUIRED.items():
        p = ROOT / relative
        data = load(p) if p.exists() else {}
        state_paths = {
            'historical_joint': ('results/historical_joint_refit_v3/RUN_STATE.json','COMPLETE_REVIEWED_JOINT_EXTENSION'),
            'full_sampling': ('results/localization_full_sampling_v3/RUN_STATE.json','COMPLETE_REVIEWED_FULL_SAMPLING'),
            'fixed_gate': ('results/fixed_gate_completion_review_v1/RUN_STATE.json','COMPLETE_REVIEWED'),
        }
        state_path, expected = state_paths[name]
        run = load(ROOT/state_path) if (ROOT/state_path).exists() else {}
        passed = data.get('status')=='PASS' and run.get('phase')==expected
        result[name] = dict(status='REVIEWED_PASS' if passed else 'UNFINISHED', run_phase=run.get('phase'),
                            review=relative, review_sha256=sha(p) if p.exists() else None)
    return result


def collect():
    rows, sources = [], {}
    required = obligations()
    for study, relative, cohort, default_access in SOURCES:
        p = ROOT / relative
        if not p.exists() or (study in required and required[study]['status'] != 'REVIEWED_PASS'):
            continue
        data = load(p)
        sources[relative] = sha(p)
        review = p.parent / 'REVIEW.json'
        reviewed = review.exists() and load(review).get('status') == 'PASS'
        observed = data.get('rows')
        if isinstance(observed, list):
            prm_population = sum(r['cell'].startswith('prm') for r in observed)
        else:
            prm_population = 6969 if cohort == 'full_v3' else None
        for arm, value in data['metrics'].items():
            prm, pb = value['prm'], value['pb']
            panels = pb.get('macros', {'q8': pb.get('macro_f1'), 'q4': None, 'all': None})
            cells = pb['cells']
            pb_population = int(sum(v['answers'] for v in cells.values()))
            population = prm.get('total_answers', prm_population)
            access = value.get('access', default_access)
            ci = next((x for x in (p.parent/'INTERVALS.json', p.parent/'PAIRED_INTERVALS.json') if x.exists()), None)
            rows.append(dict(study=study, method=arm, cohort=cohort, release_id=data.get('release_id'),
                             evidence_status='REVIEWED' if reviewed else 'SOURCE_AVAILABLE_REVIEW_NOT_VERIFIED',
                             access=access, prm_population=population,
                             prm_valid=prm.get('valid_answers', prm.get('answers')),
                             prm_mixed=prm.get('mixed_answers'), prm_pooled_auc=prm.get('auroc'),
                             prm_fold_mean_auc=prm.get('fold_mean_auc'), prm_within_auc=prm.get('within_answer_auc'),
                             pb_population=pb_population, pb_valid=int(sum(v['valid_decisions'] for v in cells.values())),
                             pb_q4=panels.get('q4'), pb_q8=panels.get('q8'), pb_all=panels.get('all'),
                             pb_cells=cells, metric_source=relative, metric_source_sha256=sources[relative],
                             uncertainty_source=str(ci.relative_to(ROOT)).replace('\\','/') if ci else None,
                             interpretation='Development evidence; compare matching cohort/access/coverage. No winner inferred.',
                             runtime_by_method=None))
    # Include the independently reproduced gate results with the unchanged PRMB scores beside them.
    audit_path = ROOT / REQUIRED['fixed_gate']
    if audit_path.exists() and load(audit_path).get('status') == 'PASS':
        audit = load(audit_path)
        sources[REQUIRED['fixed_gate']] = sha(audit_path)
        references = {r['method']: r for r in rows if r['study'] == 'full_shortlist'}
        for arm, value in audit['arms'].items():
            for rule, met in value['results'].items():
                row = dict(references[arm])
                row.update(study='fixed_gate_review', method=arm+' / '+rule,
                           access='answer-local fusion; '+('answer-local GMM' if rule == 'gmm_bic_saved' else
                                  'other-answer calibration, '+('training labels' if rule.endswith('nested_labels') else 'unlabeled quantile; development-selected q')),
                           pb_q4=met['macros']['q4'], pb_q8=met['macros']['q8'], pb_all=met['macros']['all'],
                           pb_cells=met['cells'], pb_valid=value['valid_decisions'],
                           metric_source=REQUIRED['fixed_gate'], metric_source_sha256=sources[REQUIRED['fixed_gate']],
                           uncertainty_source=REQUIRED['fixed_gate'], prm_unchanged=True)
                rows.append(row)
    # Claude's later shrinkage experiment: only the methods whose saved-score
    # metrics were independently replayed enter this consolidated table.
    relative = 'results/fusion_shrinkage_iu_codex_review_v1/REVIEW_WITH_DIAGONAL_CONTROL.json'
    audit_path = ROOT/relative
    if audit_path.exists():
        audit = load(audit_path)
        assert audit['status'] == 'PASS_FROZEN_SCORE_METRICS_REPLAY'
        sources[relative] = sha(audit_path)
        reference = next(r for r in rows if r['study']=='full_shortlist' and r['method']=='dual__iu')
        for method, met in audit['summaries'].items():
            row = dict(reference)
            row.update(study='full_shrinkage_review', method=method+' / shared entropy-q0.3',
                       evidence_status='FROZEN_SCORE_METRICS_REVIEWED',
                       access='answer-local fusion; saved IU other-answer unlabeled quantile gate; development-selected q',
                       prm_valid=met['prm_valid'], prm_mixed=met['prm_within_n'],
                       prm_pooled_auc=met['prm_pooled'], prm_within_auc=met['prm_within'],
                       prm_fold_mean_auc=None, pb_valid=met['pb_valid'], pb_all=met['pb_all'],
                       pb_q4=sum(v['f1'] for c,v in met['cells'].items() if c.endswith('q4'))/4,
                       pb_q8=sum(v['f1'] for c,v in met['cells'].items() if c.endswith('q8'))/4,
                       pb_cells=met['cells'], metric_source=relative, metric_source_sha256=sources[relative],
                       uncertainty_source=relative,
                       interpretation='Full development, saved scores reviewed. Positive primary versus IU; no clear advantage over diagonal control or entropy on PB. No fusion refit audit or untouched confirmation.')
            rows.append(row)
    return rows, sources


def families(required):
    # Decisions are historical interpretations, not automatic selection on a new maximum.
    specs = [
        ('features', 'פיצ׳רים: moments ו־context', 'האם הקשר לאורך התשובה משפר את ה־fusion?',
         'החלפת בנק הפיצ׳רים וכלל מעבר בין הבנקים.', 'הכיסוי השתנה; שיפור בכיסוי אינו הוכחה לשיפור במיקום.',
         'המתכון המקורי נשאר קו ייחוס קפוא; לא הוכח בנק אופטימלי.', 'האם יש מידע משלים מעבר לאנטרופיה?',
         'results/fusion_context_bank_pilot_v1/REPORT.html', 'REFERENCE_NOT_OPTIMUM'),
        ('groups', 'קבוצות פיצ׳רים ו־fallback', 'האם התאמות Joint תקפות יותר מובילות לתוצאה טובה יותר?',
         'זיהוי קבוצות, טיפול בזוגות, Jacobian וכללי fallback מפורשים.',
         'שופרה היכולת להתאים; כלל מעבר חדש שינה את הבנק גם ב־IU. לא בודד רווח באיכות.',
         'בדיקות תקינות נשמרו; כלל המעבר החדש לא קודם.', 'להפריד יציבות ההתאמה מאיכות המידע.',
         'results/fusion_pair_quality_v1/REPORT.html', 'NOT_PROMOTED'),
        ('ar', 'פיצ׳רי AR ו־token gap', 'האם חיזוי הטלמטריה או הסתברות הטוקן שסופק מוסיפים מידע?',
         'תשע קואורדינטות AR; בניסוי נפרד החלפת שלוש קואורדינטות surprisal.',
         'בפיילוטים לא נמצא שיפור משכנע במיקום; תקפות התאמה גבוהה יותר לא הספיקה.',
         'לא אומצו כמתכון חדש.', 'אלה בדיקות מימושים מסוימים, לא פסילה של KalmanNet או כל פיצ׳ר חיזוי.',
         'results/fusion_prediction_quality_v1/REPORT.html', 'PILOT_NOT_PROMOTED'),
        ('fusion', 'IU / CONT / Joint', 'מה משקלי fusion מוסיפים מעבר לממוצע ולאנטרופיה?',
         'מפות משקלים שונות; התאמה מקומית לעומת התאמה על תשובות אחרות.',
         'היתרון תלוי משימה ופרוטוקול. הטבלה המלאה מפרידה מסלולי גישה ומדדים.',
         'IU נשאר reference; אין הכרזה אוטומטית על מנצח.', 'ההשוואה הבאה תהיה עם אותו gate בדיוק.',
         'results/historical_fusion_refit_v3/REPORT.html', 'REFERENCE'),
        ('graph', 'גרפים ו־conditioning', 'האם מבנה הגרף מוסיף מעבר לשינוי משקלים או רגולריזציה?',
         'גרף מקורי, גרף מעורבב ו־lambda0; גבולות התניה מספרית.',
         'ב־shortlist המלא הגרף לא ביסס יתרון מול הבקרות. אין להסתפק ביתרון מול Joint היררכי אם מפת המשקלים שונה.',
         'הבקרות נשמרו; לא נפתח חיפוש λ נוסף.', 'הווריאנטים ההיסטוריים מוצגים בנפרד עם בקרת model-inverse המתאימה.',
         'results/localization_full_shortlist_v3/evaluation/REPORT.html', 'NOT_PROMOTED'),
        ('shrinkage', 'Shrinkage בתוך IU — ניסוי Claude ובדיקה עצמאית', 'האם אומדן covariance מובנה משפר את ה־fusion בתוך תשובה?',
         'אותם חלונות, בנק, ציונים ל־IU וספי entropy-q0.3; שינוי covariance בשלוש נקודות בחישוב.',
         'full/joint/LW מוסיף 0.0028 AUC בתוך תשובה ו־0.61 נקודת PB מול IU; רווחי סמך 97.5% חיוביים. ביקורת solve/diag/1.0 נותנת 31.68% מול 31.77%; אין יתרון ברור למבנה Joint מולה.',
         'מועמד פיתוח מול IU, לא מוביל מאומת. לשמור את הביקורת האלכסונית ואת האנטרופיה. שינוי solve בלבד עם target של Joint פוגע ב־PRMB.',
         'שינוי rho נתמך בהשוואת PRMB; אינו הסבר מוכח לכל הרווחים. מול entropy ב־PB לא הוכח יתרון. וריאנטים משניים אינם מנצחים שנבחרו מראש.',
         'results/fusion_shrinkage_iu_codex_review_v1/REPORT.md', 'CANDIDATE'),
        ('historical_joint', 'הווריאנטים ההיסטוריים של Claude', 'מה תרמו gate / LIU / diagonal מעבר ל־Joint ול־CONT?',
         'עשר זרועות קיימות: CONT, Joint, gate050/100, LIU010/050, diag010/050, model-inverse0 ו־permutation010. gate050/100 משנים את משקלי הפיצ׳רים; זה אינו ה־gate של החלטת יש/אין שגיאה.',
         'ההשוואה חייבת להשתמש בתוויות ובקבוצות המקור המתוקנות; מספרים מהמחשב השני נשמרים כהיסטוריה.',
         'הושלם ונבדק; אין בחירת מנצח על פי נקודה בלבד.' if required['historical_joint']['status']=='REVIEWED_PASS' else 'טרם הושלם; אין מסקנה מההתאמות החלקיות.',
         'התאמה על תשובות אחרות וסף עם תוויות הם גישה שונה מהמסלול המקומי.',
         'results/historical_joint_refit_v3/REPORT.html', required['historical_joint']['status']),
        ('sampling', 'דגימה: risk, נמוך/גבוה, quantiles ו־DUFS', 'אילו חלונות מועילים להתאמה מתוך תשובה אחת?',
         'שמונת כללי הדגימה ושבעת מנועי ה־fusion שכבר הוקפאו; ניקוד כל החלונות נשמר.',
         'הרווח הגבוה ב־AUC המאוגם בפיילוט אינו מספיק. יש לבדוק דירוג בתוך תשובה ומיקום על כלל האוכלוסייה.',
         'הבדיקה המלאה הושלמה; ההכרעה מוצגת בטבלת ההשוואות הקבועות.' if required['full_sampling']['status']=='REVIEWED_PASS' else 'הבדיקה המלאה עדיין לא הסתיימה; אין הכרעה מדעית.',
         'ניסוי הדגימה הקפוא משתמש ב־GMM המקורי; תוצאותיו עם gate האנטרופיה החדש עדיין לא נבדקו. דגימת שורות להתאמה אינה חיסכון חישובי מקצה לקצה או פתרון מובטח לשגיאות קצרות.',
         'results/localization_full_sampling_v3/evaluation/REPORT.html', required['full_sampling']['status']),
        ('scale', 'סקאלה לעומת מיקום', 'האם עליית AUC משקפת שיפור בתוך התשובה?',
         'בדיקות שמפרידות שינוי סדר מקומי מהזזה ומתיחה של ציוני תשובות.',
         'קבוע לכל תשובה יכול לשפר AUC מאוגם בלי לשנות סדר צעדים. זה הסבר חלופי לרווחי דגימת סיכון.',
         'דיווח AUC בתוך תשובה אומץ כחלק קבוע מההשוואה.', 'אין להפוך אבחון סקאלה למקומון חדש בלי ראיה.',
         'results/fusion_entropy_sampling_v1/REPORT.html', 'INFRASTRUCTURE_ADOPTED'),
        ('trajectory', 'Fusion בין trajectories: mean / GLS / IMM', 'האם שילוב עקומות משפר מעבר למשקלי הפיצ׳רים?',
         'שילוב עקומות הסיכון שכבר קיימות; IMM הוא מימוש temporal שנבדק בפועל.',
         'הממוצע נתן רווח קטן ב־PRMB בתוך תשובה; IMM החליש PRMB למרות נקודת PB גבוהה יותר. רווח PB לא הוכח.',
         'הממוצע נשמר כמועמד; GLS ו־IMM לא קודמו.', 'האם רווח הממוצע במיקום מתגלה עם gate משותף?',
         'results/localization_full_shortlist_v3/evaluation/REPORT.html', 'CANDIDATE_MEAN_ONLY'),
        ('temporal_legacy', 'HMM / Kalman / BOCPD קודמים', 'האם עיבוד רצף משפר את הקריאה מהסיכון?',
         'מחליקי רצף ושינויי משטר בפיילוטים מוקדמים.',
         'לא בוסס שיפור שניתן לקדם; הפרוטוקולים והאוכלוסיות המוקדמים שונים מהבנצ׳מרק המלא.',
         'נשמרו כהיסטוריה, לא ככיוון פעיל חדש.', 'Kalman רגיל אינו KalmanNet; שינוי משטר אינו בהכרח תחילת שגיאה.',
         'results/fused_trajectory_readout_pilot_v1/REPORT.html', 'LEGACY_PILOT'),
        ('gate_sim', 'סימולציות GMM ו־gate', 'האם תלות סדרתית גורמת לפתיחת gate ללא שגיאה?',
         'נתונים סינתטיים וכיול AR; בלי שינוי תחזיות על תשובות אמיתיות.',
         'מסננים יכולים לשנות את התנהגות BIC. כיול AR לא עבר את כל תנאי הבדיקה הסינתטיים.',
         'אבחון נשמר; לא אומץ תיקון gate על בסיס סימולציות אלה.', 'כשל בסימולציה אינו שיעור שגיאה אמיתי על ProcessBench.',
         'results/fusion_gate_calibration_v1/REPORT.html', 'SIMULATION_NOT_PROMOTED'),
        ('gate', 'Gate פשוט של Claude', 'כמה מהפער נובע מהחלטת יש/אין שגיאה?',
         'אותם peaks; החלפת GMM בסף אנטרופיה ממוצעת.',
         'השחזור העצמאי מאשר את הרווח הגדול. q0.3 הוא המועמד הקבוע ללא חיפוש נוסף.',
         'מועמד לתיקון, עדיין לא אישור על נתונים שלא שימשו לפיתוח.',
         'הכיול משתמש בתשובות אחרות; יתרון ל־fusion מול אנטרופיה עדיין נדרש.',
         'results/fixed_gate_completion_review_v1/REPORT.md', 'CANDIDATE'),
        ('location', 'מיקום השגיאה הראשונה', 'למה דירוג סביר אינו הופך לפגיעה מדויקת?',
         'ספירת peaks מוקדמים/מאוחרים, gate מדכא וקריאת max לעומת top10 היסטורית.',
         'גם לאחר תיקון gate, שיא הסיכון אינו תמיד השגיאה הראשונה. שני המדדים בודקים יכולות שונות.',
         'זה צוואר הבקבוק הבא; בדיקת readout אחת הוגדרה, לא התחילה.',
         'האם טוקן קיצוני יחיד או הצטברות סיכון מאוחרת מסיטים את ההחלטה?',
         'results/localization_full_error_modes_v3/REPORT.html', 'NEXT_DIAGNOSTIC'),
        ('protocol', 'תוויות, קבוצות מקור ושחזור', 'באילו שלבים ההשוואה הקודמת לא הייתה מהימנה?',
         'תיקוני תוויות PRMB, קבוצות מקור, התאמות היסטוריות מחדש והרחבה מהפיילוט.',
         'פיילוטים לא חזו היטב ביצועים מלאים; שינוי תוויות/קבוצות/מדד חייב גשר מפורש.',
         'תשתית ההשוואה המתוקנת אומצה; זו אמינות מחקרית, לא שיפור אלגוריתמי.',
         'מובילים חסרים ואישור חיצוני נותרים רשומים עד השלמה.',
         'docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md', 'INFRASTRUCTURE_ADOPTED'),
    ]
    keys = ('id','family','question','changed','finding','adopted','open_question','source','decision_status')
    result=[dict(zip(keys, row)) for row in specs]
    if required['historical_joint']['status']=='REVIEWED_PASS':
        data=load(ROOT/'results/historical_joint_refit_v3/METRICS.json')['metrics']
        row=next(r for r in result if r['id']=='historical_joint')
        pb=lambda arm:100*data[arm]['pb']['macros']['all']
        row['finding']=(f"PB all8: Joint היררכי {pb('internal_joint'):.2f}%, model-inverse ללא גרף "
                        f"{pb('internal_joint_modelinv_lam0'):.2f}%, LIU010 {pb('internal_joint_liu010'):.2f}%. "
                        'רוב ההפרש מול Joint קיים כבר בבקרה ללא גרף. LIU משפר PRMB מול model-inverse0 '
                        'במרווחים exploratory; היתרון ב־PB ובקרת permutation010 אינו מוכרע. '
                        'אין בקרת permutation050 תואמת בפאנל הזה.')
        row['adopted']='ההשוואה הושלמה ונבדקה. gate050/100 ו־diagonal לא קודמו; LIU נשמר כהשוואה מכניסטית, לא כמוביל PB מוכח.'
        row['open_question']='internal_cont עם קבוצות פנימיות אינו fixed_family_cont_unguarded עם קבוצות provenance קבועות; השם CONT לבדו אינו מזהה מתכון.'
    return result


def paired_evidence(paths):
    """Normalize existing registered contrasts; compute no new experiments or intervals."""
    result=[]
    for relative in paths:
        data=load(ROOT/relative)
        if 'localization_full_shortlist_v3' in relative:
            for row in data['paired'].values():
                for endpoint,delta,ci in [('prm_pooled',row['prm_delta'],row['prm']['ci95']),
                                          ('within',row['within_delta'],row['within']['ci95'])]+[
                        ('pb_'+p,row['pb'][p]['delta'],row['pb'][p]['ci95']) for p in ('q4','q8','all')]:
                    result.append(dict(source=relative,left=row['left'],right=row['right'],scope='all',endpoint=endpoint,
                                       difference=delta,ci95=ci,prm_common_answers=row['prm_left']['answers']))
        elif 'localization_full_sampling_v3' in relative:
            for row in data['paired']:
                for endpoint,delta in row['delta'].items():
                    result.append(dict(source=relative,left=row['left'],right=row['right'],scope=row['scope'],endpoint=endpoint,
                                       difference=delta,ci95=row['intervals'][endpoint]['ci95'],prm_common_answers=row['prm_common_answers']))
        elif relative.endswith('MECHANISM_CONTRASTS.json'):
            for row in data['contrasts']:
                for endpoint,v in row['endpoints'].items():
                    result.append(dict(source=relative,left=row['candidate'],right=row['control'],scope='all',
                                       endpoint='within' if endpoint=='prm_within_auc' else endpoint,
                                       difference=v['difference'],ci95=[v['low'],v['high']],prm_common_answers=row['prm_common_answers']))
    for row in result:
        ci=row['ci95']
        if ci is not None and any(v is None or not math.isfinite(v) for v in ci):
            ci=None;row['ci95']=None
        row['interpretation']=('לא מוגדר' if ci is None else 'יתרון exploratory' if ci[0]>0 else
                               'ירידה exploratory' if ci[1]<0 else 'הטווח כולל אפס; אין הכרעה')
        row['selection_uncertainty_included']=False
        row['multiplicity_adjusted']=False
    return result


def esc(value):
    return html.escape(str(value))


def link(relative, label):
    path = ROOT / relative
    if not path.exists():
        return esc(label) + ' — טרם קיים דוח סופי'
    import os
    href = os.path.relpath(path, REPORT.parent).replace('\\','/')
    return f'<a href="{esc(href)}">{esc(label)}</a>'


def fmt(value, percent=False):
    return '—' if value is None else f'{value*100:.2f}%' if percent else f'{value:.4f}'


def metric_table(rows):
    body = ''
    for r in rows:
        context=r['study']+' / '+r['cohort']+(' / '+r['release_id'] if r.get('release_id') else '')
        cells = [context, r['method'], r['access'], fmt(r['prm_pooled_auc']),fmt(r['prm_fold_mean_auc']),
                 fmt(r['prm_within_auc']),fmt(r['pb_q4'],True),fmt(r['pb_q8'],True),fmt(r['pb_all'],True),
                 f"{r['prm_valid']}/{r['prm_population']}",f"{r['pb_valid']}/{r['pb_population']}"]
        body += '<tr>' + ''.join('<td>'+esc(x)+'</td>' for x in cells) + '<td>'+link(r['metric_source'],'מקור')+'</td></tr>'
    headers = ['קבוצת תוצאות / פרוטוקול','שיטה','גישה לנתונים','PRMB pooled','PRMB fold mean','PRMB within','PB Q4','PB Q8','PB all8','כיסוי PRMB','כיסוי PB','ראיה']
    return '<div class="scroll"><table class="metrics"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def render(ledger):
    complete = ledger['status'] == 'COMPLETE_REVIEWED_CONSOLIDATION'
    text = '''<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>סדר במחקר — Fusion ו־Localization</title>
<style>body{font:17px/1.7 system-ui,Arial;margin:auto;max-width:1500px;padding:28px;color:#172739;background:#f6f8fb}h1,h2,h3{line-height:1.3}section,.card{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:22px;margin:18px 0}a{color:#005eaa}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #dbe3ed;text-align:start;vertical-align:top}th{background:#e8f0f9;position:sticky;top:0}.scroll{overflow:auto;max-height:680px}.warning{background:#fff3cd;padding:16px;border-radius:8px}.good{background:#e8f5ec;padding:16px;border-radius:8px}.metrics td:nth-child(n+4){white-space:nowrap}summary{cursor:pointer;font-weight:600;padding:12px}input{font:inherit;padding:8px;width:min(95%,600px)}.flow{display:flex;gap:10px;flex-wrap:wrap}.flow div{padding:14px;border:1px solid #b8c9dd;border-radius:8px;background:#edf4fc}.small{font-size:14px;color:#4b596c}code{direction:ltr;unicode-bidi:isolate}button{font:inherit;padding:8px}@media print{input,button{display:none}.scroll{max-height:none;overflow:visible}body{background:white;font-size:12px}section{break-inside:avoid}}</style></head><body>
<h1>איפה אנחנו עומדים — Fusion ו־Localization</h1>'''
    text += '<p class="'+('good' if complete else 'warning')+'">'+('שלב ההשלמה וה־Reflection הסתיים. ניסויי השיפור הבאים טרם התחילו.' if complete else 'טיוטת עבודה: ההשלמות עדיין רצות. אין כאן מסקנות מתוצאות חלקיות.')+'</p>'
    text += '<p>המסמך מפריד בין מה שנמדד, מה אומץ ומה עדיין חסר. כל המספרים הם נתוני פיתוח שנחשפו למחקר; אין כאן הכרזה על מוביל מאומת.</p>'
    text += '<p>'+link('results/research_consolidation_v1/LEDGER.json','רישום ממוכן מלא')+' · '+link('results/research_consolidation_v1/METRICS.csv','כל המדדים ב־CSV')+' · '+link('docs/experiments/RESEARCH_CONSOLIDATION_20260908.md','התוכנית שאושרה')+'</p>'
    text += '<section><h2>מה הושלם ומה עדיין פתוח</h2><table><tr><th>חובה</th><th>מצב</th><th>בדיקה</th></tr>'
    for name, row in ledger['obligations'].items():
        text += '<tr><td>'+esc(name)+'</td><td>'+('הושלם ונבדק' if row['status']=='REVIEWED_PASS' else 'לא הושלם')+'</td><td>'+link(row['review'],'בדיקת תקינות')+'</td></tr>'
    text += '</table><p>השלמת החובות האלה אינה השלמת כל המובילים ההיסטוריים. הרישום למטה משאיר את ההשוואות שטרם הותאמו גלויות.</p></section>'
    text += '<section><h2>מה בדיוק השיטה עושה</h2><div class="flow"><div>תשובה אחת<br>N חלונות × P פיצ׳רים</div><div>נרמול ומשקלי fusion<br>נלמדים בתוך התשובה</div><div>עקומת סיכון<br>דירוג ומיקום צעדים</div><div>gate אנטרופיה q=0.3<br>כיול מתשובות אחרות, בלי תוויות</div><div>אין שגיאה<br>או צעד השגיאה שנבחר</div></div><p>FEATURES הוא ציר שילוב המדדים. TOKENS / TRAJECTORY הם ציר התצפיות לאורך התשובה: בחירת חלונות, עיבוד כרונולוגי או שילוב עקומות הם פעולות שונות. Joint בשם השיטה אינו אומר ששני הצירים אופטימליים יחד.</p></section>'
    text += '<section><h2>למה שני הבנצ׳מרקים לא חייבים להשתפר יחד</h2><table><tr><th>בנצ׳מרק</th><th>השאלה</th><th>האתגר שנותר</th></tr><tr><td>PRMBench</td><td>האם צעדים שגויים מקבלים סיכון גבוה מצעדים נכונים?</td><td>להוסיף מידע לדירוג בתוך תשובה. הזזה של כל ציוני התשובה יכולה להעלות AUC מאוגם בלי לשנות מיקום.</td></tr><tr><td>ProcessBench</td><td>האם מצאנו בדיוק את השגיאה הראשונה, או זיהינו תשובה נקייה?</td><td>gate תקין וגם מיקום מדויק. סיכון גבוה אחרי שהטעות התפשטה אינו זיהוי תחילתה.</td></tr></table><p>Q8 הוא גשר לפיילוט הישן; all8 כולל גם Q4. אין להחליף ביניהם כאשר מתארים שינוי.</p></section>'
    audit = load(ROOT/REQUIRED['fixed_gate']) if (ROOT/REQUIRED['fixed_gate']).exists() else None
    if audit:
        delta = audit['paired_bootstrap']['contrasts']['quantile_0.3_minus_gmm']
        text += '<section><h2>התיקון הגדול שנמצא: ה־gate</h2><p>ללא שינוי משקלי fusion או peaks, IU עולה מ־20.00% ל־31.16% ב־PB all8. ההפרש הוא '+f"{100*delta['point_difference']:.2f}"+' נקודות אחוז; רווח סמך ['+f"{100*delta['ci95'][0]:.2f}, {100*delta['ci95'][1]:.2f}"+']. הסף עם תוויות מגיע ל־31.31%.</p><p>ה־bootstrap מזווג לפי קבוצות מקור, 10,000 דגימות. הוא מותנה בתחזיות הקבועות ואינו כולל את בחירת detector/q. השיפור הוא תיקון readout משמעותי; עדיין לא הוכח יתרון fusion על אנטרופיה בלבד.</p>'
        text += metric_table([r for r in ledger['metrics'] if r['study']=='fixed_gate_review' and r['method'].startswith(('dual__iu /','entropy_parent /'))])
        text += '<p>ההשוואה משתמשת בכיול חיצוני של gate בלבד. בבדיקה הועתקו כל המיקומים והכיסוי בדיוק; PRMB לא השתנה.</p></section>'
    text += '<section><h2>Reflection: איפה איבדנו כיוון</h2><ol><li>הרחבנו וריאנטים לפני שהתמונה המלאה והבקרות הפשוטות התייצבו. פיילוטים סיפקו אופטימיות שלא נשמרה על כל האוכלוסייה.</li><li>לעיתים פירשנו התאמה תקפה יותר או conditioning טוב יותר כהתקדמות באיכות. אלה הישגים הנדסיים עד שמודגם רווח במשימה.</li><li>ערבבנו בין AUC מאוגם, דירוג בתוך תשובה, מיקום ראשון והחלטת אין שגיאה. שינוי באחד אינו הוכחה לאחרים.</li><li>השוואות בין מחשבים ופרוטוקולים שונים דרשו refit וגשר תוויות/קבוצות. הצגה של מספרים ללא ההקשר הזה יצרה תמונת התקדמות מטעה.</li><li>ה־gate של Claude מבודד כעת רכיב שהיה אפשר לבדוק מוקדם יותר. זו אחריות בתכנון ובדיווח, לא סיבה לוותר על הבקשה המקורית ללמוד fusion מתשובה אחת.</li></ol></section>'
    text += '<section><h2>מה ניסינו, מה אומץ ומה לא הוכרע</h2>'
    labels={'REFERENCE_NOT_OPTIMUM':'קו ייחוס; לא הוכח כאופטימלי','NOT_PROMOTED':'לא קודם',
            'PILOT_NOT_PROMOTED':'פיילוט בלבד; לא קודם','REFERENCE':'קו ייחוס',
            'REVIEWED_PASS':'הושלם ונבדק','UNFINISHED':'לא הושלם','INFRASTRUCTURE_ADOPTED':'אומץ כתשתית',
            'CANDIDATE_MEAN_ONLY':'הממוצע נשאר מועמד','LEGACY_PILOT':'פיילוט היסטורי',
            'SIMULATION_NOT_PROMOTED':'סימולציה; לא אומץ','CANDIDATE':'מועמד לשיפור','NEXT_DIAGNOSTIC':'האתגר הבא'}
    for family in ledger['families']:
        text += '<details><summary>'+esc(family['family'])+' — '+esc(labels[family['decision_status']])+'</summary>'
        for key,label in [('question','השאלה'),('changed','מה השתנה'),('finding','מה למדנו'),('adopted','מה אומץ'),('open_question','מה עוד פתוח')]:
            text += '<p><strong>'+label+': </strong>'+esc(family[key])+'</p>'
        text += '<p>'+link(family['source'],'ראיות ומדדים')+'</p></details>'
    text += '</section><section><h2>תוצאות מלאות שניתן להשוות</h2><p>אותה אוכלוסיית v3 אינה אומרת אותה גישה לנתונים. historical משתמש בתשובות אחרות ובכיול עם תוויות. עמודת fold mean נשמרת בנפרד מ־pooled; המכנים גלויים. שורות שחוזרות בדוחות שונים הן references, לא ניסויים עצמאיים.</p><input id="search" placeholder="חיפוש שיטה או מקור בטבלאות" aria-label="חיפוש שיטה">'
    text += metric_table([r for r in ledger['metrics'] if r['cohort']=='full_v3' and r['study'] not in ('fixed_gate_review','full_shrinkage_review')])+'</section>'
    text += '<section><h2>Shrinkage בתוך IU — השוואה עם gate משותף</h2><p>כל השורות כאן משתמשות באותם ספי entropy-q0.3 השמורים של IU ובאותו readout. לכן משווים ביניהן את תרומת ה־fusion. אין לחסר אותן משורות ה־GMM בטבלה הקודמת כשיפור אלגוריתמי. המועמד הראשי משתפר מול IU; אין יתרון ברור מול הביקורת האלכסונית או מול אנטרופיה ב־PB.</p>'
    text += metric_table([r for r in ledger['metrics'] if r['study']=='full_shrinkage_review'])+'</section>'
    text += '<section><h2>איפה נמדד רווח — השוואות מזווגות</h2><p>הטבלה מציגה את שני היעדים בנפרד. הפרשי PB הם נקודות אחוז; הפרשי within הם יחידות AUC. יתרון exploratory אינו אישור לאחר בחירת שיטה. גם תוצאות עם טווח שכולל אפס נשארות גלויות.</p><div class="scroll"><table><tr><th>שיטה</th><th>מול</th><th>מדד</th><th>הפרש</th><th>95% CI</th><th>פירוש</th><th>מקור</th></tr>'
    for row in ledger['paired_evidence']:
        if row['scope']!='all' or row['endpoint'] not in ('within','pb_all'):
            continue
        # Limit the visible summary to fixed full-grid sampling controls, historical
        # mechanism contrasts and shortlist comparisons to IU; all pairs remain in JSON.
        sample='localization_full_sampling_v3' in row['source']
        if sample and (row['right'] not in ('sample_full__iu','sample_full__graph010') or
                       not row['left'].endswith(('__iu','__graph010'))):
            continue
        if 'localization_full_shortlist_v3' in row['source'] and row['right']!='dual__iu':
            continue
        pct=row['endpoint']=='pb_all'
        delta='—' if row['difference'] is None else f"{row['difference']*(100 if pct else 1):+.4f}"
        ci='—' if row['ci95'] is None else '['+', '.join(f'{v*(100 if pct else 1):+.4f}' for v in row['ci95'])+']'
        text+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (row['left'],row['right'],row['endpoint'],delta,ci,row['interpretation']))+'<td>'+link(row['source'],'ראיה')+'</td></tr>'
    text+='</table></div></section>'
    text += '<section><h2>מה הוצג למנחים קודם — ומה המשמעות כיום</h2><p>'+link('docs/meetings/Advisor_Update_Aug27_2026.md','המכתב מ־27 באוגוסט')+' הוא מקור היסטורי, לא שחזור חדש של הניסויים.</p><table><tr><th>משימה</th><th>מה דווח אז</th><th>איך מתייחסים כעת</th></tr>'
    for item in ledger['advisor_history']:
        text+='<tr>'+''.join('<td>'+esc(item[k])+'</td>' for k in ('task','reported','current_interpretation'))+'</tr>'
    text+='</table><p>הכיוון של תשובה אחת היה בקשה מפורשת שלך. הכשל בתהליך היה בהרחבת הניסויים ובפירוש פיילוטים לפני בידוד הרכיבים, ולא בעצם הבחירה לבדוק את הרעיון.</p></section>'
    text += '<section><h2>היסטוריה ופיילוטים — אינם טבלת מובילים</h2><p>הטבלאות שומרות את המספרים ואת המקורות המקוריים. legacy אינו בר־השוואה ישירה ל־v3; תיקוני תוויות וקבוצות יכולים לשנות את האומדן. מספר שורות אינו מספר רעיונות עצמאיים.</p><details><summary>פתיחת כל תוצאות הפיילוטים</summary>'+metric_table([r for r in ledger['metrics'] if r['cohort']!='full_v3'])+'</details></section>'
    text += '<section><h2>ממצאים מזווגים והחובות לבנצ׳מרק</h2><p>אין לבחור מנצח מעמודת הציון הגבוה ביותר. קובצי ההשוואות מכילים מכנים משותפים ורווחי סמך; ההשוואות הקודמות הן exploratory ואינן תיקון לכל בחירות המחקר.</p><ul>'
    for source in ledger['contrast_sources']:
        text += '<li>'+link(source,source.split('/')[-2]+' / '+source.split('/')[-1])+'</li>'
    text += '</ul><ul>'
    for item in ledger['backlog']:
        text += '<li><strong>'+esc(item['name'])+': </strong>'+esc(item['status'])+' — '+esc(item['reason'])+'</li>'
    text += '</ul><details><summary>המועמדים ההיסטוריים שעדיין רשומים כחסרים</summary><table><tr><th>משפחה</th><th>מה חסר לפי הרישום</th></tr>'
    for item in ledger['historical_registry_snapshot']['methods']:
        if 'PENDING' in item['status']:
            text+='<tr><td>'+esc(item['method'])+'</td><td>'+esc(item.get('pending',item.get('scope',item['status'])))+'</td></tr>'
    text += '</table><p>זה snapshot של רישום החובות. דוח חדש של Claude דורש אימות וקליטה לפני סימון ההשוואה כהושלמה.</p></details></section><section><h2>מה קורה לאחר שחוזרים עם הדוח</h2><ol><li>ניסיון משותף־gate: שבע השיטות הקיימות, אותם ספי q0.3. IU מול אנטרופיה והממוצע מול IU הם שתי השוואות ההכרעה.</li><li>לאחר חזרה עם תשובה: IU ואנטרופיה בלבד, max לעומת ממוצע top10 לצעד. אין חיפוש מספר אחר.</li><li>אחרי כל ניסיון עוצרים ומציגים מסקנה. תוצאה שלילית אינה מפעילה עוד sweep.</li></ol><p>שני הניסויים האלה לא הופעלו כחלק מההשלמה. אין התחייבות למנצח; יתרון ל־fusion חייב להיות מודגם, ולא להיגזר ממטרת המחקר.</p></section>'
    text += '<p class="small">בדיקה מקומית אוטומטית ובאותו session; אין טענה לביקורת מדעית חיצונית. מקורות וחתימות נשמרים ברישום הממוכן.</p><script>document.getElementById("search").addEventListener("input",function(){let q=this.value.toLowerCase();document.querySelectorAll("table.metrics tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></body></html>'
    return text


def main(draft=False):
    required = obligations()
    complete = all(v['status']=='REVIEWED_PASS' for v in required.values())
    if not draft and not complete:
        raise RuntimeError('Cannot finalize before all three obligations have reviewed PASS artifacts: '+str(required))
    if complete and not draft:
        update_completed_registry()
    rows, sources = collect()
    contrast_sources = []
    for folder in ('results/localization_full_shortlist_v3/evaluation','results/historical_fusion_refit_v3',
                   'results/historical_joint_refit_v3','results/localization_full_sampling_v3/evaluation',
                   'results/fusion_entropy_sampling_v1'):
        for name in ('INTERVALS.json','MECHANISM_CONTRASTS.json','PAIRED_INTERVALS.json'):
            p = ROOT/folder/name
            # Only completed studies contribute comparisons; never publish partial bootstrap ranks.
            review = p.parent/'REVIEW.json'
            if p.exists() and review.exists() and load(review).get('status')=='PASS':
                relative = str(p.relative_to(ROOT)).replace('\\','/')
                contrast_sources.append(relative); sources[relative]=sha(p)
    registry_path = ROOT/'results/localization_full_benchmark_v3/METHOD_REGISTRY.json'
    registry = load(registry_path)
    sources[str(registry_path.relative_to(ROOT)).replace('\\','/')]=sha(registry_path)
    sources[str(Path(__file__).relative_to(ROOT)).replace('\\','/')]=sha(Path(__file__))
    advisor_source='docs/meetings/Advisor_Update_Aug27_2026.md'
    sources[advisor_source]=sha(ROOT/advisor_source)
    nrm_source='scripts/neutral_residual_mode_prmbench_confirmation.py'
    sources[nrm_source]=sha(ROOT/nrm_source)
    advisor_history=[
        dict(task='ProcessBench localization',reported='31.36% לעומת Mind the Gap 25.71%, בפרוטוקול דאז.',
             current_interpretation='ב־v3 המסלול ההיסטורי שומר על רמה של כ־34–35%; Mind the Gap עדיין דורש השוואה תואמת. אין לחסר מספרים משני הפרוטוקולים כאפקט מבוקר.',source=advisor_source),
        dict(task='24 תאי final-answer detection',reported='13 שיטות; IU-PCR 0.7761, DUFS-LIU 0.7766, equal-family 0.7810, DEEM-B3 0.7813.',
             current_interpretation='משימת זיהוי ברמת תשובה, לא localization. זו היסטוריית המחקר והיעד לניסוי העברה עתידי.',source=advisor_source),
        dict(task='Family-NRM על PRMBench response detection',reported='0.7206 → 0.7252; במכתב דווח רווח +0.46 נקודת AUC ו־CI חיובי.',
             current_interpretation='אין לערבב עם דירוג צעדים. בדיקת הקוד מצאה תווית classification ברמת תשובה; באג אינדקסי הצעדים אינו מבטל אותה ישירות. תוקף ה־CI לפי קבוצות המקור המתוקנות טרם נבדק מחדש.',source=advisor_source),
    ]
    backlog = [
        dict(name='מובילים היסטוריים ו־Mind the Gap', status='דורש התאמה ובדיקת ראיות', reason='נקלוט תוצאות Claude כשקיימות; אין להסיק השלמה מרישום ישן או מספר היסטורי לא תואם.'),
        dict(name='ניסיון gate משותף', status='מוגדר, טרם התחיל', reason='להפריד תרומת fusion מתרומת gate.'),
        dict(name='ניסיון max מול top10', status='מוגדר, טרם התחיל', reason='לבודד את סיכום הטוקנים לצעד ללא שינוי פיצ׳רים.'),
        dict(name='MR-GSM8K / אישור חיצוני', status='ממתין להקפאה ולבדיקת חפיפה', reason='דאטה פיתוח מלא אינו test שלא נחשף; נדרש audit של מקורות GSM8K.'),
        dict(name='LOCA / Diverging Flows / KalmanNet / Shlezinger', status='עדיפות נמוכה', reason='מנגנונים אפשריים לשירות fusion, לא שיטות שהוכחו כאן או תחליף למחקר הנוכחי.'),
        dict(name='דגימה אדפטיבית חדשה וגודל חלון אופטימלי', status='נדחה', reason='השלמת הדגימה הקיימת אינה אישור לחיפוש נוסף או הוכחה ל־width אופטימלי.'),
        dict(name='24 תאי final-answer detection', status='נדחה עד מועמד localization קפוא', reason='ניסוי העברה נפרד שהמשתמש ביקש; אינו חלק מתוצאות localization.'),
        dict(name='RAG / grounding', status='מחוץ למוקד הפעיל', reason='המיקוד הוא reasoning; משמרים היסטוריה בלי לפתוח מסלול חדש.'),
    ]
    ledger = dict(status='COMPLETE_REVIEWED_CONSOLIDATION' if complete else 'IN_PROGRESS_NOT_FINAL',
                  created_unix=time.time(), obligations=required, stage_boundary='RETURN_BEFORE_NEW_IMPROVEMENT_EXPERIMENT',
                  metrics=rows, families=families(required), backlog=backlog, advisor_history=advisor_history,
                  historical_registry_snapshot=registry, historical_registry_note='Discovery snapshot; individual source reviews determine completion, not stale liveness fields.',
                  contrast_sources=contrast_sources, paired_evidence=paired_evidence(contrast_sources), source_hashes=sources,
                  limitations=['All measured populations are exposed development data.',
                  'Rows repeated across reports are references, not independent experiments.',
                  'Legacy and corrected populations/access/metrics must not be subtracted as controlled effects.',
                  'Unknown method runtime is null, not zero.', 'No new improvement trial has been run by this builder.'])
    save(OUT/'LEDGER.json',ledger)
    buffer = io.StringIO(newline='')
    fields = [k for k in rows[0] if k != 'pb_cells']
    fields = list(dict.fromkeys(fields+['prm_unchanged']))
    writer = csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
    writer.writerows(rows); (OUT/'METRICS.csv').write_text(buffer.getvalue(),encoding='utf-8-sig')
    csv_rows=list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert len(csv_rows)==len(rows)
    for source,serialized in zip(rows,csv_rows):
        for field in ('prm_pooled_auc','prm_fold_mean_auc','prm_within_auc','pb_q4','pb_q8','pb_all'):
            assert (serialized[field]=='' if source[field] is None else float(serialized[field])==source[field])
    page = render(ledger); REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(page,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.links=[];self.tables=0
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='a': self.links.append(attrs['href'])
            if tag=='table': self.tables+=1
    parser=Links();parser.feed(page)
    for path in parser.links: assert (REPORT.parent/path).resolve().exists(),path
    assert len(rows)==len({(r['study'],r['method']) for r in rows})
    for relative,h in sources.items(): assert sha(ROOT/relative)==h, 'Source changed during build: '+relative
    for r in rows:
        assert 0 <= r['pb_valid'] <= r['pb_population']
        assert r['prm_valid'] is None or r['prm_population'] is None or r['prm_valid'] <= r['prm_population']
    save(OUT/'REVIEW.json', dict(status='PASS' if complete else 'DRAFT_CHECKS_PASS_NOT_FINAL',
         rows=len(rows), links=len(parser.links), tables=parser.tables,
         report_sha256=sha(REPORT), ledger_sha256=sha(OUT/'LEDGER.json'), csv_sha256=sha(OUT/'METRICS.csv'),
         checks=['required_reviews_before_finalization','unique_study_method_keys','links_exist',
                 'source_hashes_unchanged','coverage_denominators','partial_runs_excluded','csv_numeric_roundtrip'],
         external_review=False, browser_review=False))
    print(ledger['status'],len(rows),'evidence rows;',REPORT,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true')
    main(parser.parse_args().draft)
