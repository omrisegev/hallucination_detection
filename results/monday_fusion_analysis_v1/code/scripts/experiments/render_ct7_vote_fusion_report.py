"""Presentation-only finalizer; never changes fits, metrics or their frozen code."""
import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.cvf_v2.data import digest, dump
from scripts.experiments.cvf_v2.ct7_evaluation import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args(); out = args.output.resolve()
    if not args.manifest_only:
        with open(out/'SUMMARY.csv', encoding='utf8') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ['sla', 'f1', 'within_auc', 'prmscore_inner', 'prmscore_q80']:
                row[key] = float(row[key]) if row[key] else None
        rows.sort(key=lambda r: 0 if r['method'] == 'ct7' else 1 if r['method'].startswith('fixed__all__')
                  else 2 if r['method'].startswith('fixed__errors__') else 3)
        contrasts = json.loads((out/'CT7_CONTRASTS.json').read_text(encoding='utf8'))
        health = json.loads((out/'VALIDATION.json').read_text(encoding='utf8'))
        from types import SimpleNamespace
        render(SimpleNamespace(out=out), rows, contrasts, health)
        path = out/'REPORT_HE.html'; text = path.read_text(encoding='utf8')
        text = re.sub(r'\[[-+]\d+\.\d+, [-+]\d+\.\d+\]',
                      lambda m: '<bdi dir="ltr">'+m.group(0)+'</bdi>', text)
        text = text.replace('<h2>בדיקות ואי־ודאות</h2>',
            '<h2>מה השתנה</h2><p>שימוש בבנק וב־readouts של CT7 משפר את הגישה ביחס לבנק הקודם: '
            'המיזוג הרך עם L-SML עולה מ־36.184% ל־39.996% SLA. אין להסיק מכך שמספר קטן יותר '
            'של פיצ׳רים הוא הסיבה — גם זהות הסיגנלים וצבירת הטוקנים שונות.</p>'
            '<p>מול CT7 המקורי, היתרון ב־SLA הוא 0.109 נקודת אחוז ורווח הסמך כולל אפס. '
            'ב־PRMBench ה־within-AUROC יורד מ־0.772397 ל־0.768632; רווח הסמך להפרש הוא '
            '<bdi dir="ltr">[-0.004880, -0.002680]</bdi>, מובהק גם לאחר Holm. '
            'PRMScore inner יורד מ־0.647104 ל־0.641766. לכן אין כאן עדות ליתרון על CT7 בשני המדדים.</p>'
            '<p>הבינריזציה ו־EM חלשים מהמיזוג הרך. אימון על שגויות בלבד משאיר את ה־SLA של '
            'L-SML הרך קרוב מאוד: 39.974% מול 39.996%. אין תוצאה חדשה שמצדיקה החלפה אוטומטית '
            'של CT7; ההכרעה לגבי ההמשך נשארת בידי החוקר.</p><h2>בדיקות ואי־ודאות</h2>')
        path.write_text(text, encoding='utf8')
    artifacts = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in out.rglob('*') if p.is_file() and p.name != 'REPORT_MANIFEST.json'
                 and 'browser_profile' not in p.parts}
    dump(out/'REPORT_MANIFEST.json', {'candidate_id': 'ct7-frozen-profiles-cumulative-vote-v1',
         'presentation_code': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': digest(__file__)},
         'artifacts': artifacts})


if __name__ == '__main__':
    main()
