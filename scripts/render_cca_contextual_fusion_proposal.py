"""Render the review proposal as a standalone Hebrew HTML document."""
from pathlib import Path
import html
import re

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/experiments/CCA_CONTEXTUAL_FUSION_PROPOSAL_20260915.md'

def inline(text):
    escaped=html.escape(text)
    escaped=re.sub(r'`([^`]+)`',r'<code dir="ltr">\1</code>',escaped)
    escaped=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',escaped)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',escaped)

def render(text):
    blocks=[];contents=[];heading=0
    for block in text.strip().split('\n\n'):
        lines=block.splitlines()
        match=re.match(r'^(#{1,3}) (.+)$',lines[0])
        if match:
            level=len(match[1]);heading+=1;anchor=f'section-{heading}'
            blocks.append(f'<h{level} id="{anchor}">{inline(match[2])}</h{level}>')
            if level==2:contents.append(f'<a href="#{anchor}">{inline(match[2])}</a>')
        elif lines[0].startswith('|'):
            rows=[]
            for i,line in enumerate(lines):
                if i==1:continue
                tag='th' if i==0 else 'td'
                rows.append('<tr>'+''.join(f'<{tag}>{inline(c.strip())}</{tag}>' for c in line.strip('|').split('|'))+'</tr>')
            blocks.append('<div class="table-wrap"><table>'+''.join(rows)+'</table></div>')
        elif all(line.startswith('- ') for line in lines):
            blocks.append('<ul>'+''.join('<li>'+inline(line[2:])+'</li>' for line in lines)+'</ul>')
        elif len(lines)==1 and lines[0].startswith('`') and lines[0].endswith('`'):
            blocks.append('<div class="equation" dir="ltr">'+html.escape(lines[0][1:-1])+'</div>')
        else:blocks.append('<p>'+inline(' '.join(lines))+'</p>')
    return '\n'.join(blocks),'\n'.join(contents)

def main():
    body,contents=render(SOURCE.read_text(encoding='utf8'))
    stages=['היסטוריית הפיצ׳רים','פרופיל מיקום ונרמול לאימון','CCA: ייצוג הקשר','שכנים מקבוצות מקור אחרות','covariance עם shrinkage','משקלי simplex לפי רגעי IU','ציונים מקוריים × משקלים','Top10 לכל זרם → ציון לצעד → gate ופסגה']
    flow='<div class="flow" aria-label="שלבי האלגוריתם המוצע">'+''.join(f'<div><span>{i+1}</span>{html.escape(s)}</div>' for i,s in enumerate(stages))+'</div>'
    page='''<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>הצעה לביקורת — CCA ו־fusion מותנה בהקשר</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:#1d3142;font:17px/1.8 system-ui,Arial,sans-serif}
main{max-width:1120px;margin:28px auto;padding:36px;background:white;border-radius:14px}h1{font-size:30px;line-height:1.4}h2{font-size:24px;margin-top:42px;border-bottom:2px solid #dce7ec;padding-bottom:8px}h3{font-size:20px;margin-top:30px}p{max-width:100ch}
a{color:#09687b;text-underline-offset:3px}code,.equation{font:15px/1.6 ui-monospace,Consolas,monospace;unicode-bidi:isolate}code{background:#edf3f6;padding:2px 4px;border-radius:4px}.equation{background:#edf3f6;padding:15px;overflow:auto;text-align:left;border-radius:7px}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:right;vertical-align:top;border-bottom:1px solid #dce7ec;padding:10px}th{background:#edf3f6}.table-wrap{overflow:auto}li{margin-bottom:9px}
.status{background:#fff5d9;border-right:5px solid #b58b23;padding:14px 18px;border-radius:5px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.flow div{background:#eaf5f6;padding:12px;border-radius:8px;line-height:1.5;font-size:15px}.flow span{display:block;font-weight:bold;color:#08798b}
nav{display:grid;grid-template-columns:1fr 1fr;gap:6px 20px;background:#f4f7fa;padding:18px;border-radius:8px;font-size:15px}summary{cursor:pointer;font-weight:600}h1,h2,h3{scroll-margin-top:20px}
@media(max-width:750px){main{margin:0;padding:20px}.flow{grid-template-columns:1fr 1fr}nav{grid-template-columns:1fr}h1{font-size:25px}}@media print{body{background:white}main{margin:0;max-width:none;padding:0}nav,details{display:none}h2,h3{break-after:avoid}tr{break-inside:avoid}.flow{grid-template-columns:repeat(4,1fr)}}
</style></head><body><main><div class="status"><strong>הצעה לביקורת לפני הרצה.</strong> אין במסמך תוצאות של CCA-fusion. ערכי ברירת המחדל מוצעים להקפאה לאחר הדיון. <a href="CCA_CONTEXTUAL_FUSION_PROPOSAL_20260915.md">מקור Markdown להעברה ל־Claude</a></div>
'''+flow+'<details><summary>ניווט במסמך</summary><nav>'+contents+'</nav></details>'+body+'</main></body></html>\n'
    SOURCE.with_suffix('.html').write_text(page,encoding='utf8',newline='\n')
    print(SOURCE.with_suffix('.html'))

if __name__=='__main__':main()
