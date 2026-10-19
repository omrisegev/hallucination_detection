"""Fill template.html with data.json and missing.html -> ablation5.html (run build_data.py first)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
t = (H / 'template.html').read_text(encoding='utf8')
t = t.replace('__DATA__', (H / 'data.json').read_text())
t = t.replace('__MISSING__', (H / 'missing.html').read_text(encoding='utf8'))
(H / 'ablation5.html').write_text(t, encoding='utf8')
print('wrote', H / 'ablation5.html', len(t))
