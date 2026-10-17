"""Final artifact audit plus canonical checks where digit evidence varies."""
from pathlib import Path
import sys,json,ast
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from scripts.run_digit_fusion import OUT,write,sha


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,t,a):
        if t=='a':self.links += [v for k,v in a if k=='href']
        if t=='tr':self.row=[]
        if t=='td':self.cell=''
    def handle_data(self,s):
        if self.cell is not None:self.cell+=s
    def handle_endtag(self,t):
        if t=='td':self.row.append(self.cell);self.cell=None
        if t=='tr' and self.row:self.rows.append(self.row);self.row=None


def run():
    ds=json.loads((OUT/'DIAGNOSTICS.json').read_text());live=[d for d in ds if not d['digit_constant']]
    checks=[]
    for j in np.linspace(0,len(live)-1,30,dtype=int):
        d=live[j];b=d['banks'][1];C=np.array(b['covariance'])
        fit=upcr_fit_covariance(C,**IU_FIT_DEFAULTS);w=fit.w
        if w@C@np.ones(len(w))<0:w=-w
        np.testing.assert_allclose(w,np.asarray(b['weights'])[b['live']],atol=2e-7,rtol=2e-7)
        checks.append(dict(uid=d['uid'],weights=w.tolist()))
    W=np.array([d['banks'][1]['weights'] for d in live])
    write(OUT/'IU_DIAGNOSTIC_AUDIT.json',dict(status='PASS',canonical_variable_digit_answers=len(checks),
        digit_negative_fraction=float(np.mean(W[:,-1]<0)),digit_weight_quantiles=np.quantile(W[:,-1],[0,.1,.5,.9,1]).tolist(),
        checks=checks,code_sha256=sha(Path(__file__))))
    report=ReportParser();report.feed((OUT/'REPORT.html').read_text(encoding='utf8'))
    for link in report.links:assert (OUT/link).is_file(),link
    metrics=json.loads((OUT/'METRICS.json').read_text())['metrics'];rows=report.rows[:16]
    assert len(rows)==16
    for row in rows:
        assert len(row)==4 and any(abs(float(row[1].rstrip('%'))-100*m['pb_all8'])<.000051
            and abs(float(row[2])-m['prm_within'])<.00000051
            and abs(float(row[3])-m['prmscore_q08'])<.00000051 for m in metrics.values()),row
    paths=['spectral_utils/digit_fusion.py','tests/test_digit_fusion.py',
           'scripts/run_digit_fusion.py','scripts/evaluate_digit_fusion.py',
           'scripts/report_digit_fusion.py','scripts/update_digit_fusion_docs.py','scripts/audit_digit_fusion.py']
    for name in paths:ast.parse((ROOT/name).read_text(encoding='utf8'))
    audit=json.loads((OUT/'AUDIT.json').read_text());assert audit['status']=='PASS'
    assert audit['answers']==13769 and audit['steps']==145597 and len(audit['independent_metrics'])==30
    assert sha(OUT/'SCORES_FROZEN.npz')==audit['scores_sha256'] and audit['flow_state_unchanged']
    write(OUT/'FINAL_REVIEW.json',dict(status='PASS',python_files=len(paths),new_unit_tests=5,
        checked_numeric_report_rows=len(rows),checked_report_links=len(report.links),independent_metric_bundles=30,
        extra_canonical_variable_digit_checks=30,code_sha256=sha(Path(__file__))))
    write(OUT/'REPORT_AUDIT.json',dict(status='PASS',report_sha256=sha(OUT/'REPORT.html'),
        displayed_rows=len(rows),valid_local_links=len(report.links)))
    write(OUT/'ARTIFACTS.json',dict(large_arrays_and_extraction_retained_locally=True,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file() and p.name!='ARTIFACTS.json'}))
    print('Final review PASS;30 canonical variable-digit checks;30 metric bundles;16 report rows.')


if __name__=='__main__':run()
