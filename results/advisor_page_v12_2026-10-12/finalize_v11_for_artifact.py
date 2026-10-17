"""Prepare the patched v11 page for the Claude Artifact.

Usage: python3 finalize_v11_for_artifact.py <in_html> <out_html>

- Removes the "GitHub reading copy" banner and the local/unpublished notices.
- Points every relative evidence link to the v11 package on GitHub (commit 07cb7a4),
  because the Artifact does not carry the package files.
"""
import re, sys

BASE = ('https://github.com/omrisegev/hallucination_detection/blob/07cb7a45fb01ab62ade51fa5e59717c1f27fd8ff/'
        'results/advisor_completion_2026-10-11/')
NOTICE = ('<aside class="notice">Version 11, 12 October 2026. The analysis is Codex’s reviewed v11 package '
          '(branch <code>codex/advisor-v11-completion</code>, commit 07cb7a4). Claude added the takeaway bullets under each figure '
          'and the Qwen3-4B vs 8B section; the 4B files are in the project folder and not yet in Git. '
          'Evidence links open the package on GitHub. The 8 October presentation remains '
          '<a href="https://github.com/omrisegev/hallucination_detection/blob/8996379/docs/meetings/advisor_meeting_2026-10-08/findings_page/findings.html">in the repository</a>.</aside>')
FOOTER = ('<footer><b>Evidence state:</b> v11 analysis from saved CPU results (commit 07cb7a4) plus the Qwen3-4B re-score; '
          'no new GPU inference for this page. V10 source: 53ef62e3; research exports: 080877eb.</footer>')


def finalize(s):
    s, n = re.subn(r'<aside style="margin:20px;padding:16px;border:2px solid #c99a52;background:#fff7e8">.*?</aside>', '', s, count=1, flags=re.S)
    assert n == 1
    s, n = re.subn(r'<aside class="notice">This v11 page is a local rebuild\..*?</aside>', NOTICE, s, count=1, flags=re.S)
    assert n == 1
    s, n = re.subn(r'<footer>.*?</footer>', FOOTER, s, count=1, flags=re.S)
    assert n == 1

    def fix(m):
        h = m.group(1)
        if re.match(r'(https?:|mailto:|data:|#)', h):
            return m.group(0)
        return 'href="' + BASE + h + '" target="_blank" rel="noopener"'
    s = re.sub(r'href="([^"]*)"', fix, s)
    return s


if __name__ == '__main__':
    src, dst = sys.argv[1:3]
    s = open(src, encoding='utf-8').read()
    open(dst, 'w', encoding='utf-8').write(finalize(s))
