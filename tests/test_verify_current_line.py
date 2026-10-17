import importlib.util
import json
import os
import shutil
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('vcl', os.path.join(ROOT, 'scripts', 'verify_current_line.py'))
vcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vcl)


class VerifyCurrentLine(unittest.TestCase):
    def test_registry_matches_frozen_files(self):
        report, _, failures = vcl.verify(ROOT)
        self.assertEqual(failures, [])
        self.assertEqual(len(report), 9)

    def test_detects_a_changed_number(self):
        # Positive control: a perturbed copy of one frozen CSV must fail.
        tmp = tempfile.mkdtemp()
        try:
            reg = json.load(open(os.path.join(ROOT, vcl.REGISTRY), encoding='utf-8'))
            e = reg['entries'][3]
            os.makedirs(os.path.join(tmp, 'configs'))
            reg['entries'] = [e]
            json.dump(reg, open(os.path.join(tmp, vcl.REGISTRY), 'w', encoding='utf-8'))
            dst = os.path.join(tmp, e['file'])
            os.makedirs(os.path.dirname(dst))
            shutil.copy(os.path.join(ROOT, e['file']), dst)
            text = open(dst, encoding='utf-8').read().replace(repr(e['expected'])[:8], '1' + repr(e['expected'])[1:8], 1)
            open(dst, 'w', encoding='utf-8').write(text)
            _, _, failures = vcl.verify(tmp)
            self.assertEqual(len(failures), 1)
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
