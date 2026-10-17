"""CPU regressions for the integration's review corrections."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayBoundary(unittest.TestCase):
    def setUp(self):
        self.module = load('integration_replay', 'results/remote_claims_replay_2026-10-15/replay.py')

    def test_missing_inputs_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)/'output'
            with patch.object(self.module, 'required_inputs', return_value=[Path(d)/'missing']):
                with self.assertRaises(FileNotFoundError):
                    self.module.preflight(dest)
            self.assertFalse(dest.exists())

    def test_existing_destination_and_preserved_package_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            sentinel = Path(d)/'sentinel'
            sentinel.write_text('unchanged')
            with self.assertRaises(ValueError):
                self.module.preflight(Path(d))
            self.assertEqual(sentinel.read_text(), 'unchanged')
        with self.assertRaises(ValueError):
            self.module.preflight(self.module.HERE/'new-output')

    def test_reported_csv_uses_repository_and_metric_helpers_are_unchanged(self):
        self.assertTrue(self.module.REPORTED.is_relative_to(ROOT))
        self.assertTrue(self.module.REPORTED.is_file())
        old = ROOT/'docs/reviews/integration_completion_20261017/sources/remote_claims_replay_20261015_original.py'
        new = ROOT/'results/remote_claims_replay_2026-10-15/replay.py'
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(old), functions(new)
        for name in ['answer_z', 'q80', 'auc', 'metric', 'measurements', 'orient', 'equal_cluster_score']:
            self.assertEqual(a[name], b[name], name)


class ReadingVerification(unittest.TestCase):
    def setUp(self):
        self.module = load('integration_delivery', 'scripts/verify_integration_reading_delivery.py')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root/'results'/'package'
        self.package.mkdir(parents=True)
        (self.package/'data.csv').write_bytes(b'frozen\n')
        (self.root/'source.py').write_bytes(b'x = 1\n')
        (self.root/'PROGRESS.md').write_bytes(b'new integration status\n')
        sha = lambda b: hashlib.sha256(b).hexdigest()
        manifest = {'files': [{'path': 'data.csv', 'sha256': sha(b'frozen\n')}],
                    'runtime_files': [{'path': 'source.py', 'sha256': sha(b'x = 1\n')},
                                      {'path': 'PROGRESS.md', 'sha256': sha(b'old status\n')}]}
        (self.package/'GITHUB_DELIVERY_MANIFEST.json').write_text(json.dumps(manifest))

    def test_document_drift_uses_pinned_reference(self):
        with patch.object(self.module.subprocess, 'check_output', return_value=b'old status\n'):
            result = self.module.verify(self.package, root=self.root)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['reference_documents'][0]['current_differs'])
        self.assertFalse(result['raw_arrays_verified'])

    def test_numerical_source_and_artifact_drift_fail(self):
        (self.root/'source.py').write_bytes(b'x = 2\n')
        with self.assertRaisesRegex(ValueError, 'Numerical/runtime'):
            self.module.verify(self.package, root=self.root)
        (self.root/'source.py').write_bytes(b'x = 1\n')
        (self.package/'data.csv').write_bytes(b'changed\n')
        with self.assertRaisesRegex(ValueError, 'Reading artifact'):
            self.module.verify(self.package, root=self.root)

    def test_wrong_reference_and_traversal_fail(self):
        with patch.object(self.module.subprocess, 'check_output', return_value=b'wrong revision\n'):
            with self.assertRaisesRegex(ValueError, 'Publication reference'):
                self.module.verify(self.package, root=self.root)
        with self.assertRaises(ValueError):
            self.module.checked_path(self.package, '../../source.py')


class Handoff(unittest.TestCase):
    def test_pinned_track_a_contains_dependencies_and_adapter_names_are_valid(self):
        text = (ROOT/'docs/experiments/PRMBENCH_STEP_VOTERS_HANDOFF.md').read_text()
        self.assertIn('65bcf88310d7ee98a2b4b314bd0602e7db064b9c', text)
        self.assertIn('rlhflow:rlhflow_prm pure:pure_prm mathshepherd:math_shepherd', text)
        for alias in ['pure', 'rlhflow', 'mathshepherd']:
            self.assertNotRegex(text, r'--adapter '+alias+r'(?=[`\s])')
        self.assertTrue((ROOT/'spectral_utils/error_count.py').is_file())


if __name__ == '__main__':
    unittest.main()
