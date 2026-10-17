"""Synthetic, read-only replay identity and fresh-destination guard tests."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("error_count_delivery_under_test", ROOT/"scripts/reproduce_label_free_error_count_v1.py")
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


class DeliveryReplayTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/"repository"
        reference = root/"results/reference"
        reference.mkdir(parents=True)
        source = root/"spectral_utils/error_count.py"
        source.parent.mkdir()
        source.write_text("# frozen synthetic source\n")
        code = {str(source.relative_to(root)): delivery.sha(source)}
        (reference/"CODE_FREEZE.json").write_text(json.dumps(code))
        amendments = reference/"AMENDMENTS.json"
        amendments.write_text("[]\n")
        freeze = {"python": sys.version, "executable": sys.executable,
                  "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")}}
        (reference/"INPUT_FREEZE.json").write_text(json.dumps(freeze))
        seal = {"code_sha256": code, "files": {name: {"sha256": delivery.sha(reference/name)}
                                               for name in ("AMENDMENTS.json", "INPUT_FREEZE.json")}}
        (reference/"PREDICTION_SEAL.json").write_text(json.dumps(seal))
        return root, reference, source

    def test_supported_reference_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, reference, _ = self.fixture(tmp)
            before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            delivery.validate_reference(root, reference)
            after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def run_main(self, root, reference, destination):
        validator = delivery.validate_reference
        with patch.object(delivery, "ROOT", root), patch.object(delivery, "REFERENCE", reference), \
                patch.object(delivery, "validate_reference", side_effect=lambda: validator(root, reference)), \
                patch.object(delivery.importlib.util, "spec_from_file_location", side_effect=AssertionError("Driver must not be imported")) as import_driver, \
                patch.object(sys, "argv", ["replay", "--destination", str(destination), "--predictions-only"]):
            with self.assertRaises(ValueError):
                delivery.main()
            import_driver.assert_not_called()

    def test_changed_helper_refuses_before_driver_import_or_destination_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, reference, source = self.fixture(tmp)
            source.write_text("# changed scientific implementation\n")
            destination = root/"results/fresh"
            self.run_main(root, reference, destination)
            self.assertFalse(destination.exists())

    def test_rehashed_code_freeze_cannot_override_prediction_seal_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, reference, source = self.fixture(tmp)
            source.write_text("# changed and rehashed scientific implementation\n")
            (reference/"CODE_FREEZE.json").write_text(json.dumps({str(source.relative_to(root)): delivery.sha(source)}))
            destination = root/"results/fresh"
            self.run_main(root, reference, destination)
            self.assertFalse(destination.exists())

    def test_changed_reference_metadata_refuses_without_destination_mutation(self):
        for name in ("AMENDMENTS.json", "INPUT_FREEZE.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, reference, _ = self.fixture(tmp)
                with (reference/name).open("a") as stream: stream.write(" \n")
                destination = root/"results/fresh"
                self.run_main(root, reference, destination)
                self.assertFalse(destination.exists())

    def test_recorded_runtime_drift_is_rejected(self):
        for field in ("python", "executable", "versions"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root, reference, _ = self.fixture(tmp)
                path = reference/"INPUT_FREEZE.json"
                freeze = json.loads(path.read_text())
                if field == "versions": freeze[field]["numpy"] = "different-fixture-version"
                else: freeze[field] += "-different-fixture-runtime"
                path.write_text(json.dumps(freeze))
                seal_path = reference/"PREDICTION_SEAL.json"
                seal = json.loads(seal_path.read_text())
                seal["files"]["INPUT_FREEZE.json"]["sha256"] = delivery.sha(path)
                seal_path.write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError, "runtime"):
                    delivery.validate_reference(root, reference)

    def test_outside_results_destination_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, reference, _ = self.fixture(tmp)
            destination = root/"outside-results"
            self.run_main(root, reference, destination)
            self.assertFalse(destination.exists())

    def test_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, reference, _ = self.fixture(tmp)
            destination = root/"results/existing"
            destination.mkdir()
            sentinel = destination/"sentinel.txt"
            sentinel.write_text("preserve this existing artifact\n")
            self.run_main(root, reference, destination)
            self.assertEqual(sentinel.read_text(), "preserve this existing artifact\n")
            self.assertEqual([p.name for p in destination.iterdir()], ["sentinel.txt"])


if __name__ == "__main__": unittest.main()
