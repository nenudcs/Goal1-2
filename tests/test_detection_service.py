"""Cloud-only HTTP/queue/result tests with mocked inference, never external callbacks."""
import copy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from service.detection_server import JobRunner, create_app
from src.detection.common import atomic_json, read_json, require_container
from src.detection.submission import merge_results, validate_submission, write_predictions


def setUpModule():
    require_container()


def config(directory):
    return {"paths": {"jobs_dir": str(Path(directory)/"jobs"), "answer_root": str(Path(directory)/"answer")},
            "service": {"port": 8000, "callback_enabled": False, "callback_evaluation_id_type": "string",
                        "callback_max_attempts": 2, "callback_retry_seconds": .01}}


def wait_status(runner, request_id, wanted):
    deadline = time.monotonic()+4
    while time.monotonic() < deadline:
        if runner.snapshot(request_id)["status"] == wanted:
            return
        time.sleep(.01)
    raise AssertionError(runner.snapshot(request_id))


class ServiceTests(unittest.TestCase):
    def test_fast_response_health_idempotence_and_no_partial_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            gate = threading.Event()
            payload = {"request_id": "r1", "input": {"evaluation_id": "e1", "dataset_path": directory}}
            def inference(*args, **kwargs):
                gate.wait(timeout=3)
                return {"validated": True}
            app = create_app(cfg, model_loader=lambda _: {}, infer=inference)
            with patch("requests.post", side_effect=AssertionError("No callback allowed")):
                with TestClient(app) as client:
                    start = time.monotonic()
                    self.assertEqual(client.post("/call", json=payload).status_code, 200)
                    self.assertLess(time.monotonic()-start, 5)
                    self.assertEqual(client.get("/health").status_code, 200)
                    self.assertEqual(client.post("/call", json=payload).status_code, 200)
                    other = copy.deepcopy(payload)
                    other["input"]["evaluation_id"] = "changed"
                    self.assertEqual(client.post("/call", json=other).status_code, 409)
                    other["request_id"], other["input"]["evaluation_id"] = "r2", "e1"
                    self.assertEqual(client.post("/call", json=other).status_code, 409)
                    gate.set()
                    wait_status(app.state.runner, "r1", "awaiting_merge")

    def test_failed_inference_retry_and_callback_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            fail = [True]
            def inference(*args):
                if fail[0]:
                    raise ValueError("broken input")
                return {"validated": True}
            runner = JobRunner(cfg, inference)
            runner.start()
            payload = {"request_id": "r1", "input": {"evaluation_id": "e1", "dataset_path": directory}}
            try:
                runner.submit(payload)
                wait_status(runner, "r1", "failed")
                fail[0] = False
                runner.retry("r1")
                wait_status(runner, "r1", "awaiting_merge")
            finally:
                runner.close()
            path = Path(cfg["paths"]["jobs_dir"])/"r1.json"
            job = read_json(path)
            write_predictions(job["output"], {a: {"IsNotHumanBodyProb": .1, "IsStitchedProb": .2}
                                               for a in ("A", "B")}, [{"a": "A", "b": "B", "prob": .8}])
            job["report"] = {**validate_submission(job["output"], ["A", "B"]), "merged_other_goals": True}
            job.update(status="callback_pending", phase="callback", callback_attempts=0)
            atomic_json(path, job)
            cfg["service"].update(callback_enabled=True, callback_url="http://mock.invalid/callback/", merge_from_template="/teammate/{evaluation_id}")
            calls = []
            def post(*args, **kwargs):
                calls.append(kwargs["json"])
                return SimpleNamespace(status_code=500 if len(calls) == 1 else 200)
            recovered = JobRunner(cfg, lambda *a: self.fail("Must not repeat inference for callback recovery"), post=post)
            recovered.start()
            try:
                wait_status(recovered, "r1", "completed")
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[-1]["evaluationId"], "e1")
            finally:
                recovered.close()

    def test_transactional_output_merge_and_format_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = {a: {"IsNotHumanBodyProb": .1, "IsStitchedProb": .2} for a in ("A", "B")}
            pairs = [{"a": "A", "b": "B", "prob": .8}]
            write_predictions(root/"detection", predictions, pairs)
            self.assertTrue(validate_submission(root/"detection", ["A", "B"])["validated"])
            for accession in predictions:
                path = root/"teammate"/accession
                path.mkdir(parents=True)
                (path/"mask.nii.gz").write_bytes(b"test artifact copied verbatim")
                atomic_json(path/"prediction.json", {"AccessionNumber": accession, "Prediction": {"TumorProbability": .7},
                                                    "SegmentationMaskURI": {"core": "mask.nii.gz"}})
            merge_results(root/"detection", root/"teammate", root/"merged")
            merged = read_json(root/"merged"/"A"/"prediction.json")
            self.assertEqual(merged["Prediction"], {"TumorProbability": .7})
            self.assertEqual((root/"merged"/"A"/"mask.nii.gz").read_bytes(), (root/"teammate"/"A"/"mask.nii.gz").read_bytes())
            with self.assertRaises(FileExistsError):
                write_predictions(root/"detection", predictions, pairs)
            with self.assertRaises(ValueError):
                write_predictions(root/"invalid", predictions, [{"a": "A", "b": "A", "prob": .9}])
            self.assertFalse((root/"invalid").exists())
            predictions["A"]["IsStitchedProb"] = float("nan")
            with self.assertRaises(ValueError):
                write_predictions(root/"nan", predictions, pairs)


if __name__ == "__main__":
    unittest.main()
