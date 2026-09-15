"""Cloud-only flow checks with tiny backbones; calibration quality is tested separately."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.nn import functional as F

from src.common.config import load_config
from src.detection.common import load_torch, read_json, require_container
from src.detection.models import AbnormalModel
from src.detection.training import evaluate, train_abnormal, train_duplicate
from src.detection.pipeline import load_detection_models, run_detection_batch


def setUpModule():
    require_container()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 8, 1)

    def forward_features(self, x):
        return self.layer(x).mean((-2, -1))


class TrainingTests(unittest.TestCase):
    def test_train_resume_calibrate_both_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(Path(__file__).resolve().parents[1]/"configs"/"config_detection.yaml")
            cfg["project"]["amp"] = False
            cfg["paths"]["checkpoints_dir"] = str(Path(directory)/"checkpoints")
            cfg["paths"]["output_dir"] = str(Path(directory)/"outputs")
            cfg["paths"]["competition_log_dir"] = str(Path(directory)/"logs")
            cfg["detection"]["abnormal"].update(view_size=8, tile_stride=4, bag_views=3, microbatch=2, max_epochs=1, freeze_epochs=0)
            cfg["detection"]["duplicate"].update(max_epochs=1, pair_batch_size=2)
            cfg["detection"]["data"]["negative_policy"] = "explicit"
            records, splits, pairs, features = [], {}, [], {}
            volume = np.random.default_rng(42).uniform(0, 1, (3, 8, 8)).astype(np.float32)
            torch.manual_seed(42)
            for split in ("train", "val", "calibration"):
                ids = []
                for label in range(3):
                    accession = f"{split}{label}"
                    ids.append(accession)
                    image = Path(directory)/"images"/split/accession/"S"/"S.nii.gz"
                    image.parent.mkdir(parents=True)
                    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), np.eye(4)), image)
                    records.append({"accession": accession, "series_uid": "S", "label": label,
                                    "content_sha256": {}, "kind": "nifti", "path": str(image)})
                    splits[accession] = split
                    vector = F.normalize(torch.randn(384), dim=0)
                    features[accession] = {"case": vector, "series": vector[None]}
                pairs.extend([{"a": ids[0], "b": ids[1], "label": 1}, {"a": ids[0], "b": ids[2], "label": 0}])
            manifest = {"records": records, "splits": splits, "pairs": pairs, "fingerprint": "synthetic",
                        "negative_policy": "explicit", "duplicate_ready": True}
            def make_model(*args, **kwargs):
                return AbnormalModel(TinyBackbone(), feature_dim=8, attention_dim=4)
            with ExitStack() as stack:
                stack.enter_context(patch("src.detection.training.cuda_device", return_value=torch.device("cpu")))
                stack.enter_context(patch("src.detection.training.load_manifest", return_value=manifest))
                stack.enter_context(patch("src.detection.training.model_identity", return_value={"test": "tiny"}))
                stack.enter_context(patch("src.detection.training.build_abnormal", side_effect=make_model))
                stack.enter_context(patch("src.detection.training.build_duplicate_encoder", return_value=nn.Identity()))
                stack.enter_context(patch("src.detection.training.encode_records",
                    side_effect=lambda records, *args: {r["accession"]: features[r["accession"]] for r in records}))
                stack.enter_context(patch("src.detection.training.load_sequence", return_value=(volume, {})))
                stack.enter_context(patch("src.detection.pipeline.load_sequence", return_value=(volume, {})))
                # This flow test verifies binding/persistence; random tiny features are not calibration evidence.
                stack.enter_context(patch("src.detection.training.fit_calibration", return_value={"coefficient": 1., "intercept": 0.}))
                train_abnormal(cfg)
                train_duplicate(cfg)
                for branch in ("abnormal", "duplicate"):
                    path = Path(cfg["paths"]["checkpoints_dir"])/branch
                    state = load_torch(path/"last.pth")
                    self.assertEqual(state["epoch"], 1)
                    self.assertTrue(state["validated"])
                    self.assertEqual(read_json(path/"calibration.json")["manifest"], state["manifest"])
                    cfg["detection"][branch]["max_epochs"] = 2
                train_abnormal(cfg, continue_training=True)
                train_duplicate(cfg, continue_training=True)
                for branch in ("abnormal", "duplicate"):
                    state = load_torch(Path(cfg["paths"]["checkpoints_dir"])/branch/"last.pth")
                    self.assertEqual(state["epoch"], 2)
                stack.enter_context(patch("src.detection.pipeline.cuda_device", return_value=torch.device("cpu")))
                stack.enter_context(patch("src.detection.pipeline.model_identity", return_value={"test": "tiny"}))
                self.assertEqual(set(evaluate(cfg)), {"definition", "abnormal", "duplicate"})
                manifest["pairs"][0]["label"] = 0
                with self.assertRaisesRegex(ValueError, "pair labels"):
                    evaluate(cfg, "duplicate")
                manifest["pairs"][0]["label"] = 1
                stack.enter_context(patch("src.detection.pipeline.build_abnormal", side_effect=make_model))
                stack.enter_context(patch("src.detection.pipeline.build_duplicate_encoder", return_value=nn.Identity()))
                stack.enter_context(patch("src.detection.pipeline.encode_records",
                    side_effect=lambda records, *args: {r["accession"]: features[r["accession"]] for r in records}))
                models = load_detection_models(cfg)
                output = Path(directory)/"submission"
                report = run_detection_batch(Path(directory)/"images"/"val", output, models, cfg)
                self.assertTrue(report["validated"])
                self.assertEqual(set(report["study_ids"]), {"val0", "val1", "val2"})
                self.assertEqual(report, run_detection_batch(Path(directory)/"images"/"val", output, models, cfg))
                models["checkpoint_hashes"]["abnormal/calibration.json"] = "changed"
                with self.assertRaises(FileExistsError):
                    run_detection_batch(Path(directory)/"images"/"val", output, models, cfg)
                manifest["duplicate_ready"] = False
                with self.assertRaisesRegex(ValueError, "not ready"):
                    train_duplicate(cfg)


if __name__ == "__main__":
    unittest.main()
