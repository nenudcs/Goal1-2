"""Synthetic tensor checks; these do not establish clinical/competition accuracy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.detection.common import load_torch, require_container, save_torch
from src.detection.metrics import binary_metrics, fit_calibration, calibrated
from src.detection.models import AbnormalModel, _strict_load, pair_features, view_locations
from src.detection.pipeline import candidate_pairs, cap_pairs, duplicate_evaluation


def setUpModule():
    require_container()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward_features(self, x):
        return self.conv(x).mean((-2, -1))


class ModelTests(unittest.TestCase):
    def test_complete_views_streaming_pool_and_gradient(self):
        locations = list(view_locations((3, 13, 19), 8, 4))
        coverage = np.zeros((3, 13, 19), dtype=bool)
        for z, y, x in locations:
            if y is not None:
                coverage[z, y:y+8, x:x+8] = True
        self.assertTrue(coverage.all())
        cfg = {"project": {"amp": False}, "detection": {"abnormal": {
            "view_size": 8, "tile_stride": 4, "microbatch": 1, "bag_views": 5}}}
        model = AbnormalModel(TinyBackbone(), feature_dim=8, attention_dim=4).eval()
        volume = np.random.default_rng(42).uniform(0, 1, (3, 13, 19)).astype(np.float32)
        one = model.sequence_logits(volume, cfg, torch.device("cpu"))
        cfg["detection"]["abnormal"]["microbatch"] = 7
        many = model.sequence_logits(volume, cfg, torch.device("cpu"))
        torch.testing.assert_close(one, many, atol=1e-6, rtol=1e-5)
        model.train()
        model.sequence_logits(volume, cfg, torch.device("cpu"), training=True).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            save_torch(path, model.state_dict())
            other = AbnormalModel(TinyBackbone(), feature_dim=8, attention_dim=4).eval()
            other.load_state_dict(load_torch(path))
            model.eval()
            torch.testing.assert_close(model.sequence_logits(volume, cfg, torch.device("cpu")),
                                       other.sequence_logits(volume, cfg, torch.device("cpu")))
        with self.assertRaises(ValueError):
            _strict_load(model, {})

    def test_symmetric_pair_and_partial_sequence_candidate(self):
        features = {
            "A": {"case": F.normalize(torch.tensor([1., 1.]), dim=0), "series": torch.eye(2)},
            "B": {"case": torch.tensor([-1., 0.]), "series": torch.tensor([[1., 0.], [-1., 0.], [-1., 0.]])},
            "C": {"case": torch.tensor([0., -1.]), "series": torch.tensor([[0., -1.]])}}
        torch.testing.assert_close(pair_features(features["A"], features["B"]), pair_features(features["B"], features["A"]))
        cfg = {"detection": {"duplicate": {"candidate_k": 1, "reference_block": 1, "max_degree": 200}}}
        self.assertIn(("A", "B"), candidate_pairs(features, cfg, torch.device("cpu")))
        probabilities = {("center", f"id{i:03}"): .99-i*.001 for i in range(250)}
        self.assertEqual(len(cap_pairs(probabilities)), 200)
        metrics = duplicate_evaluation([{"a": "A", "b": "B", "label": 1}, {"a": "A", "b": "C", "label": 0}],
                                       {("A", "C"): 1.0}, cfg)
        self.assertEqual(metrics["candidate_recall"], 0)
        self.assertEqual(metrics["submitted_positive_recall"], 0)

    def test_metrics_ties_missing_classes_and_monotonic_calibration(self):
        self.assertEqual(binary_metrics([1, 0], [.5, .5])["AP"], .5)
        self.assertEqual(binary_metrics([1, 0], [.9, .1])["ROC_AUC"], 1.)
        with self.assertRaises(ValueError):
            binary_metrics([1], [.5])
        calibration = fit_calibration([0, 0, 1, 1], [-2., -1., 1., 2.])
        self.assertLess(calibrated(calibration, -1), calibrated(calibration, 1))


if __name__ == "__main__":
    unittest.main()
