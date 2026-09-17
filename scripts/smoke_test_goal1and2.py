from pathlib import Path
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.train_utils import load_checkpoint, save_checkpoint
from src.models.medicalnet3d import MedicalNetClassifier
from src.data.split import split_duplicate_components
from src.pipeline.goal1and2_pipeline import run_goal_pipeline
from src.tasks.abnormal.dataset import AbnormalDataset
from src.tasks.duplicate.dataset import DuplicatePairDataset, duplicate_collate
from src.tasks.duplicate.model import DuplicateModel


def write_case(root, accession, series_uid, seed):
    series_dir = Path(root) / accession / series_uid
    series_dir.mkdir(parents=True, exist_ok=True)
    array = np.random.default_rng(seed).normal(size=(18, 20, 16)).astype(np.float32)
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        series_dir / f"{series_uid}.nii.gz",
    )


def build_config(root, quality_checkpoint, duplicate_checkpoint):
    return {
        "project": {"device": "cpu", "amp": False, "num_workers": 0, "pin_memory": False},
        "data": {
            "target_shape": [16, 32, 32],
            "intensity_clip_percentiles": [0.5, 99.5],
            "case_aggregation": "max",
            "source_dirs": {
                "true": "", "fake": "fake", "nonhuman": "fake",
                "composition": "Composition", "duplicate": "duplicate",
            },
        },
        "abnormal": {
            "class_names": ["true", "fake", "composition"],
            "checkpoint_path": str(quality_checkpoint),
            "label_aliases": {
                "true": "true", "fake": "fake", "nonhuman": "fake",
                "composition": "composition", "compositing": "composition",
            },
        },
        "duplicate": {
            "embedding_dim": 32, "negative_ratio": 2,
            "topk_candidates": 2, "similarity_chunk_size": 2,
            "series_batch_size": 4, "checkpoint_path": str(duplicate_checkpoint),
        },
        "paths": {"annotation_root": str(root)},
    }


def main():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="goal1and2_smoke_") as temp:
        temp = Path(temp)
        annotation = temp / "annotation"
        labels_dir = temp / "labels"
        labels_dir.mkdir()
        abnormal_rows = []
        sources = {"true": annotation, "fake": annotation / "fake", "compositing": annotation / "Composition"}
        for label_index, (label, source) in enumerate(sources.items()):
            for case_index in range(2):
                accession = f"{label[:1]}{case_index}"
                series_uid = f"s{label_index}{case_index}"
                write_case(source, accession, series_uid, 10 * label_index + case_index)
                abnormal_rows.append({
                    "AccessionNumber": accession, "SeriesUid": series_uid, "Label": label,
                })
        abnormal_manifest = labels_dir / "1_abnormal.csv"
        pd.DataFrame(abnormal_rows).to_csv(abnormal_manifest, index=False)

        duplicate_root = annotation / "duplicate"
        duplicate_ids = [f"d{index}" for index in range(6)]
        for index, accession in enumerate(duplicate_ids):
            write_case(duplicate_root, accession, "s0", 100 + index)
        duplicate_manifest = labels_dir / "2_duplicate.csv"
        pd.DataFrame([
            {"src_img": "d0", "desc_img": "d1"},
            {"src_img": "d2", "desc_img": "d3"},
        ]).to_csv(duplicate_manifest, index=False)
        train_ids, val_ids = split_duplicate_components(
            duplicate_ids, [("d0", "d1"), ("d2", "d3")], 0.34, 7
        )
        assert train_ids.isdisjoint(val_ids)
        for a, b in (("d0", "d1"), ("d2", "d3")):
            assert (a in train_ids and b in train_ids) or (a in val_ids and b in val_ids)

        quality_checkpoint = temp / "quality.pth"
        duplicate_checkpoint = temp / "duplicate.pth"
        cfg = build_config(annotation, quality_checkpoint, duplicate_checkpoint)
        abnormal_ds = AbnormalDataset(
            abnormal_manifest, annotation,
            [row["AccessionNumber"] for row in abnormal_rows],
            cfg["data"]["target_shape"], cfg["data"]["intensity_clip_percentiles"],
            cfg["abnormal"]["label_aliases"], cfg["data"]["source_dirs"],
        )
        volumes, labels, _, _ = next(iter(DataLoader(abnormal_ds, batch_size=2)))
        quality_model = MedicalNetClassifier(3)
        quality_optimizer = torch.optim.Adam(quality_model.parameters(), lr=1e-4)
        quality_logits = quality_model(volumes)
        quality_loss = torch.nn.functional.cross_entropy(quality_logits, labels)
        quality_loss.backward()
        quality_optimizer.step()
        quality_model.eval()
        expected_quality = quality_model(volumes).detach()
        save_checkpoint(quality_model, quality_optimizer, 1, quality_checkpoint, {
            "model_name": "MedicalNet-ResNet10-3class",
            "class_names": ["true", "fake", "composition"],
            "target_shape": cfg["data"]["target_shape"],
            "aggregation": "max",
        })
        reloaded_quality = MedicalNetClassifier(3)
        load_checkpoint(reloaded_quality, quality_checkpoint, "cpu", strict=True)
        reloaded_quality.eval()
        assert torch.allclose(expected_quality, reloaded_quality(volumes), atol=1e-6)

        duplicate_ds = DuplicatePairDataset(
            duplicate_manifest, annotation, cfg["data"]["target_shape"],
            cfg["data"]["intensity_clip_percentiles"], 2, 7, duplicate_ids,
            cfg["data"]["source_dirs"],
        )
        xa, xb, pair_labels, _, _ = next(iter(DataLoader(
            duplicate_ds, batch_size=2, collate_fn=duplicate_collate,
        )))
        duplicate_model = DuplicateModel(32)
        duplicate_optimizer = torch.optim.Adam(duplicate_model.parameters(), lr=1e-4)
        pair_logits = torch.cat([
            duplicate_model.similarity(a, b, 4) for a, b in zip(xa, xb)
        ])
        duplicate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pair_logits, pair_labels
        )
        duplicate_loss.backward()
        duplicate_optimizer.step()
        duplicate_model.eval()
        expected_pair = duplicate_model.similarity(xa[0], xb[0], 4).detach()
        save_checkpoint(duplicate_model, duplicate_optimizer, 1, duplicate_checkpoint, {
            "model_name": "Siamese-MedicalNet-ResNet10",
            "embedding_dim": 32,
            "target_shape": cfg["data"]["target_shape"],
            "case_pooling": "mean_then_l2",
        })
        reloaded_duplicate = DuplicateModel(32)
        load_checkpoint(reloaded_duplicate, duplicate_checkpoint, "cpu", strict=True)
        reloaded_duplicate.eval()
        assert torch.allclose(
            expected_pair, reloaded_duplicate.similarity(xa[0], xb[0], 4), atol=1e-6
        )

        test_input = temp / "test"
        for index in range(3):
            write_case(test_input, f"test{index}", "series0", 200 + index)
        goal1_output, goal2_output = temp / "goal1_out", temp / "goal2_out"
        run_goal_pipeline("goal1", test_input, goal1_output, cfg, torch.device("cpu"))
        run_goal_pipeline("goal2", test_input, goal2_output, cfg, torch.device("cpu"))
        for output in (goal1_output, goal2_output):
            for index in range(3):
                record = json.loads(
                    (output / f"test{index}" / "prediction.json").read_text(encoding="utf-8")
                )
                for key in ("IsNotHumanBodyProb", "IsStitchedProb"):
                    assert 0.0 <= record[key] <= 1.0
        lines = [
            json.loads(line) for line in
            (goal2_output / "duplicate_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        seen, degree = set(), {}
        for record in lines:
            a, b = record["StudyUID"], record["StudyUID_dup"]
            pair = tuple(sorted((a, b)))
            assert a != b and pair not in seen and 0.0 <= record["PairProb"] <= 1.0
            seen.add(pair)
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        assert lines and max(degree.values()) <= cfg["duplicate"]["topk_candidates"]
        print("SMOKE TEST PASSED")
        print(f"quality_loss={quality_loss.item():.6f}")
        print(f"duplicate_loss={duplicate_loss.item():.6f}")
        print(f"prediction_cases=3 duplicate_pairs={len(lines)}")


if __name__ == "__main__":
    main()
