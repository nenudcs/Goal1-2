"""Sequence supervision and confirmed-pair training, with validated resumable checkpoints."""
import logging
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from src.common.logging_utils import CompetitionJSONLLogger
from .common import (atomic_json, branch_settings, cuda_device, experiment_identity, file_hash,
                     fingerprint, load_torch, manifest_identity, model_identity, read_json,
                     restore_rng, rng_state, save_torch, seed_all, source_hash)
from .data import load_sequence
from .metrics import binary_metrics, calibrated, fit_calibration, logit
from .models import build_abnormal, build_duplicate_encoder, DuplicateHead, pair_features
from .pipeline import (abnormal_scores, candidate_pairs, duplicate_evaluation,
                       encode_records, records_by_study, score_pairs)

LOGGER = logging.getLogger("detection")


def load_manifest(cfg):
    manifest = read_json(cfg["paths"]["manifest"])
    if manifest["fingerprint"] != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("Manifest content was changed: rebuild index")
    if manifest["negative_policy"] != cfg["detection"]["data"]["negative_policy"]:
        raise ValueError("Negative policy changed: rebuild index before training")
    configured = manifest.get("data_settings")
    if configured is not None and configured != cfg["detection"]["data"]:
        raise ValueError("Data configuration changed: rebuild index")
    for path, expected in manifest["source_tables"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Label/split file changed: rebuild index ({path})")
    for record in manifest["records"]:
        for path, expected in record["content_sha256"].items():
            if file_hash(path) != expected:
                raise ValueError(f"Image changed: rebuild index ({path})")
    return manifest


def split_records(manifest, split):
    return [r for r in manifest["records"] if manifest["splits"][r["accession"]] == split]


def split_pairs(manifest, split):
    return [p for p in manifest["pairs"] if manifest["splits"][p["a"]] == split
            and manifest["splits"][p["b"]] == split]


def check_class_coverage(manifest):
    for split in ("train", "val", "calibration"):
        records = split_records(manifest, split)
        labels = {r["label"] for r in records if r["label"] is not None}
        if labels != {0, 1, 2}:
            raise ValueError(f"{split} lacks three-class supervision: {labels}")
        complete = [group for group in records_by_study(records).values() if all(r["label"] is not None for r in group)]
        for k in (1, 2):
            truth = {int(any(r["label"] == k for r in group)) for group in complete}
            if truth != {0, 1}:
                raise ValueError(f"{split} lacks fully labeled positive/negative studies for class {k}")


def _checkpoint(model, optimizer, cfg, manifest, branch, identities, signature,
                epoch, step, best_score, bad_epochs, metrics):
    return {"format_version": 1, "branch": branch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
            "best_score": best_score, "bad_epochs": bad_epochs, "metrics": metrics,
            "rng": rng_state(), "signature": signature, "validated": True,
            "classes": ["true", "fake", "compositing"], "settings": branch_settings(cfg, branch),
            "model_identity": identities, "manifest": manifest_identity(manifest),
            "label_manifest": manifest["fingerprint"], "negative_policy": manifest["negative_policy"],
            "implementation": source_hash(Path(__file__).parent)}


def _resume(model, optimizer, directory, signature, continue_training):
    path = directory / "last.pth"
    if not path.exists():
        return 0, 0, -float("inf"), 0
    state = load_torch(path)
    if state.get("signature") != signature:
        raise ValueError(f"Incompatible data/model/settings on resume: use a new experiment directory ({path})")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    if not (directory / "best.pth").is_file():
        raise ValueError("Resume checkpoint exists but validated best.pth is missing")
    return state["epoch"], state["step"], state["best_score"], 0 if continue_training else state["bad_epochs"]


def _can_train(epoch, bad_epochs, settings):
    maximum = settings.get("max_epochs")
    return bad_epochs < int(settings["patience"]) and (maximum is None or epoch < int(maximum))


def train_abnormal(cfg, continue_training=False):
    device = cuda_device(cfg)
    manifest = load_manifest(cfg)
    check_class_coverage(manifest)
    seed_all(cfg["project"]["seed"])
    identities = model_identity(cfg, "abnormal")
    signature = experiment_identity(cfg, manifest, "abnormal", identities)
    model = build_abnormal(cfg, device)
    settings = cfg["detection"]["abnormal"]
    backbone = list(model.backbone.parameters())
    heads = list(model.attention.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": settings["backbone_lr"]},
                                   {"params": heads, "lr": settings["head_lr"]}], weight_decay=settings["weight_decay"])
    directory = Path(cfg["paths"]["checkpoints_dir"]) / "abnormal"
    epoch, step, best, bad = _resume(model, optimizer, directory, signature, continue_training)
    records = [r for r in split_records(manifest, "train") if r["label"] is not None]
    counts = Counter(r["label"] for r in records)
    weights = torch.tensor([len(records)/(3*counts[k]) for k in range(3)], device=device)
    logger = CompetitionJSONLLogger(cfg["paths"]["competition_log_dir"], "detection_abnormal.jsonl", strict=True)
    logger.write(max(1, epoch), step, "train", "training", "official/train",
                 pretrained_from=cfg["detection"]["models"]["convnext_weights"])
    while _can_train(epoch, bad, settings):
        epoch += 1
        model.train()
        model.set_backbone_trainable(epoch > int(settings["freeze_epochs"]))
        order = list(records)
        random.shuffle(order)
        for record in order:
            volume, _ = load_sequence(record)
            optimizer.zero_grad(set_to_none=True)
            logits = model.sequence_logits(volume, cfg, device, training=True)
            # Manual class weighting: CE's default weighted mean cancels weights for batch size 1.
            loss = F.cross_entropy(logits[None], torch.tensor([record["label"]], device=device)) * weights[record["label"]]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite abnormal training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                logger.write(epoch, step, "train", "training", "official/train", float(loss.item()),
                             settings["head_lr"], pretrained_from=cfg["detection"]["models"]["convnext_weights"])
        _, _, metrics = abnormal_scores(model, split_records(manifest, "val"), cfg, device, with_metrics=True)
        score = metrics["selection_AP"]
        improved = score > best
        best, bad = max(best, score), 0 if improved else bad+1
        state = _checkpoint(model, optimizer, cfg, manifest, "abnormal", identities, signature,
                            epoch, step, best, bad, metrics)
        if improved:
            save_torch(directory / "best.pth", state)
        save_torch(directory / "last.pth", state)
        logger.write(epoch, step, "val", "training", "internal/validation", checkpoint=str(directory / "last.pth"))
        atomic_json(directory / "validation.json", metrics)
        LOGGER.info("Abnormal epoch=%d mean case AP=%.6f patience=%d/%d", epoch, score, bad, settings["patience"])
    calibrate_abnormal(cfg, manifest, model, device)


def calibrate_abnormal(cfg, manifest, model, device):
    directory = Path(cfg["paths"]["checkpoints_dir"]) / "abnormal"
    best = load_torch(directory / "best.pth")
    model.load_state_dict(best["model"], strict=True)
    predictions, truths, _ = abnormal_scores(model, split_records(manifest, "calibration"), cfg, device, with_metrics=True)
    ids = sorted(truths)
    result = {name: fit_calibration([truths[a][i] for a in ids], [logit(predictions[a][i]) for a in ids])
              for i, name in enumerate(("fake", "compositing"))}
    result.update(checkpoint_sha256=file_hash(directory / "best.pth"), manifest=best["manifest"],
                  split="calibration", score_input="logit of max sequence class probability")
    atomic_json(directory / "calibration.json", result)


def train_duplicate(cfg, continue_training=False):
    device = cuda_device(cfg)
    manifest = load_manifest(cfg)
    if not manifest["duplicate_ready"]:
        raise ValueError("Duplicate supervision not ready: confirm negative policy and positive/negative coverage in all splits")
    seed_all(cfg["project"]["seed"])
    identities = model_identity(cfg, "duplicate")
    signature = experiment_identity(cfg, manifest, "duplicate", identities)
    encoder = build_duplicate_encoder(cfg, device)
    features = encode_records(manifest["records"], encoder, cfg, device, identities)
    del encoder
    head = DuplicateHead(cfg["detection"]["duplicate"]["feature_dim"]).to(device)
    settings = cfg["detection"]["duplicate"]
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    directory = Path(cfg["paths"]["checkpoints_dir"]) / "duplicate"
    epoch, step, best, bad = _resume(head, optimizer, directory, signature, continue_training)
    positives = [p for p in split_pairs(manifest, "train") if p["label"] == 1]
    negatives = [p for p in split_pairs(manifest, "train") if p["label"] == 0]
    val_features = {a: f for a, f in features.items() if manifest["splits"][a] == "val"}
    val_candidates = candidate_pairs(val_features, cfg, device)
    batch_size = int(settings["pair_batch_size"])
    logger = CompetitionJSONLLogger(cfg["paths"]["competition_log_dir"], "detection_duplicate.jsonl", strict=True)
    logger.write(max(1, epoch), step, "train", "training", "official/confirmed_pairs",
                 pretrained_from=cfg["detection"]["models"]["dino_weights"])
    while _can_train(epoch, bad, settings):
        epoch += 1
        count = min(len(negatives), max(1, int(len(positives)*settings["negative_ratio"])))
        # Mine exclusively within the annotated negative pool in the training split.
        negative_scores = score_pairs([(p["a"], p["b"]) for p in negatives], features, head, cfg, device)
        ranked = sorted(negatives, key=lambda p: (-negative_scores[(p["a"], p["b"])], p["a"], p["b"]))
        hard_count = count//2
        samples = positives + ranked[:hard_count] + random.sample(ranked[hard_count:], count-hard_count)
        random.shuffle(samples)
        head.train()
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start+batch_size]
            inputs = torch.stack([pair_features(features[p["a"]], features[p["b"]]) for p in chunk]).to(device)
            labels = torch.tensor([p["label"] for p in chunk], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(inputs), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite duplicate loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            logger.write(epoch, step, "train", "training", "official/confirmed_pairs", float(loss.item()),
                         settings["lr"], pretrained_from=cfg["detection"]["models"]["dino_weights"])
        logits = score_pairs(val_candidates, val_features, head, cfg, device)
        metrics = duplicate_evaluation(split_pairs(manifest, "val"), logits, cfg)
        score = metrics["AP"]
        improved = score > best
        best, bad = max(best, score), 0 if improved else bad+1
        state = _checkpoint(head, optimizer, cfg, manifest, "duplicate", identities, signature,
                            epoch, step, best, bad, metrics)
        if improved:
            save_torch(directory / "best.pth", state)
        save_torch(directory / "last.pth", state)
        logger.write(epoch, step, "val", "training", "internal/validation", checkpoint=str(directory / "last.pth"))
        atomic_json(directory / "validation.json", metrics)
        LOGGER.info("Duplicate epoch=%d submitted AP=%.6f patience=%d/%d", epoch, score, bad, settings["patience"])
    best_state = load_torch(directory / "best.pth")
    head.load_state_dict(best_state["model"], strict=True)
    pairs = split_pairs(manifest, "calibration")
    scores = score_pairs([(p["a"], p["b"]) for p in pairs], features, head, cfg, device)
    calibration = fit_calibration([p["label"] for p in pairs], [scores[(p["a"], p["b"])] for p in pairs])
    atomic_json(directory / "calibration.json", {"pair": calibration, "split": "calibration",
                "manifest": best_state["manifest"], "checkpoint_sha256": file_hash(directory / "best.pth")})


def evaluate(cfg, branch="all"):
    from .pipeline import load_branch_checkpoint
    device = cuda_device(cfg)
    logger = CompetitionJSONLLogger(cfg["paths"]["competition_log_dir"], "detection_evaluation.jsonl", strict=True)
    logger.write(None, 0, "val", "inference", "internal/validation", checkpoint=cfg["paths"]["checkpoints_dir"])
    manifest = load_manifest(cfg)
    result = {"definition": "Internal validation used for selection, not an independent official test score"}
    if branch in {"all", "abnormal"}:
        state, calibration = load_branch_checkpoint(cfg, "abnormal")
        if state["manifest"] != manifest_identity(manifest):
            raise ValueError("Evaluation manifest differs from checkpoint")
        model = build_abnormal(cfg, device, pretrained=False)
        model.load_state_dict(state["model"], strict=True)
        predictions, truths, metrics = abnormal_scores(model, split_records(manifest, "val"), cfg, device, True)
        ids = sorted(truths)
        metrics["calibrated_case"] = {name: binary_metrics([truths[a][i] for a in ids],
            [calibrated(calibration[name], logit(predictions[a][i])) for a in ids])
            for i, name in enumerate(("fake", "compositing"))}
        result["abnormal"] = metrics
        del model
    if branch in {"all", "duplicate"}:
        if not manifest["duplicate_ready"]:
            raise ValueError("Duplicate labels are not ready for evaluation")
        state, calibration = load_branch_checkpoint(cfg, "duplicate")
        if state["signature"] != experiment_identity(cfg, manifest, "duplicate", model_identity(cfg, "duplicate")):
            raise ValueError("Duplicate evaluation data, pair labels or split differ from checkpoint")
        encoder = build_duplicate_encoder(cfg, device)
        features = encode_records(split_records(manifest, "val"), encoder, cfg, device, model_identity(cfg, "duplicate"))
        head = DuplicateHead(cfg["detection"]["duplicate"]["feature_dim"]).to(device)
        head.load_state_dict(state["model"], strict=True)
        pairs = candidate_pairs(features, cfg, device)
        scores = score_pairs(pairs, features, head, cfg, device)
        result["duplicate"] = duplicate_evaluation(split_pairs(manifest, "val"), scores, cfg, calibration["pair"])
    atomic_json(Path(cfg["paths"]["output_dir"]) / f"evaluation_{branch}.json", result)
    logger.write(None, None, "val", "inference", "internal/validation", checkpoint=cfg["paths"]["checkpoints_dir"])
    return result
