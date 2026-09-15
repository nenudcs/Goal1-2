"""Shared inference, bounded candidate retrieval and cached study encoding."""
import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .common import (branch_settings, cuda_device, file_hash, fingerprint,
                     load_torch, model_identity, read_json, save_torch, source_hash)
from .data import discover_cases, load_sequence
from .metrics import binary_metrics, calibrated, classification_metrics, logit
from .models import build_abnormal, build_duplicate_encoder, DuplicateHead, encode_study, pair_features

LOGGER = logging.getLogger("detection")


def records_by_study(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["accession"]].append(record)
    return dict(sorted(groups.items()))


def encode_records(records, encoder, cfg, device, identity):
    cache = Path(cfg["paths"]["cache_dir"]) / "dino"
    outputs = {}
    code = source_hash(Path(__file__).parent)
    for index, (accession, group) in enumerate(records_by_study(records).items(), 1):
        files = sorted({path for record in group for path in
                        ([record["path"]] if record["kind"] == "nifti" else record["files"])})
        key = fingerprint({"accession": accession, "files": {p: file_hash(p) for p in files},
                           "identity": identity, "implementation": code, "preprocess": "all-slices-gray224-v1"})
        path = cache / f"{key}.pth"
        if path.is_file():
            result = load_torch(path)
        else:
            result = encode_study(encoder, group, cfg, device)
            save_torch(path, result)
        if (set(result) != {"case", "series"} or result["case"].shape != (384,)
                or result["series"].shape != (len(group), 384)
                or any(not torch.isfinite(t).all() for t in result.values())):
            raise ValueError(f"Invalid feature cache for {accession}; remove this cache entry: {path}")
        for tensor in result.values():
            norms = tensor.norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                raise ValueError(f"Feature cache is not normalized: {path}")
        outputs[accession] = result
        LOGGER.info("Encoded study %d (%s)", index, accession)
    return outputs


@torch.no_grad()
def candidate_pairs(features, cfg, device):
    """Exact blockwise search; maximum sequence similarity preserves local matches.

    ponytail: exact search is quadratic in study count; replace with validated ANN
    only when measured cohort size makes this the bottleneck.
    """
    ids = sorted(features)
    if len(ids) < 2:
        return []
    settings = cfg["detection"]["duplicate"]
    k = min(int(settings["candidate_k"]), len(ids)-1)
    block = int(settings["reference_block"])
    if not 1 <= k <= 400 or block < 1:
        raise ValueError("candidate_k must be 1..400 and reference_block positive")
    reference_blocks = []
    for start in range(0, len(ids), block):
        references = ids[start:start+block]
        series = [features[a]["series"] for a in references]
        owners = torch.repeat_interleave(torch.arange(len(references)), torch.tensor([len(s) for s in series]))
        reference_blocks.append((references, torch.stack([features[a]["case"] for a in references]),
                                 torch.cat(series), owners))
    candidates = set()
    for accession in ids:
        query = features[accession]
        qcase = query["case"].to(device)
        qseries = query["series"].to(device)
        best = []
        for references, cases, series, owners in reference_blocks:
            case_scores = cases.to(device) @ qcase
            sequence_scores = torch.full_like(case_scores, -float("inf"))
            for offset in range(0, len(series), 1024):
                rb = series[offset:offset+1024].to(device)
                scores = torch.full((len(rb),), -float("inf"), device=device)
                for qa in qseries.split(128):
                    scores = torch.maximum(scores, (qa @ rb.T).amax(dim=0))
                sequence_scores.scatter_reduce_(0, owners[offset:offset+1024].to(device), scores,
                                                reduce="amax", include_self=True)
            combined = torch.maximum(case_scores, sequence_scores).cpu().tolist()
            for target, score in zip(references, combined):
                if target == accession:
                    continue
                best.append((score, target))
            best = sorted(best, key=lambda p: (-p[0], p[1]))[:k]
        candidates.update(tuple(sorted((accession, target))) for _, target in best)
    return sorted(candidates)


@torch.no_grad()
def score_pairs(pairs, features, head, cfg, device):
    batch_size = int(cfg["detection"]["duplicate"]["pair_batch_size"])
    if batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    head.eval()
    result = {}
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start+batch_size]
        values = torch.stack([pair_features(features[a], features[b]) for a, b in chunk]).to(device)
        logits = head(values).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite duplicate logits")
        result.update({pair: float(value) for pair, value in zip(chunk, logits.cpu().tolist())})
    return result


def cap_pairs(probabilities, max_degree=200):
    if not 1 <= max_degree <= 200:
        raise ValueError("max_degree must be 1..200")
    degrees = Counter()
    output = []
    for (a, b), score in sorted(probabilities.items(), key=lambda item: (-item[1], item[0])):
        if a == b:
            raise ValueError("Self-pair")
        if degrees[a] < max_degree and degrees[b] < max_degree:
            output.append({"a": a, "b": b, "prob": score})
            degrees[a] += 1
            degrees[b] += 1
    return output


def duplicate_evaluation(pairs, logits, cfg, calibration=None):
    probabilities = {pair: (calibrated(calibration, score) if calibration else
                            float(torch.sigmoid(torch.tensor(score)).item())) for pair, score in logits.items()}
    submitted = cap_pairs(probabilities, cfg["detection"]["duplicate"]["max_degree"])
    kept = {(p["a"], p["b"]): p["prob"] for p in submitted}
    labels = [p["label"] for p in pairs]
    scores = [kept.get(tuple(sorted((p["a"], p["b"]))), 0.0) for p in pairs]
    metrics = binary_metrics(labels, scores)
    positive = [tuple(sorted((p["a"], p["b"]))) for p in pairs if p["label"] == 1]
    metrics.update({"candidate_recall": sum(p in logits for p in positive)/len(positive),
                    "submitted_positive_recall": sum(p in kept for p in positive)/len(positive),
                    "candidate_count": len(logits), "submitted_count": len(kept),
                    "unlabeled_pairs": "excluded from metric truth; missing labeled candidates scored zero"})
    return metrics


@torch.no_grad()
def abnormal_scores(model, records, cfg, device, with_metrics=False):
    model.eval()
    predictions, truths = {}, {}
    sequence_y, sequence_p = [], []
    for accession, group in records_by_study(records).items():
        probabilities = []
        for record in group:
            volume, _ = load_sequence(record)
            logits = model.sequence_logits(volume, cfg, device, training=False)
            probability = logits.softmax(-1).cpu().tolist()
            probabilities.append(probability)
            if record.get("label") is not None:
                sequence_y.append(record["label"])
                sequence_p.append(max(range(3), key=lambda k: probability[k]))
        predictions[accession] = [max(p[k] for p in probabilities) for k in (1, 2)]
        if all(record.get("label") is not None for record in group):
            truths[accession] = [int(any(r["label"] == k for r in group)) for k in (1, 2)]
    if not with_metrics:
        return predictions
    ids = sorted(truths)
    case_metrics = {name: binary_metrics([truths[a][i] for a in ids], [predictions[a][i] for a in ids])
                    for i, name in enumerate(("fake", "compositing"))}
    return predictions, truths, {"sequence": classification_metrics(sequence_y, sequence_p),
                                "case": case_metrics, "excluded_partially_labeled_studies": len(predictions)-len(ids),
                                "selection_AP": sum(m["AP"] for m in case_metrics.values())/2}


def load_branch_checkpoint(cfg, branch):
    path = Path(cfg["paths"]["checkpoints_dir"]) / branch / "best.pth"
    state = load_torch(path)
    identities = model_identity(cfg, branch)
    if (state.get("format_version") != 1 or state.get("branch") != branch
            or state.get("classes") != ["true", "fake", "compositing"]
            or state.get("model_identity") != identities
            or state.get("settings") != branch_settings(cfg, branch)
            or state.get("implementation") != source_hash(Path(__file__).parent)
            or not state.get("validated")):
        raise ValueError(f"Incompatible or unvalidated {branch} checkpoint: {path}")
    calibration = read_json(path.parent / "calibration.json")
    if calibration.get("checkpoint_sha256") != file_hash(path) or calibration.get("manifest") != state.get("manifest"):
        raise ValueError(f"Calibration does not match checkpoint {path}")
    return state, calibration


def load_detection_models(cfg, device=None):
    device = device or cuda_device(cfg)
    if cfg["detection"]["data"]["negative_policy"] == "unknown":
        raise ValueError("Duplicate negative policy is unknown; formal inference is blocked")
    abnormal_state, abnormal_calibration = load_branch_checkpoint(cfg, "abnormal")
    duplicate_state, duplicate_calibration = load_branch_checkpoint(cfg, "duplicate")
    if abnormal_state["manifest"] != duplicate_state["manifest"]:
        raise ValueError("The two checkpoints use different data/split manifests")
    if duplicate_state.get("negative_policy") not in {"explicit", "closed_world"}:
        raise ValueError("Duplicate checkpoint lacks confirmed negative supervision")
    abnormal = build_abnormal(cfg, device, pretrained=False)
    abnormal.load_state_dict(abnormal_state["model"], strict=True)
    duplicate = DuplicateHead(cfg["detection"]["duplicate"]["feature_dim"]).to(device)
    duplicate.load_state_dict(duplicate_state["model"], strict=True)
    return {"abnormal": abnormal.eval(), "duplicate": duplicate.eval(),
            "encoder": build_duplicate_encoder(cfg, device), "device": device,
            "abnormal_calibration": abnormal_calibration, "duplicate_calibration": duplicate_calibration,
            "dino_identity": model_identity(cfg, "duplicate"),
            "checkpoint_hashes": {f"{branch}/{filename}": file_hash(Path(cfg["paths"]["checkpoints_dir"]) / branch / filename)
                                  for branch in ("abnormal", "duplicate") for filename in ("best.pth", "calibration.json")}}


def run_detection_batch(dataset_path, output_dir, models, cfg, merge_from=None):
    from .submission import write_predictions, validate_submission
    from src.common.logging_utils import CompetitionJSONLLogger
    logger = CompetitionJSONLLogger(cfg["paths"]["competition_log_dir"], "detection_inference.jsonl", strict=True)
    logger.write(epoch=None, step=0, phase="test", mode="inference", data_source=str(dataset_path),
                 checkpoint=cfg["paths"]["checkpoints_dir"])
    records = discover_cases(dataset_path, cfg)
    ids = sorted({r["accession"] for r in records})
    if len(ids) < 2:
        raise ValueError("Official pair file requires at least one valid non-self pair: input has fewer than two studies")
    source = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Input and output directories must not overlap")
    files = {p for r in records for p in ([r["path"]] if r["kind"] == "nifti" else r["files"])}
    provenance = {"input_files": fingerprint({p: file_hash(p) for p in sorted(files)}),
                  "checkpoints": models["checkpoint_hashes"], "ids": ids,
                  "merge_source": str(Path(merge_from).resolve()) if merge_from else None,
                  "merge_hash": fingerprint({str(p.relative_to(merge_from)): file_hash(p)
                                  for p in sorted(Path(merge_from).rglob("*")) if p.is_file()}) if merge_from else None}
    if output.exists():
        if read_json(output / "detection_manifest.json") != provenance:
            raise FileExistsError("Existing output does not match this input/model/merge source")
        report = validate_submission(output, ids)
        report["merged_other_goals"] = merge_from is not None
        return report
    raw = abnormal_scores(models["abnormal"], records, cfg, models["device"])
    predictions = {a: {"IsNotHumanBodyProb": calibrated(models["abnormal_calibration"]["fake"], logit(p[0])),
                       "IsStitchedProb": calibrated(models["abnormal_calibration"]["compositing"], logit(p[1]))}
                   for a, p in raw.items()}
    features = encode_records(records, models["encoder"], cfg, models["device"], models["dino_identity"])
    pairs = candidate_pairs(features, cfg, models["device"])
    logits = score_pairs(pairs, features, models["duplicate"], cfg, models["device"])
    probabilities = {p: calibrated(models["duplicate_calibration"]["pair"], s) for p, s in logits.items()}
    selected = cap_pairs(probabilities, cfg["detection"]["duplicate"]["max_degree"])
    write_predictions(output_dir, predictions, selected, merge_from=merge_from, provenance=provenance)
    report = validate_submission(output_dir, ids)
    report["merged_other_goals"] = merge_from is not None
    logger.write(
        epoch=None, step=None, phase="test", mode="inference", data_source=str(dataset_path),
        checkpoint=cfg["paths"]["checkpoints_dir"])
    return report
