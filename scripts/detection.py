"""Container CLI: python -m scripts.detection --config ... <command>."""
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path

from src.common.config import load_config
from src.detection.common import atomic_json, cuda_device, load_torch, require_container, save_torch


def _doctor_impl(cfg, metadata_only=False):
    from src.detection.common import model_identity
    report = {"status": "running", "metadata_only": metadata_only, "gpu_verified": False, "versions": {}, "paths": {}}
    atomic_json(Path(cfg["paths"]["output_dir"]) / "doctor.json", report)
    for package in ("torch", "torchvision", "timm", "numpy", "nibabel", "pydicom", "openpyxl", "scikit-learn", "fastapi", "pydantic"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
    for name, value in cfg["detection"]["models"].items():
        path = Path(value)
        report["paths"][name] = {"path": str(path), "exists": path.is_dir() if name.endswith("repo") else path.is_file()}
    if not all(p["exists"] for p in report["paths"].values()):
        atomic_json(Path(cfg["paths"]["output_dir"]) / "doctor.json", report)
        raise FileNotFoundError("Missing model source/weight paths; see doctor.json. No downloads attempted.")
    if not metadata_only:
        import numpy as np
        import torch
        from src.detection.models import amp_context, build_abnormal, build_duplicate_encoder, DuplicateHead
        device = cuda_device(cfg)
        report.update(device=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                      capability=list(torch.cuda.get_device_capability(device)), torch_arches=torch.cuda.get_arch_list())
        model = build_abnormal(cfg, device)
        model.train()
        model.set_backbone_trainable(True)
        rng = np.random.default_rng(42)
        volume = rng.uniform(0, 1, (3, 224, 224)).astype(np.float32)
        logits = model.sequence_logits(volume, cfg, device, training=True)
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([0], device=device))
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise ValueError("ConvNeXt full-backbone gradient check failed")
        model.eval()
        before = model.sequence_logits(volume, cfg, device)
        path = Path(cfg["paths"]["output_dir"]) / "doctor_roundtrip.pth"
        save_torch(path, model.state_dict())
        model.load_state_dict(load_torch(path), strict=True)
        torch.testing.assert_close(before, model.sequence_logits(volume, cfg, device))
        path.unlink()
        del model
        encoder = build_duplicate_encoder(cfg, device)
        with torch.no_grad(), amp_context(cfg, device):
            features = encoder(torch.randn(2, 3, 224, 224, device=device))
        if features.shape != (2, 384) or not torch.isfinite(features).all():
            raise ValueError("DINOv2 feature check failed")
        head = DuplicateHead().to(device)
        pair_input = torch.randn(2, 772, device=device)
        head(pair_input).sum().backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
            raise ValueError("Pair-head backward check failed")
        head.eval()
        before = head(pair_input).detach()
        save_torch(path, head.state_dict())
        head.load_state_dict(load_torch(path), strict=True)
        torch.testing.assert_close(before, head(pair_input))
        path.unlink()
        torch.cuda.synchronize()
        report.update(gpu_verified=True, models_verified=["ConvNeXt-Tiny", "DINOv2-ViT-S14", "DuplicateHead"],
                      model_identity=model_identity(cfg))
    report["status"] = "metadata_checked" if metadata_only else "passed"
    atomic_json(Path(cfg["paths"]["output_dir"]) / "doctor.json", report)
    return report


def doctor(cfg, metadata_only=False):
    try:
        return _doctor_impl(cfg, metadata_only)
    except Exception as exc:
        from src.detection.common import read_json
        path = Path(cfg["paths"]["output_dir"]) / "doctor.json"
        report = read_json(path) if path.is_file() else {}
        report.update(status="failed", gpu_verified=False, error=f"{type(exc).__name__}: {exc}")
        atomic_json(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_detection.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor")
    check.add_argument("--metadata-only", action="store_true")
    commands.add_parser("index")
    for name in ("train-abnormal", "train-duplicate"):
        train = commands.add_parser(name)
        train.add_argument("--continue-training", action="store_true", help="Reset early-stop counter while restoring model/optimizer")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--branch", choices=("all", "abnormal", "duplicate"), default="all")
    infer = commands.add_parser("infer")
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--merge-from")
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--input", help="Optional unlabeled image root; otherwise use saved manifest")
    retrieve.add_argument("--output", required=True, help="Diagnostic JSON path, NOT a competition submission")
    merge = commands.add_parser("merge")
    merge.add_argument("--detection", required=True)
    merge.add_argument("--teammate", required=True)
    merge.add_argument("--output", required=True)
    validate = commands.add_parser("validate-output")
    validate.add_argument("--output", required=True)
    validate.add_argument("--input", required=True, help="Test image root for accession completeness")
    commands.add_parser("serve")
    args = parser.parse_args()
    require_container()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.command == "doctor":
        result = doctor(cfg, args.metadata_only)
    elif args.command == "index":
        from src.detection.data import build_manifest
        manifest = build_manifest(cfg)
        atomic_json(cfg["paths"]["manifest"], manifest)
        result = {k: manifest[k] for k in ("coverage", "warnings", "duplicate_ready", "fingerprint")}
        atomic_json(Path(cfg["paths"]["output_dir"]) / "data_summary.json", result)
    elif args.command in {"train-abnormal", "train-duplicate", "evaluate"}:
        from src.detection.training import train_abnormal, train_duplicate, evaluate as run_evaluate
        if args.command == "evaluate":
            result = run_evaluate(cfg, args.branch)
        else:
            trainer = train_abnormal if args.command == "train-abnormal" else train_duplicate
            trainer(cfg, continue_training=args.continue_training)
            result = {"status": "trained_and_calibrated", "branch": args.command}
    elif args.command == "infer":
        from src.detection.pipeline import load_detection_models, run_detection_batch
        result = run_detection_batch(args.input, args.output, load_detection_models(cfg), cfg, args.merge_from)
    elif args.command == "retrieve":
        from src.detection.common import model_identity
        from src.detection.data import discover_cases
        from src.detection.models import build_duplicate_encoder
        from src.detection.pipeline import candidate_pairs, encode_records
        from src.detection.training import load_manifest
        records = discover_cases(args.input, cfg) if args.input else load_manifest(cfg)["records"]
        device = cuda_device(cfg)
        features = encode_records(records, build_duplicate_encoder(cfg, device), cfg, device, model_identity(cfg, "duplicate"))
        pairs = candidate_pairs(features, cfg, device)
        result = {"diagnostic_only": True, "calibrated": False, "candidate_pairs": [list(p) for p in pairs],
                  "notice": "Candidate IDs only; this is not duplicate_pairs.jsonl and cannot be submitted"}
        atomic_json(args.output, result)
        result = {"candidate_count": len(pairs), "output": args.output, "diagnostic_only": True}
    elif args.command == "merge":
        from src.detection.submission import merge_results
        result = merge_results(args.detection, args.teammate, args.output)
    elif args.command == "validate-output":
        from src.detection.data import discover_cases
        from src.detection.submission import validate_submission
        ids = {r["accession"] for r in discover_cases(args.input, cfg)}
        result = validate_submission(args.output, ids)
    else:
        from service.detection_server import main as serve
        serve(args.config)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
