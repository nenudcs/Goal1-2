"""Validated, non-destructive goal 1/2 outputs and teammate-result merging."""
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .common import atomic_json, read_json

FIELDS = ("IsNotHumanBodyProb", "IsStitchedProb")


def identifier(value):
    if (not isinstance(value, str) or value != value.strip() or value in {".", ".."}
            or not re.fullmatch(r"[^/\\:\x00-\x1f]{1,200}", value)):
        raise ValueError("Identifier must be safe, nonempty text without path separators")
    return value


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected a finite probability in [0,1], got {value!r}")
    return float(value)


def validate_submission(output_dir, expected_ids, require_pair=True):
    root = Path(output_dir)
    expected = {identifier(a) for a in expected_ids}
    if not expected:
        raise ValueError("No expected studies")
    actual = {p.parent.name for p in root.glob("*/prediction.json")}
    if actual != expected:
        raise ValueError(f"Prediction accession mismatch: missing={expected-actual}, extra={actual-expected}")
    for accession in sorted(expected):
        obj = read_json(root / accession / "prediction.json")
        if obj.get("AccessionNumber") != accession:
            raise ValueError(f"AccessionNumber mismatch for {accession}")
        for key in FIELDS:
            probability(obj[key])
    degrees, seen = Counter(), set()
    with (root / "duplicate_pairs.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line {number}")
            pair = json.loads(line)
            if set(pair) != {"StudyUID", "StudyUID_dup", "PairProb"}:
                raise ValueError(f"Unexpected pair fields on line {number}")
            a, b = identifier(pair["StudyUID"]), identifier(pair["StudyUID_dup"])
            if a == b or a not in expected or b not in expected:
                raise ValueError(f"Invalid pair {a}, {b}")
            key = tuple(sorted((a, b)))
            if key in seen:
                raise ValueError(f"Repeated pair {key}")
            seen.add(key)
            probability(pair["PairProb"])
            degrees.update((a, b))
    if require_pair and not seen:
        raise ValueError("Official submission requires at least one valid duplicate-pair row")
    if any(v > 200 for v in degrees.values()):
        raise ValueError("Final undirected pair degree exceeds 200")
    return {"studies": len(expected), "study_ids": sorted(expected), "pairs": len(seen), "max_degree": max(degrees.values(), default=0),
            "validated": True, "scope": "goals_1_2_only"}


def _validate_relative_uris(value, case_dir, source_root, uri=False):
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_relative_uris(item, case_dir, source_root, uri or key.lower().endswith("uri"))
    elif isinstance(value, list):
        for item in value:
            _validate_relative_uris(item, case_dir, source_root, uri)
    elif uri and isinstance(value, str) and value:
        if Path(value).is_absolute() or "://" in value or "\\" in value:
            raise ValueError("Teammate artifact URIs must be relative local paths before merging")
        choices = [(case_dir / value).resolve(), (source_root / value).resolve()]
        if not any(p.is_relative_to(source_root) and p.is_file() for p in choices):
            raise ValueError(f"Missing or escaping teammate artifact URI: {value}")


def write_predictions(output_dir, predictions, pairs, merge_from=None, provenance=None):
    output = Path(output_dir).resolve()
    ids = {identifier(a) for a in predictions}
    if output.exists():
        raise FileExistsError(f"Use a new output directory; existing results are never overwritten: {output}")
    source = Path(merge_from).resolve() if merge_from else None
    if source:
        if not source.is_dir() or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Merge source and output must be distinct, non-overlapping directories")
        actual = {p.parent.name for p in source.glob("*/prediction.json")}
        if actual != ids:
            raise ValueError("Teammate accession set does not exactly match goal 1/2 results")
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Symlinks are not accepted in teammate output trees")
        for accession in ids:
            obj = read_json(source / accession / "prediction.json")
            if obj.get("AccessionNumber") != accession:
                raise ValueError("Teammate AccessionNumber does not match directory")
            _validate_relative_uris(obj, source / accession, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".detection-stage-", dir=output.parent))
    try:
        if source:
            shutil.copytree(source, stage, dirs_exist_ok=True)
        for accession, result in predictions.items():
            path = stage / accession / "prediction.json"
            obj = read_json(path) if path.is_file() else {"AccessionNumber": accession}
            for key in FIELDS:
                obj[key] = probability(result[key])
            atomic_json(path, obj)
        with (stage / "duplicate_pairs.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for pair in pairs:
                row = {"StudyUID": identifier(pair["a"]), "StudyUID_dup": identifier(pair["b"]),
                       "PairProb": probability(pair["prob"])}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        validate_submission(stage, ids)
        if provenance is not None:
            atomic_json(stage / "detection_manifest.json", provenance)
        os.rename(stage, output)
    finally:
        # This directory is created above, never computed from a request identifier.
        if stage.exists() and stage.resolve().parent == output.parent and stage.name.startswith(".detection-stage-"):
            shutil.rmtree(stage)


def merge_results(detection_dir, teammate_dir, output_dir):
    root = Path(detection_dir)
    predictions = {p.parent.name: read_json(p) for p in root.glob("*/prediction.json")}
    validate_submission(root, predictions)
    pairs = []
    with (root / "duplicate_pairs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            p = json.loads(line)
            pairs.append({"a": p["StudyUID"], "b": p["StudyUID_dup"], "prob": p["PairProb"]})
    write_predictions(output_dir, predictions, pairs, merge_from=teammate_dir)
    report = validate_submission(output_dir, predictions)
    report["merged_other_goals"] = True
    return report
