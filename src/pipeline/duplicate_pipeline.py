import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.nifti import load_preprocessed_volume
from src.data.paths import list_case_series


@torch.no_grad()
def encode_case(case_dir, model, cfg, device):
    volumes = []
    for _, path in list_case_series(case_dir):
        volume, _ = load_preprocessed_volume(
            path, cfg["data"]["target_shape"],
            cfg["data"]["intensity_clip_percentiles"],
        )
        volumes.append(volume)
    if not volumes:
        return None
    return model.encode_case(
        torch.stack(volumes).to(device),
        cfg["duplicate"].get("series_batch_size", 4),
    ).squeeze(0).cpu()


@torch.no_grad()
def run_duplicate_pipeline(case_dirs, output_jsonl, model, cfg, device):
    ids, embeddings = [], []
    for case_dir in sorted(map(Path, case_dirs)):
        embedding = encode_case(case_dir, model, cfg, device)
        if embedding is not None:
            ids.append(case_dir.name)
            embeddings.append(embedding)
    if len(ids) < 2:
        raise ValueError("Duplicate inference requires at least two readable cases.")

    embeddings = F.normalize(torch.stack(embeddings), dim=1)
    topk = min(int(cfg["duplicate"].get("topk_candidates", 200)), len(ids) - 1)
    chunk_size = int(cfg["duplicate"].get("similarity_chunk_size", 256))
    candidates = {}
    scale = model.logit_scale.detach().cpu().exp().clamp(max=100.0)
    bias = model.bias.detach().cpu()
    for start in range(0, len(ids), chunk_size):
        similarities = embeddings[start:start + chunk_size] @ embeddings.T
        for local_index in range(similarities.shape[0]):
            index = start + local_index
            similarities[local_index, index] = -float("inf")
            values, indices = torch.topk(similarities[local_index], k=topk)
            for value, other in zip(values.tolist(), indices.tolist()):
                pair = tuple(sorted((ids[index], ids[other])))
                probability = float(torch.sigmoid(scale * value + bias))
                candidates[pair] = max(candidates.get(pair, 0.0), probability)

    degrees = {identifier: 0 for identifier in ids}
    selected = []
    for (a, b), probability in sorted(
        candidates.items(), key=lambda item: item[1], reverse=True
    ):
        if degrees[a] >= topk or degrees[b] >= topk:
            continue
        selected.append((a, b, probability))
        degrees[a] += 1
        degrees[b] += 1
    if not selected:
        raise RuntimeError("Duplicate inference did not produce any valid pair.")

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as stream:
        for a, b, probability in selected:
            stream.write(json.dumps({
                "StudyUID": a,
                "StudyUID_dup": b,
                "PairProb": probability,
            }, ensure_ascii=False) + "\n")
    return selected
