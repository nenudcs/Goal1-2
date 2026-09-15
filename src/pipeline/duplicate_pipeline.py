from pathlib import Path
import json
import torch
import torch.nn.functional as F

from src.pipeline.case_pipeline import list_series
from src.data.nifti import load_preprocessed_volume, check_nifti

@torch.no_grad()
def encode_case(case_dir, model, cfg, device):
    shape = cfg["data"]["target_shape"]
    clip = cfg["data"]["intensity_clip_percentiles"]
    vols = []
    for _, p in list_series(case_dir):
        if not check_nifti(p):
            continue
        try:
            x, _ = load_preprocessed_volume(p, shape, clip)
            vols.append(x)
        except Exception as e:
            print(f"[DuplicatePipeline][NIFTI SKIP] {p} | {type(e).__name__}: {e}")
    if not vols:
        return None
    x = torch.stack(vols).to(device)
    return model.encode_case(x).squeeze(0).cpu()

@torch.no_grad()
def run_duplicate_pipeline(dataset_path, output_jsonl, model, cfg, device):
    """
    V1 为清晰起见使用全量 cosine，相当于 baseline。
    提交前数据规模很大时可换 FAISS/ANN。
    每例最终仅保留 Top-K。
    """
    dataset_path = Path(dataset_path)
    case_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]

    ids, embs = [], []
    for cdir in sorted(case_dirs):
        emb = encode_case(cdir, model, cfg, device)
        if emb is not None:
            ids.append(cdir.name)
            embs.append(emb)

    if len(embs) < 2:
        Path(output_jsonl).write_text("", encoding="utf-8")
        return

    E = F.normalize(torch.stack(embs), dim=1)
    sim = E @ E.T
    sim.fill_diagonal_(-1)

    topk = min(cfg["duplicate"]["topk_candidates"], len(ids)-1)
    seen = {}

    for i, a in enumerate(ids):
        vals, inds = torch.topk(sim[i], k=topk)
        for v, j in zip(vals.tolist(), inds.tolist()):
            b = ids[j]
            key = tuple(sorted((a, b)))
            prob = float((v + 1.0) / 2.0)  # baseline: cosine [-1,1] -> [0,1]
            seen[key] = max(seen.get(key, 0.0), prob)

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for (a, b), prob in sorted(seen.items(), key=lambda x: x[1], reverse=True):
            f.write(json.dumps({
                "StudyUID": a,
                "StudyUID_dup": b,
                "PairProb": prob
            }, ensure_ascii=False) + "\n")
