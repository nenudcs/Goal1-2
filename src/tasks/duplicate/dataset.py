import random

import torch
from torch.utils.data import Dataset

from src.data.nifti import load_preprocessed_volume
from src.data.paths import PathResolver, read_manifest


def read_positive_pairs(manifest_path):
    frame = read_manifest(manifest_path)
    required = {"src_img", "desc_img"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Duplicate manifest is missing columns: {sorted(missing)}")
    pairs = set()
    for a, b in zip(frame["src_img"], frame["desc_img"]):
        a, b = str(a).strip(), str(b).strip()
        if a and b and a != b:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def build_negative_pairs(accessions, positive_pairs, negative_ratio=3, seed=42):
    accessions = sorted(set(map(str, accessions)))
    positive_set = set(positive_pairs)
    possible = len(accessions) * (len(accessions) - 1) // 2 - len(positive_set)
    needed = min(possible, len(positive_set) * int(negative_ratio))
    if needed <= 0:
        return []
    rng = random.Random(seed)
    negatives = set()
    for _ in range(max(100, needed * 20)):
        if len(negatives) >= needed:
            break
        pair = tuple(sorted(rng.sample(accessions, 2)))
        if pair not in positive_set:
            negatives.add(pair)
    if len(negatives) < needed:
        for index, a in enumerate(accessions):
            for b in accessions[index + 1:]:
                pair = (a, b)
                if pair not in positive_set:
                    negatives.add(pair)
                if len(negatives) >= needed:
                    break
            if len(negatives) >= needed:
                break
    return sorted(negatives)


class DuplicatePairDataset(Dataset):
    def __init__(
        self, positive_manifest, annotation_root, target_shape, clip_percentiles,
        negative_ratio=3, seed=42, accessions=None, source_dirs=None,
    ):
        self.resolver = PathResolver(annotation_root, source_dirs=source_dirs)
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles
        available = {
            identifier for identifier in self.resolver.duplicate_accessions()
            if self.resolver.list_case_series(
                self.resolver.duplicate_accession_dir(identifier)
            )
        }
        allowed = available if accessions is None else available & set(map(str, accessions))
        positives = [
            pair for pair in read_positive_pairs(positive_manifest)
            if pair[0] in allowed and pair[1] in allowed
        ]
        negatives = build_negative_pairs(allowed, positives, negative_ratio, seed)
        self.items = [(a, b, 1.0) for a, b in positives]
        self.items += [(a, b, 0.0) for a, b in negatives]
        if not self.items:
            raise ValueError("Duplicate dataset contains no usable pairs.")

    def __len__(self):
        return len(self.items)

    def _load_case(self, accession):
        paths = self.resolver.list_case_series(
            self.resolver.duplicate_accession_dir(accession)
        )
        if not paths:
            raise RuntimeError(f"No valid NIfTI series in duplicate/{accession}")
        volumes = [
            load_preprocessed_volume(path, self.target_shape, self.clip_percentiles)[0]
            for path in paths
        ]
        return torch.stack(volumes)

    def __getitem__(self, idx):
        a, b, label = self.items[idx]
        return self._load_case(a), self._load_case(b), torch.tensor(label), a, b


def duplicate_collate(batch):
    xa, xb, labels, a, b = zip(*batch)
    return list(xa), list(xb), torch.stack(labels), list(a), list(b)
