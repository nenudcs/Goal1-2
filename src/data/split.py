from pathlib import Path
import json
import math
from sklearn.model_selection import train_test_split

def split_accessions(accessions, val_ratio=0.1, random_state=42, labels=None):
    ids = sorted(set(map(str, accessions)))
    if len(ids) < 2:
        return set(ids), set()
    stratify = None
    if labels is not None:
        # labels: mapping accession -> class label
        vals = [labels.get(i) for i in ids]
        if all(v is not None for v in vals):
            counts = {v: vals.count(v) for v in set(vals)}
            val_count = max(1, math.ceil(len(ids) * float(val_ratio)))
            if (
                min(counts.values()) >= 2
                and val_count >= len(counts)
                and len(ids) - val_count >= len(counts)
            ):
                stratify = vals
    train_ids, val_ids = train_test_split(
        ids, test_size=val_ratio, random_state=random_state, stratify=stratify
    )
    return set(train_ids), set(val_ids)

def save_split(path, train_ids, val_ids, meta=None):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    obj={'train':sorted(map(str,train_ids)), 'val':sorted(map(str,val_ids)), 'meta':meta or {}}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def load_split(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    return set(obj['train']), set(obj['val']), obj.get('meta',{})


def split_duplicate_components(accessions, positive_pairs, val_ratio=0.1, random_state=42):
    """Split whole duplicate-connected components to prevent pair leakage."""
    import random

    ids = sorted(set(map(str, accessions)))
    parent = {item: item for item in ids}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a, b):
        ra, rb = find(str(a)), find(str(b))
        if ra != rb:
            parent[rb] = ra

    missing_ids = sorted({str(item) for pair in positive_pairs for item in pair} - set(ids))
    if missing_ids:
        raise ValueError(
            f"Duplicate manifest IDs are missing from duplicate/: {missing_ids[:10]}"
        )
    positive_nodes = set()
    for a, b in positive_pairs:
        positive_nodes.update((str(a), str(b)))
        union(a, b)

    components = {}
    for item in sorted(parent):
        components.setdefault(find(item), set()).add(item)
    positive_groups = [group for group in components.values() if group & positive_nodes]
    other_groups = [group for group in components.values() if not group & positive_nodes]
    if len(positive_groups) < 2:
        raise ValueError(
            "Component-safe validation requires at least two disconnected positive groups."
        )

    rng = random.Random(random_state)
    rng.shuffle(positive_groups)
    rng.shuffle(other_groups)
    target = max(1, round(len(parent) * float(val_ratio)))
    val_ids = set(positive_groups[0])
    fill_groups = other_groups + positive_groups[1:-1]
    for group in fill_groups:
        if len(val_ids) < target:
            val_ids.update(group)
    train_ids = set(parent) - val_ids
    if not train_ids or not val_ids:
        raise ValueError("Could not create non-empty component-safe train/validation split.")
    return train_ids, val_ids

def get_or_create_true_case_split(abnormal_xlsx, split_path, val_ratio=0.1, random_state=42):
    """One stable 90/10 split for true/normal cases used by downstream tasks and simulation."""
    path=Path(split_path)
    if path.exists():
        return load_split(path)[:2]
    import pandas as pd
    df=pd.read_excel(abnormal_xlsx)
    df['AccessionNumber']=df['AccessionNumber'].astype(str)
    true_ids=sorted(df.loc[df['Label'].astype(str).str.lower().eq('true'),'AccessionNumber'].unique())
    train_ids,val_ids=split_accessions(true_ids,val_ratio,random_state)
    save_split(path,train_ids,val_ids,{'source':'1_abnormal.xlsx','label':'true','val_ratio':val_ratio,'random_state':random_state})
    return train_ids,val_ids
