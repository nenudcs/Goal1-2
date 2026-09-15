from pathlib import Path
import json
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
            if min(counts.values()) >= 2:
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
