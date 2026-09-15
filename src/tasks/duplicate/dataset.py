import random
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.paths import PathResolver
from src.data.nifti import load_preprocessed_volume, check_nifti

def build_negative_pairs(abnormal_xlsx, positive_xlsx, negative_ratio=3, seed=42):
    rng=random.Random(seed)
    adf=pd.read_excel(abnormal_xlsx); adf['AccessionNumber']=adf['AccessionNumber'].astype(str)
    true_ids=sorted(adf.loc[adf['Label'].astype(str).str.lower().eq('true'),'AccessionNumber'].drop_duplicates().tolist())
    pdf=pd.read_excel(positive_xlsx); pdf['src_img']=pdf['src_img'].astype(str); pdf['desc_img']=pdf['desc_img'].astype(str)
    positive_set={tuple(sorted((a,b))) for a,b in zip(pdf['src_img'],pdf['desc_img'])}
    n_needed=len(pdf)*int(negative_ratio); negatives=set(); attempts=0; max_attempts=max(1000,n_needed*100)
    if len(true_ids)<2: return []
    while len(negatives)<n_needed and attempts<max_attempts:
        attempts+=1; a,b=rng.sample(true_ids,2); pair=tuple(sorted((a,b)))
        if pair not in positive_set: negatives.add(pair)
    return [(a,b,0) for a,b in sorted(negatives)]

def _list_series_under_accession(accession_dir: Path):
    items=[]
    if not accession_dir.exists(): return items
    for sd in sorted(accession_dir.iterdir()):
        if not sd.is_dir(): continue
        p=sd/f'{sd.name}.nii.gz'
        if p.is_file() and check_nifti(p): items.append(p)
    return items

class DuplicatePairDataset(Dataset):
    def __init__(self, positive_xlsx, abnormal_xlsx, annotation_root, target_shape, clip_percentiles, negative_ratio=3, seed=42, accessions=None):
        self.resolver=PathResolver(annotation_root); self.target_shape=target_shape; self.clip_percentiles=clip_percentiles
        pdf=pd.read_excel(positive_xlsx)
        positives=[(str(a),str(b),1) for a,b in zip(pdf['src_img'],pdf['desc_img'])]
        negatives=build_negative_pairs(abnormal_xlsx,positive_xlsx,negative_ratio,seed)
        items=positives+negatives
        allowed=set(map(str,accessions)) if accessions is not None else None
        self.items=[]; skipped=[]
        for a,b,y in items:
            if allowed is not None and not (a in allowed and b in allowed): continue
            pos=(y==1)
            da=self.resolver.duplicate_accession_dir(a) if pos else self.resolver.root/a
            db=self.resolver.duplicate_accession_dir(b) if pos else self.resolver.root/b
            sa=_list_series_under_accession(da); sb=_list_series_under_accession(db)
            if not sa or not sb:
                skipped.append((a,b,y,len(sa),len(sb))); continue
            self.items.append((a,b,y))
        print(f'[DuplicatePairDataset] original_pairs={len(items)} valid_pairs={len(self.items)} missing_or_broken_skipped={len(skipped)}')
        for x in skipped: print(f'[DuplicatePairDataset][SKIP] pair={x[:3]} seriesA={x[3]} seriesB={x[4]}')
    def __len__(self): return len(self.items)
    def _case_volume_list(self,accession,positive_side=False):
        base=self.resolver.duplicate_accession_dir(accession) if positive_side else self.resolver.root/accession
        paths=_list_series_under_accession(base); vols=[]
        for p in paths:
            try:
                x,_=load_preprocessed_volume(p,self.target_shape,self.clip_percentiles); vols.append(x)
            except Exception as e:
                print(f'[DuplicatePairDataset][NIFTI SKIP] {p} | {type(e).__name__}: {e}')
        if not vols: raise RuntimeError(f'No valid NIfTI for case {accession}')
        return torch.stack(vols,dim=0)
    def __getitem__(self,idx):
        a,b,y=self.items[idx]; pos=(y==1)
        return self._case_volume_list(a,pos),self._case_volume_list(b,pos),torch.tensor(float(y)),a,b

def duplicate_collate(batch):
    xa,xb,y,a,b=zip(*batch); return list(xa),list(xb),torch.stack(y),list(a),list(b)
