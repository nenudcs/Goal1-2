import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.data.paths import PathResolver
from src.data.nifti import load_nifti, robust_normalize, resize_volume, resize_mask, check_nifti

class SegmentationDataset(Dataset):
    """Multiple masks in one SeriesUid are OR-unioned into one training target."""
    def __init__(self, xlsx_path, annotation_root, accessions, task, target_shape, clip_percentiles):
        df=pd.read_excel(xlsx_path)
        df['AccessionNumber']=df['AccessionNumber'].astype(str)
        df['SeriesUid']=df['SeriesUid'].astype(str)
        df['Task']=df['Task'].astype(str).str.strip().str.lower()
        df['SeriesLabel']=df['SeriesLabel'].astype(str).str.strip().str.upper()
        df=df[(df['Task']==task)&(df['AccessionNumber'].isin(set(accessions)))]
        self.resolver=PathResolver(annotation_root)
        self.target_shape=target_shape
        self.clip_percentiles=clip_percentiles
        groups=[]; skipped=[]
        for (acc,suid,slabel,t),g in df.groupby(['AccessionNumber','SeriesUid','SeriesLabel','Task']):
            mask_names=[str(x) for x in g['Maskname'].dropna().tolist()]
            img=self.resolver.series_path(acc,suid,'true')
            if not check_nifti(img):
                skipped.append((acc,suid,'image',str(img))); continue
            bad=[str(self.resolver.mask_path(acc,suid,m)) for m in mask_names if not check_nifti(self.resolver.mask_path(acc,suid,m))]
            if bad:
                skipped.append((acc,suid,'mask',bad)); continue
            groups.append({'AccessionNumber':acc,'SeriesUid':suid,'SeriesLabel':slabel,'Task':t,'Masknames':mask_names})
        self.items=groups
        print(f'[SegmentationDataset:{task}] original_groups={len(groups)+len(skipped)} valid_groups={len(groups)} missing_or_broken_skipped={len(skipped)}')
        for x in skipped: print(f'[SegmentationDataset:{task}][SKIP] {x}')
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        r=self.items[idx]
        img_path=self.resolver.series_path(r['AccessionNumber'],r['SeriesUid'],'true')
        img,_,_=load_nifti(img_path); img=robust_normalize(img,*self.clip_percentiles); x=resize_volume(img,self.target_shape)
        union=None
        for mask_name in r['Masknames']:
            m,_,_=load_nifti(self.resolver.mask_path(r['AccessionNumber'],r['SeriesUid'],mask_name)); m=m>0
            union=m if union is None else np.logical_or(union,m)
        if union is None: union=np.zeros_like(img,dtype=bool)
        y=(resize_mask(union.astype(np.float32),self.target_shape)>0.5).float()
        return x,y,r['AccessionNumber'],r['SeriesUid'],str(img_path)
