from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_ROOT))
import argparse
import pandas as pd
from src.common.config import load_config
from src.common.logging_utils import build_logger
from src.data.paths import PathResolver
from src.data.nifti import check_nifti

def main(cfg):
    logger=build_logger('validate_data'); labels=Path(cfg['paths']['labels_dir']); resolver=PathResolver(cfg['paths']['annotation_root']); report_dir=Path(cfg['paths']['output_dir'])/'data_validation'; report_dir.mkdir(parents=True,exist_ok=True)
    required=['1_abnormal.xlsx','2_duplicate.xlsx','3_serieslabel.xlsx','4_masklabel.xlsx','5_characteristics.xlsx']
    logger.info('========== Data validation (missing/broken files are skipped) ==========')
    for name in required: logger.info('Found label file: %s',labels/name) if (labels/name).exists() else logger.error('Missing label file: %s',labels/name)
    records=[]
    p=labels/'1_abnormal.xlsx'
    if p.exists():
        df=pd.read_excel(p); df['Label']=df['Label'].astype(str).str.lower(); logger.info('[1_abnormal] label counts:\n%s',df['Label'].value_counts().to_string())
        for i,r in df.iterrows():
            path=resolver.series_path(str(r['AccessionNumber']),str(r['SeriesUid']),str(r['Label'])); ok=check_nifti(path);
            if not ok: records.append({'table':'1_abnormal','row':int(i)+2,'AccessionNumber':str(r['AccessionNumber']),'SeriesUid':str(r['SeriesUid']),'type':'missing_or_broken_image','path':str(path)})
        logger.info('[1_abnormal] rows=%d invalid_skipped=%d',len(df),sum(x['table']=='1_abnormal' for x in records))
    p=labels/'3_serieslabel.xlsx'
    if p.exists():
        df=pd.read_excel(p); bad=0
        for i,r in df.iterrows():
            path=resolver.series_path(str(r['AccessionNumber']),str(r['SeriesUid']),'true');
            if not check_nifti(path): bad+=1; records.append({'table':'3_serieslabel','row':int(i)+2,'AccessionNumber':str(r['AccessionNumber']),'SeriesUid':str(r['SeriesUid']),'type':'missing_or_broken_image','path':str(path)})
        logger.info('[3_serieslabel] rows=%d invalid_skipped=%d',len(df),bad); logger.info('class counts:\n%s',df['SeriesLabel'].value_counts(dropna=False).to_string())
    p=labels/'4_masklabel.xlsx'
    if p.exists():
        df=pd.read_excel(p); bi= bm=0
        for i,r in df.iterrows():
            img=resolver.series_path(str(r['AccessionNumber']),str(r['SeriesUid']),'true'); mask=resolver.mask_path(str(r['AccessionNumber']),str(r['SeriesUid']),str(r['Maskname']))
            if not check_nifti(img): bi+=1; records.append({'table':'4_masklabel','row':int(i)+2,'AccessionNumber':str(r['AccessionNumber']),'SeriesUid':str(r['SeriesUid']),'type':'missing_or_broken_image','path':str(img)})
            if not check_nifti(mask): bm+=1; records.append({'table':'4_masklabel','row':int(i)+2,'AccessionNumber':str(r['AccessionNumber']),'SeriesUid':str(r['SeriesUid']),'type':'missing_or_broken_mask','path':str(mask)})
        logger.info('[4_masklabel] rows=%d invalid_images=%d invalid_masks=%d',len(df),bi,bm)
    out=report_dir/'invalid_files.csv'; pd.DataFrame(records).to_csv(out,index=False,encoding='utf-8-sig')
    logger.info('Validation finished. Invalid/missing report: %s',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/config.yaml'); args=parser.parse_args(); main(load_config(args.config))
