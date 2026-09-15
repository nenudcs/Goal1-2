import argparse, json, shutil
from pathlib import Path
from src.common.config import load_config
from src.common.logging_utils import build_logger
from src.common.train_utils import get_device
from src.data.split import get_or_create_true_case_split
from src.pipeline.load_models import load_all_models
from src.pipeline.batch_pipeline import run_batch

def main(cfg):
    logger=build_logger('simulate_validation')
    labels=Path(cfg['paths']['labels_dir']); root=Path(cfg['paths']['annotation_root'])
    split_path=Path(cfg['paths']['output_dir'])/'splits'/'true_case_split_0.1.json'
    train_ids,val_ids=get_or_create_true_case_split(labels/'1_abnormal.xlsx',split_path,cfg['validation_simulation']['ratio'],cfg['validation_simulation']['random_state'])
    out=Path(cfg['paths']['output_dir'])/cfg['validation_simulation']['output_subdir']; input_root=out/'input'; result_root=out/'results'
    input_root.mkdir(parents=True,exist_ok=True)
    for acc in sorted(val_ids):
        src=root/acc; dst=input_root/acc
        if dst.exists(): continue
        try: dst.symlink_to(src,target_is_directory=True)
        except OSError: shutil.copytree(src,dst)
    (out/'split.json').write_text(json.dumps({'train':sorted(train_ids),'validation':sorted(val_ids)},ensure_ascii=False,indent=2),encoding='utf-8')
    device=get_device(cfg); logger.info('Validation cases=%d | input=%s | results=%s | device=%s',len(val_ids),input_root,result_root,device)
    models=load_all_models(cfg,device); run_batch(input_root,result_root,models,cfg,device)
    logger.info('Simulation finished. Outputs are in %s',result_root)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/config.yaml'); args=parser.parse_args(); main(load_config(args.config))
