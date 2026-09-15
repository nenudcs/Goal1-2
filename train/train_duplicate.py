import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from src.common.config import load_config
from src.common.seed import seed_everything
from src.common.logging_utils import build_logger, CompetitionJSONLLogger
from src.common.train_utils import get_device, save_checkpoint
from src.data.split import split_accessions
from src.tasks.duplicate.dataset import DuplicatePairDataset, duplicate_collate
from src.tasks.duplicate.model import DuplicateModel

def pair_loss(sim,y,margin=0.25):
    pos=y*(1.0-sim); neg=(1.0-y)*torch.relu(sim-margin); return (pos+neg).mean()

def main(cfg):
    if cfg.get("detection", {}).get("enabled"):
        from src.detection.training import train_duplicate
        return train_duplicate(cfg)
    logger=build_logger('train_duplicate'); seed_everything(cfg['project']['seed']); device=get_device(cfg)
    ppath=Path(cfg['paths']['labels_dir'])/'2_duplicate.xlsx'; apath=Path(cfg['paths']['labels_dir'])/'1_abnormal.xlsx'
    pdf=pd.read_excel(ppath); adf=pd.read_excel(apath); adf['AccessionNumber']=adf['AccessionNumber'].astype(str)
    positive_ids=set(pdf['src_img'].astype(str))|set(pdf['desc_img'].astype(str))
    true_ids=set(adf.loc[adf['Label'].astype(str).str.lower().eq('true'),'AccessionNumber'])
    all_ids=sorted(positive_ids|true_ids)
    train_ids,val_ids=split_accessions(all_ids,cfg['split']['val_ratio'],cfg['split']['random_state'])
    train_ds=DuplicatePairDataset(ppath,apath,cfg['paths']['annotation_root'],cfg['data']['target_shape'],cfg['data']['intensity_clip_percentiles'],cfg['duplicate']['negative_ratio'],cfg['project']['seed'],train_ids)
    val_ds=DuplicatePairDataset(ppath,apath,cfg['paths']['annotation_root'],cfg['data']['target_shape'],cfg['data']['intensity_clip_percentiles'],cfg['duplicate']['negative_ratio'],cfg['project']['seed']+1,val_ids)
    loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=0,collate_fn=duplicate_collate)
    vloader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=duplicate_collate)
    logger.info('========== Duplicate training =========='); logger.info('Train cases=%d | Val cases=%d | Train pairs=%d | Val pairs=%d | Device=%s',len(train_ids),len(val_ids),len(train_ds),len(val_ds),device)
    model=DuplicateModel(cfg['duplicate']['embedding_dim']).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    jlog=CompetitionJSONLLogger(cfg['paths']['competition_log_dir'],'duplicate_train.jsonl',fallback_dir=str(Path(cfg['paths']['output_dir'])/'logs'))
    best=float('inf'); global_step=0
    for epoch in range(1,cfg['training']['epochs']+1):
        model.train(); losses=[]
        for step,(xa_list,xb_list,y,a,b) in enumerate(loader,1):
            global_step+=1; xa=xa_list[0].to(device); xb=xb_list[0].to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            sim=model.similarity(xa,xb); loss=pair_loss(sim,y,cfg['duplicate']['margin']); loss.backward(); opt.step(); losses.append(loss.item())
            if step%cfg['training']['log_every']==0 or step==1: logger.info('[TRAIN] epoch=%d step=%d/%d pair=(%s,%s) label=%d similarity=%.4f loss=%.6f seriesA=%d seriesB=%d',epoch,step,len(loader),a[0],b[0],int(y.item()),float(sim.item()),loss.item(),xa.shape[0],xb.shape[0])
        train_loss=sum(losses)/max(len(losses),1)
        model.eval(); vloss=[]; preds=[]; gts=[]
        with torch.no_grad():
            for xa_list,xb_list,y,a,b in vloader:
                sim=model.similarity(xa_list[0].to(device),xb_list[0].to(device)); y=y.to(device); vloss.append(pair_loss(sim,y,cfg['duplicate']['margin']).item()); preds.append(float(((sim+1)/2).item())); gts.append(int(y.item()))
        val_loss=sum(vloss)/max(len(vloss),1); val_acc=0.0 if not gts else accuracy_score(gts,[int(p>=0.5) for p in preds]); val_auc=0.0
        if len(set(gts))==2: val_auc=roc_auc_score(gts,preds)
        logger.info('[VAL] epoch=%d train_loss=%.6f val_loss=%.6f val_acc=%.4f val_auc=%.4f',epoch,train_loss,val_loss,val_acc,val_auc)
        ckpt_dir=Path(cfg['paths']['checkpoints_dir'])/'duplicate'; ckpt=ckpt_dir/f'epoch_{epoch}.pth'; save_checkpoint(model,opt,epoch,ckpt,{'train_loss':train_loss,'val_loss':val_loss,'val_acc':val_acc,'val_auc':val_auc})
        jlog.write(epoch,global_step,'val','training','internal/validation_v2',val_loss,opt.param_groups[0]['lr'],ckpt)
        if val_loss<best: best=val_loss; save_checkpoint(model,opt,epoch,ckpt_dir/'best.pth',{'val_loss':val_loss,'val_acc':val_acc,'val_auc':val_auc})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/config.yaml'); args=parser.parse_args(); main(load_config(args.config))
