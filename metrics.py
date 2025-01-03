import math
import numpy as np
import torch
import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
 
from torchmetrics.classification import(
    MultilabelAUROC,
    MultilabelAveragePrecision, 
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelRankingAveragePrecision,
    MultilabelRankingLoss
)


TOL = 1e-8
 

@torch.no_grad()
def test(trainer, loader, criterion):
    running_loss = 0.

    trainer.eval()

    device = trainer.device
    n_labels = trainer.n_labels
 
    rloss = MultilabelRankingLoss(n_labels).to(device) 
    mAP = MultilabelRankingAveragePrecision(n_labels).to(device)
    macro_f1 = MultilabelF1Score(n_labels, average='macro').to(device)
    micro_f1 = MultilabelF1Score(n_labels, average='micro').to(device)

    for batch in tqdm.tqdm(loader, colour='green'):
        # Pass to gpu or cpu
        pred = trainer.predict(batch) #.round()
        target = batch['labels'].to(device)
 
        loss = criterion(pred, target.float())
        running_loss += loss.item()
        
        target = target.int()
 
        rloss.update(pred, target) 
        mAP.update(pred, target)
        macro_f1.update(pred, target)
        micro_f1.update(pred, target)

    learn_loss = running_loss / len(loader)

    res_doc = {
        'loss': learn_loss, 
        'rloss': rloss.compute().item(), 
        'mAP': mAP.compute().item(), 
        'macro_f1': macro_f1.compute().item(),
        'micro_f1': micro_f1.compute().item(),
    }
     
    return res_doc
           
     