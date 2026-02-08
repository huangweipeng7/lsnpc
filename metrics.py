import math
import numpy as np
import torch
import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from sklearn.metrics import (
    accuracy_score,
    coverage_error, 
    hamming_loss, 
    label_ranking_loss, 
    label_ranking_average_precision_score
)
from torchmetrics.classification import( 
    MultilabelAveragePrecision, 
    MultilabelF1Score,
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
    macro_mAP = MultilabelAveragePrecision(n_labels, average='macro').to(device)
    micro_mAP = MultilabelAveragePrecision(n_labels, average='micro').to(device)
    macro_f1 = MultilabelF1Score(n_labels, average='macro').to(device)
    micro_f1 = MultilabelF1Score(n_labels, average='micro').to(device)

    for batch in tqdm.tqdm(loader, colour='green'):
        # Pass to gpu or cpu
        pred = trainer.predict(batch) 
        target = batch['labels'].to(device)

        # print(pred[:3], target[:3])
        # assert (pred >= 0).all() and (pred <= 1).all(), pred.mean()
        # assert (target >= 0).all() and (target <= 1).all()
        
        loss = criterion(pred, target.float())
        running_loss += loss.item()
        
        target = target.int()
 
        macro_mAP.update(pred, target)
        micro_mAP.update(pred, target)
        macro_f1.update(pred, target)
        micro_f1.update(pred, target)

    learn_loss = running_loss / len(loader)

    res_doc = {
        'loss': learn_loss, 
        'micro_mAP': round(micro_mAP.compute().item(), 4), 
        'macro_mAP': round(macro_mAP.compute().item(), 4), 
        'macro_f1': round(macro_f1.compute().item(), 4),
        'micro_f1': round(micro_f1.compute().item(), 4),
    }
     
    return res_doc
