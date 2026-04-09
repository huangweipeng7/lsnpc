import torch
import warnings
warnings.filterwarnings(
    'ignore', category=UserWarning, 
    message='TypedStorage is deprecated'
)
 
from torchmetrics.classification import( 
    MultilabelAveragePrecision, 
    MultilabelF1Score 
)
from tqdm import tqdm


TOL = 1e-8
 

@torch.inference_mode()
def test(trainer, loader, criterion):
    running_loss = 0.

    trainer.eval()
    device = trainer.device
    n_labels = trainer.n_labels
 
    mAP = MultilabelAveragePrecision(n_labels, average='macro').to(device) 
    macro_f1 = MultilabelF1Score(n_labels, average='macro').to(device)
    micro_f1 = MultilabelF1Score(n_labels, average='micro').to(device)

    for batch in tqdm(loader, colour='green'):
        # Pass to gpu or cpu
        pred = trainer.predict(batch) 
        target = batch['labels'].to(device)
    
        loss = criterion(pred, target.float())
        running_loss += loss.item()
        
        target = target.int()
    
        mAP.update(pred, target) 
        macro_f1.update(pred, target)
        micro_f1.update(pred, target)

    return {
        'loss': running_loss / len(loader),  
        'mAP': round(mAP.compute().item(), 4), 
        'macro_f1': round(macro_f1.compute().item(), 4),
        'micro_f1': round(micro_f1.compute().item(), 4),
    }
      