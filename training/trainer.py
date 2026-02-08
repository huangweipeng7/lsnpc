import numpy as np
import json
import random
import sklearn
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import utils

from copy import deepcopy
from cv2 import transform
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToPILImage
from typing import Callable, Dict, Union

from metrics import test
from utils import WithIndices


def print_metric(data_type, batch):
    print(
        f"{data_type} loss: {batch['loss']:.4f}, "
        f"macro f1: {batch['macro_f1']:.4f}, "
        f"micro f1: {batch['micro_f1']:.4f}, "
        f"macro mAP: {batch['macro_mAP']:.4f}, "
        f"micro mAP: {batch['micro_mAP']:.4f}"
    )


class Trainer:
    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer, 
        arg_dict: Dict,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: Union[str, torch.device] = 'cpu',
        train_on_val: bool = False,
        eval_test_at_final_loop_only: bool = True,
        metric_storing_path: Union[str, Path] = './runs/results.csv'
    ):
        super(Trainer, self).__init__()

        self.model = model 
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.uid = arg_dict['uid']
        self.arg_dict = arg_dict
        self.train_on_val = train_on_val

        self.n_labels = n_labels
        self.model.to(self.device)

        self.eval_test_at_final_loop_only = eval_test_at_final_loop_only
        
        self.metric_storing_path = Path(metric_storing_path)
        self.metric_storing_path.parent.mkdir(parents=True, exist_ok=True)
  
        self.res_path = Path(self.arg_dict['result_dir']) / (
            f'./{self.arg_dict["dataset"]}_{self.arg_dict["noise_type"]}_'
            f'{self.arg_dict["noise_rate"]}_{self.arg_dict["img_encoder"]}_'
            f'ep{self.arg_dict["n_train_epoch"]}_rd{self.arg_dict["run_index"]}/'
        )
 
        self.metric = 0.0
        # self.patience_count = 0
        self.best_ep = 0
 
    def train_model(
        self, 
        n_epochs: int,
        train_loader: DataLoader, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        verbose: bool = False,
        clean_set_loader: DataLoader = None
    ):  
        print(f'self.uid: {self.uid}')
  
        for epoch in range(n_epochs):
            print(f'Epoch {epoch}')
            train_loss = self.train_one_epoch(
                train_loader, val_loader, test_loader, epoch=epoch
            ) 
              
            if self.train_on_val: 
                assert clean_set_loader is not None 
                self.train_on_val_one_epoch(clean_set_loader)   

            self.eval_and_save(
                epoch, n_epochs, val_loader, test_loader, verbose
            )

            if self.lr_scheduler is not None: 
                self.lr_scheduler.step()

    @torch.no_grad()
    def eval_and_save(
        self,   
        epoch: int, 
        n_epochs: int, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        verbose: bool = False
    ): 
        self.eval()

        # It seems sufficient to not use the patience as it may always be unused in most cases.
        if val_loader is not None: #and self.arg_dict['patience'] > self.patentice_count: #====> This seems a bit buggy 
            v_batch = test(self, val_loader, nn.BCELoss(reduction='mean')) 

            if verbose:
                print_metric('val', v_batch) 
 
            v = v_batch['macro_mAP']  
            # v = v_batch['micro_f1']
            if v >= self.metric:
                self.metric = v
                self.best_ep = epoch
                if epoch > n_epochs // 2:
                    self.save_model(self.arg_dict, self.res_path)
                # self.tmp_model = deepcopy(self.model).cpu()

        if test_loader is not None:
            if not self.eval_test_at_final_loop_only:
                t_batch = test(self, test_loader, nn.BCELoss()) 
            elif epoch == n_epochs - 1:
                # Test at the last epoch
                # Load the checkpoint we stored 
                self.model.load_state_dict(
                    torch.load(self.res_path / f'{self.uid}.pth', weights_only=True)
                )
 
                t_batch = test(self, test_loader, nn.BCELoss())
            else:
                return 
 
            if verbose:
                print_metric('test', t_batch)
                print('best epoch:', self.best_ep)
                
            utils.store_results(
                {**t_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'test'},
                self.metric_storing_path
            )
 
    def train_one_epoch(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader = None, 
        test_loader: DataLoader = None,
        epoch: int = -1
    ) -> float:
        self.train()
    
        loss_all = 0.
        n_runs = 0
        for batch in (pbar:=tqdm.tqdm(train_loader)):  
            data, target = (
                batch['data'].to(self.device), 
                batch['labels'].float().to(self.device) 
            )      
            self.optimizer.zero_grad()
    
            pred = self.model(data)
            loss = self.loss_fn(pred, target)
    
            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()
            
            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')
        return loss_all / n_runs   

    @torch.no_grad()
    def predict(self, batch: Dict) -> torch.Tensor:
        data = batch['data'].to(self.device)
        return F.sigmoid(self.model(data))

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save_model(self, arg_dict: Dict, path: Union[str, Path]):
        # Save the model
        path = Path(path)  
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.model.state_dict(), path / f'{self.uid}.pth'
        )
        with open(path / f'{self.uid}.json', 'wt') as f:
            json.dump(arg_dict, f, indent=4)


class MCMTrainer(Trainer):
    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer, 
        arg_dict: Dict,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: Union[str, torch.device] = 'cpu',
        train_on_val: bool = False,
        eval_test_at_final_loop_only: bool = True,
        metric_storing_path: Union[str, Path] = './runs/results.csv'
    ):
        super().__init__(
            n_labels=n_labels,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            device=device,
            train_on_val=train_on_val,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path
        )
 
    def train_model(
        self, 
        n_epochs: int,
        train_loader: DataLoader, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        verbose: bool = False,
        clean_set_loader: DataLoader = None
    ):  
        print(f'self.uid: {self.uid}')
  
        for epoch in range(n_epochs):
            print(f'Epoch {epoch}')
            train_loss = self.train_one_epoch(
                train_loader, val_loader, test_loader, epoch=epoch
            )
            torch.cuda.empty_cache()
            
            # writer.add_scalar('training loss', train_loss, epoch)
       
            if self.train_on_val: 
                assert clean_set_loader is not None 
                self.train_on_val_one_epoch(clean_set_loader)  

            self.eval_and_save(
                epoch, n_epochs, val_loader, test_loader, verbose
            )

            if self.lr_scheduler is not None: 
                self.lr_scheduler.step()

    # @torch.no_grad()
    # def eval_and_save(
    #     self,   
    #     epoch: int, 
    #     n_epochs: int, 
    #     val_loader: DataLoader = None,
    #     test_loader: DataLoader = None, 
    #     verbose: bool = False
    # ): 
    #     self.eval()

    #     # It seems sufficient to not use the patience as it may always be unused in most cases.
    #     if val_loader is not None: #and self.arg_dict['patience'] > self.patentice_count: #====> This seems a bit buggy 
    #         v_batch = test(self, val_loader, nn.BCELoss()) 

    #         if verbose:
    #             print(
    #                 f"val loss: {v_batch['loss']:.4f}, rloss: {v_batch['rloss']:.4f}, " 
    #                 f"macro f1: {v_batch['macro_f1']:.4f}, micro f1: {v_batch['micro_f1']:.4f}, "
    #                 f"mAP: {v_batch['mAP']:.4f}"
    #             )

    #         # utils.store_results({**v_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'val'})
 
    #         if v_batch['micro_f1'] >= self.metric:
    #             self.metric = v_batch['micro_f1'] 
    #             self.best_ep = epoch
    #             self.save_model(self.arg_dict, self.res_path)
    #             # self.tmp_model = deepcopy(self.model).cpu()

    #     if test_loader is not None:
    #         if not self.eval_test_at_final_loop_only:
    #             t_batch = test(self, test_loader, nn.BCELoss()) 
    #         elif epoch == n_epochs - 1:
    #             # Test at the last epoch
    #             # Load the checkpoint we stored 
    #             self.model.load_state_dict(
    #                 torch.load(self.res_path / f'{self.uid}.pth', weights_only=True)
    #             )
 
    #             t_batch = test(self, test_loader, nn.BCELoss())
    #         else:
    #             return 
 
    #         if verbose:
    #             print(
    #                 f"test loss: {t_batch['loss']:.4f}, rloss: {t_batch['rloss']:.4f}, "  
    #                 f"macro f1: {t_batch['macro_f1']:.4f}, micro f1: {t_batch['micro_f1']:.4f}, "
    #                 f"mAP: {t_batch['mAP']:.4f}"
    #             )
    #             print('best epoch:', self.best_ep)
                
    #         utils.store_results(
    #             {**t_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'test'},
    #             self.metric_storing_path
    #         )
 
    def train_one_epoch(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader = None, 
        test_loader: DataLoader = None,
        epoch: int = -1
    ) -> float:
        self.train()
    
        loss_all = 0.
        n_runs = 0
        for batch in (pbar:=tqdm.tqdm(train_loader)):  
            data, target = (
                batch['data'].to(self.device), 
                batch['labels'].float().to(self.device) 
            )      
            self.optimizer.zero_grad()
    
            y_preds, noisy_probs = self.model(data)
            loss, _ = self.loss_fn(noisy_probs, target, self.model.pred_sigmoid(y_preds))
    
            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optimizer.step()
            
            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')
        return loss_all / n_runs   

    @torch.no_grad()
    def predict(self, batch: Dict) -> torch.Tensor:
        data = batch['data'].to(self.device)
        y, _ = self.model(data)
        return F.sigmoid(y)
 

class HLCTrainer(Trainer):
    def __init__(
        self,
        n_labels,
        model,
        loss_fn,
        optimizer, 
        arg_dict,
        lr_scheduler=None,
        device='cpu',
        delta=.4,
        epoch_update_start=5,
        beta=.5,
        eval_test_at_final_loop_only=False,
        metric_storing_path='./runs/results.csv'
    ):
        super().__init__(
            n_labels,
            model,
            loss_fn,
            optimizer, 
            arg_dict,
            lr_scheduler,
            device,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path
        )

        self.delta = delta 
        self.beta = beta 
        self.epoch_update_start = epoch_update_start

    def train_model(
        self, 
        n_epochs,
        train_loader, 
        val_loader=None,
        test_loader=None,  
        verbose=False
    ):  
        print(f'self.uid: {self.uid}')
        # writer = SummaryWriter(
        #     log_dir=self.log_dir,
        #     filename_suffix=self.uid
        # )
 
        # Original labels
        self.labels = []
        for batch in tqdm.tqdm(train_loader):
            self.labels.append(batch['labels'].tolist())
        print('here')

        for ep in range(n_epochs):
            print(f'Epoch {ep}')
            if ep < self.epoch_update_start:
                train_loss = self.train_one_epoch(
                    ep, train_loader, val_loader, test_loader
                )
            else:
                delta = self.delta * max(0, 0.2*(10-ep))
                train_loss, self.labels, corrected_num = self.train_one_epoch_hlc(
                    ep, 
                    delta,
                    train_loader,
                    val_loader,
                    test_loader,
                )
            
            # writer.add_scalar('training loss', train_loss, ep)
                
            self.eval_and_save(ep, n_epochs, val_loader, test_loader, verbose)

            if self.lr_scheduler is not None: 
                self.lr_scheduler.step()

    def train_one_epoch(
        self, 
        ep,
        train_loader: DataLoader, 
        val_loader: DataLoader = None, 
        test_loader: DataLoader = None,
    ) -> float:
        self.train()
    
        loss_all = 0.
        n_runs = 0
        for i, batch in enumerate(pbar:=tqdm.tqdm(train_loader)): 
            data, target = (
                batch['data'].to(self.device), 
                #batch['labels'].float().to(self.device) 
                torch.tensor(self.labels[i], dtype=torch.float32, device=self.device)
            )      
            self.optimizer.zero_grad()
    
            pred, _ = self.model(data)
            loss = self.loss_fn(pred, target)
    
            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optimizer.step()
            
            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')
        return loss_all / n_runs   

    def train_one_epoch_hlc(
        self,
        ep: int, 
        delta: float,
        train_loader: DataLoader, 
        val_loader: DataLoader = None, 
        test_loader: DataLoader = None, 
    ):
        self.train()

        loss_all = 0.
        n_runs = 0
        corrected_targets = list()
        for i, batch in enumerate(pbar:=tqdm.tqdm(train_loader)):
            data, target = (
                batch['data'].to(self.device),
                torch.tensor(self.labels[i], dtype=torch.float32, device=self.device) #self.labels[i].to(self.device)
            )

            self.optimizer.zero_grad()

            pred, label_dependency = self.model(data)
            corrected_labels_batch = torch.zeros((target.size(0), target.size(1)))

            corrected_num = 0
            for j in range(pred.size(0)):
                t_pred = pred[j]
                t_num_labels = torch.nonzero(target[j]).size(0)
                t_noisy_labels = torch.nonzero(target[j])
                t_pred_labels = torch.topk(t_pred, int(t_num_labels)).indices

                original_sc = self.beta * torch.sum(torch.sigmoid(t_pred[t_noisy_labels])) \
                              + (1-self.beta) * utils.label_dependency_capture(label_dependency[j], t_noisy_labels)
                predicted_sc = self.beta * torch.sum(torch.sigmoid(t_pred[t_pred_labels])) \
                               + (1-self.beta) * utils.label_dependency_capture(label_dependency[j], t_pred_labels)

                SR = original_sc / predicted_sc

                if SR <= delta:
                    corrected_labels_batch[j, t_pred_labels] = 1.
                    corrected_num += 1
                else:
                    corrected_labels_batch[j, t_noisy_labels] = 1.

            loss = self.loss_fn(pred.to(self.device), corrected_labels_batch.to(self.device))
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            loss_all += loss.item()
            n_runs += 1

            corrected_targets.append(corrected_labels_batch)

            pbar.set_description(f'training loss: {loss_all/n_runs:.4f} -- HLC (delta={delta}) corrected {corrected_num}/{pred.size(0)}')
        return loss_all / n_runs, corrected_targets, corrected_num

    @torch.no_grad()
    def predict(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor: 
        data = batch['data'].to(self.device)
        pred, _ = self.model(data)
        return F.sigmoid(pred)


class VAETrainer(Trainer):
    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer, 
        arg_dict: Dict,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: Union[str, torch.device] = 'cpu',
        train_on_val: bool = True,
        eval_test_at_final_loop_only: bool = True,
        grad_norm: int = 2, 
        metric_storing_path='./runs/results.csv'
    ):
        super().__init__(
            n_labels=n_labels,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            device=device,
            train_on_val=train_on_val,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path
        )

        self.grad_norm = grad_norm 

    def train_one_epoch(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        epoch: int = -1
    ) -> float:
        self.train()

        loss_all = 0.
        n_runs = 0
        for batch in (pbar:=tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device) 
    
            target_dist = self.get_target(batch)
            target = torch.bernoulli(target_dist)   
            # target = target_dist.round().int() 

            self.optimizer.zero_grad()
  
            res_doc = self.model(data, target)
            loss = self.loss_fn(res_doc, target)
  
            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            
            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')
        return loss_all / n_runs

    def train_on_val_one_epoch(
        self, train_loader: DataLoader  
    ) -> float:
        self.train()

        loss_all = 0.
        n_runs = 0
        # bce = nn.BCELoss()
        for batch in (pbar:=tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device) 
            target = batch['labels'].to(self.device).float() 
            
            # target_hat_dist = self.get_target(batch)
            # target_hat = target_hat_dist.round().int() 
            target_hat = torch.bernoulli(self.get_target(batch))   

            self.optimizer.zero_grad() 
            
            y_pred = self.model(data, target_hat, target) #['y'] 
            loss = self.loss_fn(y_pred, target_hat, target)  

            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            
            pbar.set_description(f'training loss on validation set: {loss_all/n_runs:.4f}')
        return loss_all / n_runs

    @torch.no_grad()
    def predict(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        data = batch['data'].to(self.device) 
        target_dist = self.get_target(batch) 
        sample_type = kwargs.get('sample_type', 'mean')

        self.model.eval()
        if sample_type == 'sample': 
            y = 0.
            n_samples = kwargs.get('n_samples', 8)
            for i in range(n_samples):
                target = torch.bernoulli(target_dist)  
                y0 = self.model(data, target)['y'] 
                y += y0 / n_samples  
        elif sample_type == 'mean':
            y = 0.
            n_samples = kwargs.get('n_samples', 8)
            for i in range(n_samples):
                y += self.model(data, target_dist)['y'] / n_samples
        else:
            raise AttributeError("Only are 'sample' and 'mean' supported.")
 
        assert torch.isnan(y).sum() == 0, y
        return torch.clamp(y, min=0, max=1).float()

    def train(self):
        self.model.train()
        self.model.pretrained_clf.eval()

    def eval(self):
        self.model.eval()

    @torch.no_grad
    def get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.pretrained_clf(batch['data'].to(self.device))
        
        return F.sigmoid(
            out if not isinstance(out, tuple) else out[0]
        )

    # @torch.no_grad()
    # def eval_and_save(
    #     self,  
    #     writer: SummaryWriter, 
    #     epoch: int, 
    #     n_epochs: int, 
    #     val_loader: DataLoader = None,
    #     test_loader: DataLoader = None, 
    #     verbose: bool = False
    # ): 
    #     self.eval()

    #     if not self.train_on_val:
    #         # It seems sufficient to not use the patience as it may always be unused in most cases.
    #         if val_loader is not None: #and self.arg_dict['patience'] > self.patentice_count: #====> This seems a bit buggy 
    #             v_batch = test(self, val_loader, nn.BCELoss())
    #             # for key, item in v_batch.items():
    #             #     writer.add_scalar(f'validation {key}', item, epoch)

    #             if verbose:
    #                 print(
    #                     f"val loss: {v_batch['loss']:.4f}, rloss: {v_batch['rloss']:.4f}, " 
    #                     f"macro f1: {v_batch['macro_f1']:.4f}, micro f1: {v_batch['micro_f1']:.4f}, "
    #                     f"mAP: {v_batch['mAP']:.4f}"
    #                 )

    #             if v_batch['micro_f1'] >= self.metric:
    #                 self.metric = v_batch['micro_f1']
    #                 self.save_model(self.arg_dict, self.res_path) 

    #     elif epoch == n_epochs - 1:
    #         self.save_model(self.arg_dict, self.res_path) 

    #     if test_loader is not None:
    #         if not self.eval_test_at_final_loop_only:
    #             t_batch = test(self, test_loader, nn.BCELoss())
    #             # for key, item in t_batch.items():
    #             #     writer.add_scalar(f'test {key}', item, epoch)
 
    #         elif epoch == n_epochs - 1:
    #             # Test at the last epoch
    #             # Load the checkpoint we stored 
    #             self.model.load_state_dict(
    #                 torch.load(self.res_path / f'{self.uid}.pth', weights_only=True)
    #             )

    #             t_batch = test(self, test_loader, nn.BCELoss())
    #             # for key, item in t_batch.items():
    #             #     writer.add_scalar(f'test {key}', item, epoch)
    #         else:
    #             return 

    #         if verbose:
    #             print(
    #                 f"test loss: {t_batch['loss']:.4f}, rloss: {t_batch['rloss']:.4f}, " 
    #                 f"macro f1: {t_batch['macro_f1']:.4f}, micro f1: {t_batch['micro_f1']:.4f}, "
    #                 f"mAP: {t_batch['mAP']:.4f}"
    #             )

    #         utils.store_results({**t_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'test'})


class KNNTrainer:
    def __init__(
        self,
        n_labels: int,
        pretrained_clf: nn.Module,
        model: KNeighborsClassifier,
        arg_dict: Dict,
        metric_storing_path: str,
        encoder = None,
    ):

        self.pretrained_clf = pretrained_clf
        self.model = model 
        self.device = 'cpu'
        self.uid = arg_dict['uid']
        self.n_labels = n_labels
        self.arg_dict = arg_dict

        self.encoder = encoder
        
        self.metric_storing_path = metric_storing_path
        
        # if 'pretrained_clf' in self.arg_dict:
        #     self.log_dir = f'runs/{self.arg_dict["post_model"]}_{self.arg_dict["clf_name"]}_{self.arg_dict["dataset"]}_{self.arg_dict["run_index"]}'
        # else:
        #     self.log_dir = f'runs/{self.arg_dict["clf_name"]}_{self.arg_dict["dataset"]}_{self.arg_dict["run_index"]}'

        self.models = []
        for i in range(self.n_labels):
            self.models.append(deepcopy(self.model))

    def train_model(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        verbose: bool = False,
    ):  
        print(f'self.uid: {self.uid}')
        # writer = SummaryWriter(
        #     log_dir=self.log_dir,
        #     filename_suffix=self.uid
        # )

        self.pretrained_clf.to(self.device)
        embeddings = []
        preds = []
        for batch in (pbar:=tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device)
            with torch.no_grad():
                emb = self.encoder(data)
                res = self.pretrained_clf(data)
                if isinstance(res, tuple):
                    res = res[0]
                probs = torch.sigmoid(res) 
                embeddings.extend(emb.tolist())
                preds.extend(probs.tolist())
                
        # Store predictions of validation
        for batch in (pbar:=tqdm.tqdm(val_loader)):
            data = batch['data'].to(self.device)
            with torch.no_grad():
                emb = self.encoder(data)
                res = self.pretrained_clf(data)
                if isinstance(res, tuple):
                    res = res[0]
                probs = torch.sigmoid(res)
                embeddings.extend(emb.tolist())
                preds.extend(probs.tolist())
        embeddings = np.array(embeddings)
        preds = np.array(preds)
        print(embeddings.shape, preds.shape)

        # Training
        conf_threshold = 0.1
        for i in range(self.n_labels):
            print(f'Fitting {i}-th KNN')
            preds_i = preds[:, i] 
            print(f'preds_i: {preds_i}')
            preds_i_mask = ((preds_i<conf_threshold) | (preds_i > 1.0-conf_threshold))
            print(f'{i}-th label filtered confident examples: {np.sum(preds_i_mask)}')
            labels_i = np.round(preds_i[preds_i_mask]).astype(int)
            print(f'labels_i: {labels_i}')
            self.models[i].fit(embeddings[preds_i_mask], labels_i)
            print('predictions', self.models[i].predict(embeddings[preds_i_mask]))

        # Eval
        self.eval_and_save(0, val_loader, test_loader, verbose)

    def eval(self):
        ...

    def predict(
        self, 
        batch: Dict[str, torch.Tensor], 
        **kwargs
    ):
        
        data = batch['data'].to(self.device) 
        with torch.no_grad():
            emb = self.encoder(data)
            #res = torch.tensor(self.model.predict(emb)).float()
            res = []
            for i in range(self.n_labels):
                res.append(self.models[i].predict(emb))
            res = np.array(res).T
            # print(res[:5])
            clf_res = self.pretrained_clf(data)
            if isinstance(clf_res, tuple):
                clf_res = clf_res[0]
            # print(torch.sigmoid(clf_res).round())
            return torch.tensor(res).float()

    def eval_and_save(
        self,  
        ep: int, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        verbose: bool = False
    ):
        self.eval()

        if val_loader is not None:
            v_batch = test(self, val_loader, nn.BCELoss())
            # for key, item in v_batch.items():
            #     writer.add_scalar(f'validation {key}', item, ep)

            if verbose:
                print_metric('val', v_batch)

            utils.store_results(
                {**v_batch, **self.arg_dict, 'epoch': ep, 'data_split': 'val'}, 
                self.metric_storing_path
            )

        if test_loader is not None:
            t_batch = test(self, test_loader, nn.BCELoss()) 
                
            if verbose:
                print_metric('test', t_batch)

            utils.store_results(
                {**t_batch, **self.arg_dict, 'epoch': ep, 'data_split': 'test'}, 
                self.metric_storing_path
            )
    

class NPCModTrainer(VAETrainer):
    def __init__(
        self,
        n_labels: int,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer, 
        arg_dict: Dict,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: Union[str, torch.device] = 'cpu',
        train_on_val: bool = False,
        eval_test_at_final_loop_only: bool = True,
        grad_norm: int = 2, 
        metric_storing_path: Union[str, Path] = './runs/results.csv'
    ):
        super().__init__(
            n_labels=n_labels,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            arg_dict=arg_dict,
            lr_scheduler=lr_scheduler,
            device=device,
            train_on_val=train_on_val,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path
        )

        # self.log_dir /= (
        #     'npc_mod_'
        #     f'{self.arg_dict["clf_name"]}_'
        #     f'{self.arg_dict["dataset"]}_'
        #     f'{self.arg_dict["noise_type"]}_'
        #     f'{self.arg_dict["noise_rate"]}_'
        #     f'{self.arg_dict["img_encoder"]}_'
        #     f'ep{self.arg_dict["n_train_epoch"]}_'
        #     f'rd{self.arg_dict["run_index"]}'
        # )

        self.grad_norm = grad_norm 

    def train_one_epoch(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader = None,
        test_loader: DataLoader = None, 
        epoch: int = -1
    ) -> float:
        self.train()

        loss_all = 0.
        n_runs = 0
        for batch in (pbar:=tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device) 
    
            target_dist = self.get_target(batch)
            target = torch.bernoulli(target_dist)   
 
            self.optimizer.zero_grad()
  
            res_doc = self.model(data, target)
            loss = self.loss_fn(res_doc, target, target_dist)
  
            loss.backward()
        
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            
            pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')
        loss_val = loss_all / n_runs

    def train_on_val_one_epoch(
        self, train_loader: DataLoader  
    ) -> float:
        self.train()

        loss_all = 0.
        n_runs = 0
        bce = nn.BCELoss()
        for batch in (pbar:=tqdm.tqdm(train_loader)):
            data = batch['data'].to(self.device) 
            target = batch['labels'].to(self.device).float() 
              
            target_hat = torch.bernoulli(self.get_target(batch))   

            self.optimizer.zero_grad() 
            
            y_pred = self.model(data, target_hat, target)['y'] 
            loss = bce(y_pred, target)  

            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            
            pbar.set_description(f'training loss on validation set: {loss_all/n_runs:.4f}')
        loss_val = loss_all / n_runs 


    @torch.no_grad()
    def predict(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        data = batch['data'].to(self.device) 
        target_dist = self.get_target(batch) 

        sample_type = kwargs.get('sample_type', 'test')

        y = self.model.sample(data, target_dist) 
  
        return y.float()

    def train(self):
        self.model.train()
        self.model.pretrained_clf.eval()

    def eval(self):
        self.model.eval()

    @torch.no_grad
    def get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.pretrained_clf(batch['data'].to(self.device))
        return F.sigmoid(
            out if not isinstance(out, tuple) else out[0]
        ) 
 

class BalanceMixTrainer(Trainer):
    def __init__(
        self,
        n_labels,
        model,
        loss_fn,
        optimizer, 
        arg_dict,
        lr_scheduler=None,
        device='cpu',
        warmup_epochs=5,
        alpha=4.0,
        relabel_weight=1.0, 
        ambiguous_weight=1.0,
        eval_test_at_final_loop_only=False,
        metric_storing_path='./runs/results.csv'
    ):
        super().__init__(
            n_labels,
            model,
            loss_fn,
            optimizer, 
            arg_dict,
            lr_scheduler,
            device,
            eval_test_at_final_loop_only=eval_test_at_final_loop_only,
            metric_storing_path=metric_storing_path
        )

        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.relabel_weight = relabel_weight
        self.ambiguous_weight = ambiguous_weight

    def train_model(
        self, 
        n_epochs,
        train_loader, 
        val_loader=None,
        test_loader=None,  
        verbose=False,
        clean_set_loader: DataLoader = None
    ):  
        
        self.batch_size = train_loader.batch_size

        # Add indices to dataset
        dataset = train_loader.dataset
        self.p_clean = torch.ones(len(dataset), self.n_labels, device=self.device)
        self.clean_mask = torch.ones(len(dataset), self.n_labels, dtype=torch.bool, device=self.device)
        self.labels = torch.zeros(len(dataset), self.n_labels, device=self.device)

        train_loader = DataLoader(
            WithIndices(dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        # Keep updated labels across training, only useful when label refinement persists across epochs
        for batch in train_loader:
            batch_labels = batch['labels'].float().to(self.device)
            batch_indices = batch['index']
            # Assign each batch to its correct positions in self.labels
            self.labels[batch_indices] = batch_labels

        self.dataset = train_loader.dataset

        for epoch in range(n_epochs):
            print(f'Epoch {epoch}')

            # Need to update before 1st epoch, different from Algorithm 1 for mixing samples
            self.update_minority_sampling_probs(train_loader)

            if epoch < self.warmup_epochs:
                # ------------------- Warm-Up Stage -------------------
                if epoch == 0:
                    print('#' * 50)
                    print("Warm-up stage (no label refinement)...")

                self.train_one_epoch(
                    train_loader,
                    epoch=epoch,
                    alpha=self.alpha
                )
            else:
                # ------------------- Full BalanceMix Stage -------------------
                if epoch == self.warmup_epochs: 
                    print('#' * 50)
                    print("Full BalanceMix stage...")

                self.train_one_epoch(
                    train_loader,
                    epoch=epoch,
                    alpha=self.alpha,
                    label_refinement=True,
                )

            # Update GMMs each epoch
            self.fit_gmms_on_training_data(train_loader)
            
            # Keep the same as basic Trainer
            torch.cuda.empty_cache()
        
            if self.train_on_val: 
                assert clean_set_loader is not None 
                self.train_on_val_one_epoch(clean_set_loader)   

            self.eval_and_save(
                epoch, n_epochs, val_loader, test_loader, verbose
            )

            if self.lr_scheduler is not None: 
                self.lr_scheduler.step()
    
    def train_one_epoch(
            self, 
            train_loader: DataLoader,
            epoch: int,
            alpha: float = 4.0, 
            label_refinement: bool = False
        ) -> float:
        
        self.train()
    
        loss_all = 0.
        n_runs = 0
        for batch in (pbar:=tqdm.tqdm(train_loader)):  
            self.optimizer.zero_grad()

            # Get original data and labels 
            target = batch['labels'].float().to(self.device) 
            # Get updated labels from self.labels
            # target = self.labels[batch['index']].float().to(self.device)

            # Label refinement
            if label_refinement:
                new_labels, ambiguous_mask, ratio_clean, ratio_relabel, ratio_ambiguous = \
                    self.gmm_label_refinement_three_state(epoch, batch)
                refined_target = new_labels.to(self.device)
            else:
                refined_target = target

            # Sampling from two samplers and mixup
            imgs, lbs, weights = self.balancemix_two_sampler_batch(
                epoch,
                batch, 
                refined_target, 
                alpha=alpha, 
                label_refinement=label_refinement,
                ambiguous_mask=ambiguous_mask if label_refinement else None
            )

            pred = self.model(imgs)
            loss = self.weighted_bce_loss(
                pred, 
                lbs, 
                weights=weights if label_refinement else None
            )
            # ------------------------------
    
            loss.backward()
    
            loss_all += loss.item()
            n_runs += 1
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optimizer.step()
            
            if label_refinement:
                pbar.set_description(
                    f'training loss: {loss_all/n_runs:.4f} | '
                    f'clean: {ratio_clean:.4f}, relabel: {ratio_relabel:.4f}, ambiguous: {ratio_ambiguous:.4f}'
                )
            else: 
                pbar.set_description(f'training loss: {loss_all/n_runs:.4f}')

        return loss_all / n_runs  
    
    @torch.no_grad()
    def update_minority_sampling_probs(self, train_loader, epsilon=1e-6):
        all_preds = []
        all_labels = []
        all_indices = []
        for batch in train_loader:  # dataset wrapped with WithIndices
            probs = self.predict(batch)
            all_preds.append(probs)
            all_labels.append(batch['labels'])
            all_indices.append(batch['index'])

        all_preds = torch.cat(all_preds)
        all_indices = torch.cat(all_indices)
        all_labels = torch.cat(all_labels).to(self.device)
        # print('all labels:', all_labels[:10])
        # all_labels = self.labels[all_indices].to(self.device)
        # print('updated all labels:', all_labels[:10])

        per_label_conf = all_labels * all_preds + (1 - all_labels) * (1 - all_preds)
        score = per_label_conf.mean(dim=1)
        hardness = 1.0 / (score + epsilon)    # Eq.4: inverse score version
        prob_sampling = hardness / hardness.sum()

        # Initialize tensor for all predictions
        ordered_prob_sampling = torch.empty_like(prob_sampling)
        # Place predictions in their original positions
        ordered_prob_sampling[all_indices] = prob_sampling
        self.prob_sampling = ordered_prob_sampling

        print('Updated minority sampling probabilities.')

    @torch.no_grad()
    def gmm_label_refinement_three_state(self, epoch, batch, eps=0.975):
        """ Returns new_labels including refined and weights for ambiguous samples """
        self.eval()

        data, target = (
            batch['data'].to(self.device), 
            batch['labels'].float().to(self.device) 
            # self.labels[batch['index']].float().to(self.device)
        )     

        clean_mask = self.clean_mask[batch['index']]

        # Decay eps over epochs to 0.95
        # eps -= 0.025 * min(1.0, (epoch - self.warmup_epochs) / 20)

        # Only operate on non-clean labels
        unlabeled = ~clean_mask

        # ----- Apply augmentation to each image in the batch -----
        x_aug1 = []
        x_aug2 = []

        for i in range(data.size(0)):
            # Apply augmentation to individual images
            aug1 = self.rand_aug(data[i])  # x[i] is now 3D
            aug2 = self.rand_aug(data[i])  # x[i] is now 3D
            x_aug1.append(aug1)
            x_aug2.append(aug2)
    
        # Stack back to batch format
        x_aug1 = torch.stack(x_aug1)
        x_aug2 = torch.stack(x_aug2)

        # ----- Relabeling based on augmented ones -----
        p1 = torch.sigmoid(self.model(x_aug1))
        p2 = torch.sigmoid(self.model(x_aug2))
        conf = 0.5 * (p1 + p2)

        new_labels = target.clone()

        # re-label to 1
        mask1 = (conf > eps) & unlabeled
        new_labels[mask1] = 1

        # re-label to 0
        mask0 = (conf < 1-eps) & unlabeled
        new_labels[mask0] = 0

        # ----- Ambiguous: neither clean nor re-labeled -----
        ambiguous = unlabeled & ~(mask1 | mask0)

        ratio_clean = clean_mask.float().mean().item()
        ratio_relabel = (mask1.float().mean() + mask0.float().mean()).item()
        ratio_ambiguous = ambiguous.float().mean().item()

        # update self.labels with new_labels
        self.labels[batch['index']] = new_labels.to(self.device)

        self.train()
        return new_labels, ambiguous, ratio_clean, ratio_relabel, ratio_ambiguous

    @torch.no_grad()
    def fit_gmms_on_training_data(self, train_loader):
        """ Fit GMMs on the training data losses every epoch """
        self.eval()

        labels = []
        probs = []
        indices = []
        for batch in train_loader:
            labels.append(batch['labels'].float().to(self.device))
            indices.append(batch['index'])
            probs.append(self.predict(batch))
        probs = torch.cat(probs, dim=0)
        labels = torch.cat(labels, dim=0)
        
        indices = torch.cat(indices, dim=0)
        # use updated self.labels
        # labels = self.labels[indices].float().to(self.device)

        loss_mat, loss_pos, loss_neg, pos_idx, neg_idx = self.compute_bce_losses(
            probs,
            labels
        )

        self.gmms_pos = self.fit_gmm(loss_pos)
        self.gmms_neg = self.fit_gmm(loss_neg)
        
        new_p_clean = self.compute_clean_prob(loss_mat, pos_idx, neg_idx)
        # p_clean in right positions
        self.p_clean[indices] = new_p_clean

        # Smooth update of clean probabilities
        # if not hasattr(self, 'p_clean'):
        #     self.p_clean = new_p_clean
        # else:
        #     self.p_clean = 0.9 * self.p_clean + 0.1 * new_p_clean

        self.clean_mask[indices] = self.classify_labels(self.p_clean[indices])
        print('Fitted GMMs - current clean ratio:', self.clean_mask.float().mean().item())

        self.train()

    def balancemix_two_sampler_batch(
            self, 
            epoch,
            batch, 
            target, 
            alpha=4.0, 
            label_refinement=False, 
            ambiguous_mask=None
        ):
        """ Generate a batch using two-sampler Mixup strategy """
        data = batch['data'].to(self.device)
        
        # batch_m, batch_d = self.sample_two_batches(epoch, batch, target, self.batch_size)

        all_indices = list(range(len(target)))
        batch_d = random.sample(all_indices, k=min(self.batch_size, len(all_indices)))

        # Batch sampling with minority sampler
        # batch_m = random.choices(
        #     all_indices,  
        #     weights=self.prob_sampling[batch['index']], 
        #     k=min(self.batch_size, len(all_indices))
        # )
        # Global sampling with minority sampler
        batch_m = random.choices(
            range(len(self.dataset)),  
            weights=self.prob_sampling, 
            k=min(self.batch_size, len(all_indices))
        )

        # print(f"Sampled minority indices: {batch_m}")

        # ----- vectorized version -----
        # Convert indices to tensors for vectorized operations
        # batch_m = torch.tensor(batch_m, device=self.device)
        batch_d = torch.tensor(batch_d, device=self.device)
        # Vectorized sampling using index_select
        # x_m = data[batch_m]     # Shape: [batch_size, channels, height, width]
        # y_m = target[batch_m]   # Shape: [batch_size, n_labels]
        x_m = [self.dataset[i]['data'] for i in batch_m]     # Shape: [batch_size, channels, height, width]
        y_m = [self.labels[i] for i in batch_m]   # Shape: [batch_size, n_labels]
        # Convert to tensors and stack
        x_m = torch.stack(x_m).to(self.device)
        y_m = torch.stack(y_m).to(self.device)
        x_d = data[batch_d]     # Shape: [batch_size, channels, height, width]
        y_d = target[batch_d]   # Shape: [batch_size, n_labels]

        # Vectorized mixing
        images, labels, weights = self.mix_batch(
            x_d, x_m, y_d, y_m, 
            alpha, 
            label_refinement, 
            ambiguous_mask
        )

        return images, labels, weights
    
    @torch.no_grad()
    def mix_batch(
        self, 
        x1_batch, 
        x2_batch, 
        y1_batch, 
        y2_batch, 
        alpha=4.0, 
        label_refinement=False, 
        ambiguous_mask=None
    ):
        """  Mix two batches with Mixup strategy.
        
        Args:
            x1_batch: batch from random sampler
            x2_batch: batch from minority sampler
            y1_batch: labels for x1_batch
            y2_batch: labels for x2_batch
            alpha: parameter for Beta distribution
            label_refinement: whether to compute weights for ambiguous samples
            ambiguous_mask: mask for ambiguous samples in the batch

        Returns:
            mixed images and labels for a batch with weights for ambiguous samples
        """
        batch_size = x1_batch.size(0)
        weights = None
        
        if alpha > 0:
            # Generate lambda values for entire batch
            lam_values = np.random.beta(alpha, alpha, size=batch_size)
            # Ensure lambda >= 0.5 by taking max(lambda, 1-lambda)
            lam_values = np.maximum(lam_values, 1 - lam_values)
            # print('Lambda values for the batch:', lam_values)
            lam_tensor = torch.tensor(lam_values, dtype=torch.float32, device=x1_batch.device)
            
            # Reshape for broadcasting: [batch_size, 1, 1, 1] for images
            lam_img = lam_tensor.view(batch_size, 1, 1, 1)
            # Reshape for broadcasting: [batch_size, 1] for labels
            lam_label = lam_tensor.view(batch_size, 1)
        else:
            lam_img = torch.ones(batch_size, 1, 1, 1, device=x1_batch.device)
            lam_label = torch.ones(batch_size, 1, device=x1_batch.device)
        
        # Vectorized mixing
        mixed_images = lam_img * x1_batch + (1 - lam_img) * x2_batch
        mixed_labels = lam_label * y1_batch + (1 - lam_label) * y2_batch

        if label_refinement:
            self.eval()

            # Compute weights for samples
            p1 = torch.sigmoid(self.model(x1_batch))
            loss_mat1, _, _, pos_idx1, neg_idx1 = self.compute_bce_losses(
                p1, y1_batch
            )
            p_clean_batch1 = self.compute_clean_prob(loss_mat1, pos_idx1, neg_idx1)
            w1 = self.ambiguous_weights(p_clean_batch1, ambiguous_mask)

            p2 = torch.sigmoid(self.model(x2_batch))
            loss_mat2, _, _, pos_idx2, neg_idx2 = self.compute_bce_losses(
                p2, y2_batch
            )
            p_clean_batch2 = self.compute_clean_prob(loss_mat2, pos_idx2, neg_idx2)
            w2 = self.ambiguous_weights(p_clean_batch2, ambiguous_mask)

            weights = lam_label * w1 + (1 - lam_label) * w2

            self.train()
        
        return mixed_images, mixed_labels, weights

    def compute_bce_losses(self, probs, labels):
        """ Returns loss_pos[k], loss_neg[k], and index lists """
        bce = nn.BCELoss(reduction='none')
        loss_mat = bce(probs, labels)   # shape [B, K]

        pos_idx = (labels == 1)
        neg_idx = (labels == 0)

        loss_pos = [loss_mat[pos_idx[:, k], k].detach().cpu().numpy() for k in range(labels.shape[1])]
        loss_neg = [loss_mat[neg_idx[:, k], k].detach().cpu().numpy() for k in range(labels.shape[1])]

        # print('loss_mat shape:', loss_mat.shape)
        # print('loss_pos len:', len(loss_pos))
        # print('loss_neg len:', len(loss_neg))

        return loss_mat, loss_pos, loss_neg, pos_idx, neg_idx
    
    def fit_gmm(self, loss_lists):
        """ Fit Gaussian Mixture Models for each class based on loss lists """
        gmms = []
        for losses in loss_lists:     # losses is list of arrays for each class
            if len(losses) < 2:       # avoid crash
                gmms.append(None)
                continue
            losses = np.array(losses).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-4, random_state=42)
            gmm.fit(losses)
            gmms.append(gmm)

        # print(f'{len(gmms)} GMMs fitted.')
        return gmms
    
    def compute_clean_prob(self, loss_mat, pos_idx, neg_idx):
        """ Return computed p_clean for each sample and each class """
        B, K = loss_mat.shape
        p_clean = torch.zeros_like(loss_mat)

        for k in range(K):
            # positive labels
            idx = pos_idx[:, k]
            if self.gmms_pos[k] is not None and idx.any():
                losses = loss_mat[idx, k].detach().cpu().numpy().reshape(-1, 1)
                prob = self.gmms_pos[k].predict_proba(losses)
                small_comp = np.argmin(self.gmms_pos[k].means_)
                p_clean[idx, k] = torch.tensor(prob[:, small_comp], dtype=torch.float32, device=loss_mat.device)

            # negative labels
            idx = neg_idx[:, k]
            if self.gmms_neg[k] is not None and idx.any():
                losses = loss_mat[idx, k].detach().cpu().numpy().reshape(-1, 1)
                prob = self.gmms_neg[k].predict_proba(losses)
                small_comp = np.argmin(self.gmms_neg[k].means_)
                p_clean[idx, k] = torch.tensor(prob[:, small_comp], dtype=torch.float32, device=loss_mat.device)
            
            # Normalize to the same range for all classes
            # p_clean[:, k] = (p_clean[:, k] - p_clean[:, k].min()) / (p_clean[:, k].max() - p_clean[:, k].min() + 1e-8)

        return p_clean
    
    def ambiguous_weights(self, p_clean, ambiguous_mask, scaling_factor=1.0):
        """ Compute weights for ambiguous samples """
        w = torch.ones_like(p_clean)
        w[ambiguous_mask] = p_clean[ambiguous_mask] * scaling_factor
        return w
    
    def rand_aug(self, x):
        """ RandAug transformation: the parameters are not clear from the paper """
        to_pil = ToPILImage()
        transform = transforms.Compose([
            transforms.RandAugment(num_ops=1, magnitude=2),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor()
        ])
        pil_img = to_pil(x.cpu())
        return transform(pil_img).to(x.device)
    
    def classify_labels(self, p_clean, thresh=0.5):
        clean_mask = p_clean > thresh
        return clean_mask

    def weighted_bce_loss(self, logits, labels, weights=None):
        """ Weighted BCE Loss supporting three states """
        if weights is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='mean')
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        return loss