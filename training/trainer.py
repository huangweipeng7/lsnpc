import numpy as np
import json
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import utils
import sklearn

from copy import deepcopy
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, Union

from metrics import test


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
        
        self.metric_storing_path = metric_storing_path

        self.log_dir = Path('runs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if 'pretrained_clf' in self.arg_dict:
            self.log_dir /= f'{self.arg_dict["post_model"]}'

        self.log_dir /= (
            f'{self.arg_dict["clf_name"]}_'
            f'{self.arg_dict["dataset"]}_'
            f'{self.arg_dict["noise_type"]}_'
            f'{self.arg_dict["noise_rate"]}_'
            f'{self.arg_dict["img_encoder"]}_'
            f'ep{self.arg_dict["n_train_epoch"]}_'
            f'rd{self.arg_dict["run_index"]}'
        )

        self.res_path = Path(self.arg_dict['result_dir']) / (
            f'./{self.arg_dict["dataset"]}_{self.arg_dict["noise_type"]}_'
            f'{self.arg_dict["noise_rate"]}_{self.arg_dict["img_encoder"]}_'
            f'ep{self.arg_dict["n_train_epoch"]}_rd{self.arg_dict["run_index"]}/'
        )
 
        self.metric = 0 
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
            v_batch = test(self, val_loader, nn.BCELoss()) 

            if verbose:
                print(
                    f"val loss: {v_batch['loss']:.4f}, rloss: {v_batch['rloss']:.4f}, " 
                    f"macro f1: {v_batch['macro_f1']:.4f}, micro f1: {v_batch['micro_f1']:.4f}, "
                    f"mAP: {v_batch['mAP']:.4f}"
                )

            # utils.store_results({**v_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'val'})
 
            if v_batch['micro_f1'] >= self.metric:
                self.metric = v_batch['micro_f1'] 
                self.best_ep = epoch
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
                print(
                    f"test loss: {t_batch['loss']:.4f}, rloss: {t_batch['rloss']:.4f}, "  
                    f"macro f1: {t_batch['macro_f1']:.4f}, micro f1: {t_batch['micro_f1']:.4f}, "
                    f"mAP: {t_batch['mAP']:.4f}"
                )
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
    
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
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
            self.model.cpu().state_dict(), 
            path / f'{self.uid}.pth'
        )
        with open(path / f'{self.uid}.json', 'wt') as f:
            json.dump(arg_dict, f, indent=4)
        # Back to GPU in case keep training
        self.model.to(self.device)


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
        log_dir=None,
        verbose=False
    ):  
        print(f'self.uid: {self.uid}')
        writer = SummaryWriter(
            log_dir=self.log_dir,
            filename_suffix=self.uid
        )

        print('there')
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

            pbar.set_description(f'training loss: {loss_all/n_runs:.4f} -- ' 
                                  f'HLC (delta={delta:.4f}) corrected '
                                  f'{corrected_num}/{pred.size(0)}')
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

            target_hat = torch.bernoulli(self.get_target(batch))   

            self.optimizer.zero_grad() 
            
            y_pred = self.model(data, target_hat, target)  
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
            n_samples = kwargs.get('n_samples', 5)
            for i in range(n_samples):
                target = torch.bernoulli(target_dist)  
                y0 = self.model(data, target)['y'] 
                y += y0 / n_samples  
        elif sample_type == 'mean':
            y = 0.
            n_samples = kwargs.get('n_samples', 5)
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
 

class KNNTrainer:
    def __init__(
        self,
        n_labels: int,
        pretrained_clf: nn.Module,
        model: KNeighborsClassifier,
        arg_dict: Dict,
        encoder = None,
    ):

        self.pretrained_clf = pretrained_clf
        self.model = model 
        self.device = 'cpu'
        self.uid = arg_dict['uid']
        self.n_labels = n_labels
        self.arg_dict = arg_dict

        self.encoder = encoder
        
        if 'pretrained_clf' in self.arg_dict:
            self.log_dir = f'runs/{self.arg_dict["post_model"]}_{self.arg_dict["clf_name"]}_{self.arg_dict["dataset"]}_{self.arg_dict["run_index"]}'
        else:
            self.log_dir = f'runs/{self.arg_dict["clf_name"]}_{self.arg_dict["dataset"]}_{self.arg_dict["run_index"]}'

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
        writer = SummaryWriter(
            log_dir=self.log_dir,
            filename_suffix=self.uid
        )

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
            res = []
            for i in range(self.n_labels):
                res.append(self.models[i].predict(emb))
            res = np.array(res).T 

            clf_res = self.pretrained_clf(data)
            if isinstance(clf_res, tuple):
                clf_res = clf_res[0] 
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

            if verbose:
                print(
                    f"val loss: {v_batch['loss']:.4f}, rloss: {v_batch['rloss']:.4f}, " 
                    f"macro f1: {v_batch['macro_f1']:.4f}, micro f1: {v_batch['micro_f1']:.4f}, "
                    f"mAP: {v_batch['mAP']:.4f}"
                )

            utils.store_results({**v_batch, **self.arg_dict, 'epoch': ep, 'data_split': 'val'})

        if test_loader is not None:
            t_batch = test(self, test_loader, nn.BCELoss())
            # for key, item in t_batch.items():
            #     writer.add_scalar(f'test {key}', item, ep)
                
            if verbose:
                print(
                    f"test loss: {t_batch['loss']:.4f}, rloss: {t_batch['rloss']:.4f}, " 
                    f"macro f1: {t_batch['macro_f1']:.4f}, micro f1: {t_batch['micro_f1']:.4f}, "
                    f"mAP: {v_batch['mAP']:.4f}"
                )

            utils.store_results({**t_batch, **self.arg_dict, 'epoch': ep, 'data_split': 'test'})
    

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

        self.log_dir /= (
            'npc_mod_'
            f'{self.arg_dict["clf_name"]}_'
            f'{self.arg_dict["dataset"]}_'
            f'{self.arg_dict["noise_type"]}_'
            f'{self.arg_dict["noise_rate"]}_'
            f'{self.arg_dict["img_encoder"]}_'
            f'ep{self.arg_dict["n_train_epoch"]}_'
            f'rd{self.arg_dict["run_index"]}'
        )
        self.sw_train = SummaryWriter(log_dir=self.log_dir / 'training', filename_suffix=self.uid)
        self.sw_val = SummaryWriter(log_dir=self.log_dir /  'validation', filename_suffix=self.uid)
        self.sw_test = SummaryWriter(log_dir=self.log_dir / 'test', filename_suffix=self.uid)

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
        self.sw_train.add_scalar('loss', loss_val, epoch)

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
        
        if val_loader is not None:  
            v_batch = test(self, val_loader, nn.BCELoss())
            for key, item in v_batch.items():
                self.sw_val.add_scalar(f'{key}', item, epoch)

            if verbose:
                print(
                    f"val loss: {v_batch['loss']:.4f}, rloss: {v_batch['rloss']:.4f}, " 
                    f"macro f1: {v_batch['macro_f1']:.4f}, micro f1: {v_batch['micro_f1']:.4f}, "
                    f"mAP: {v_batch['mAP']:.4f}"
                )

            print('hah?')
            utils.store_results(
                {**v_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'val'},
                self.metric_storing_path
            )
 
            if v_batch['micro_f1'] >= self.metric:
                self.metric = v_batch['micro_f1']
                self.best_ep = epoch
                self.save_model(self.arg_dict, self.res_path) 

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

            for key, item in t_batch.items():
                self.sw_test.add_scalar(f'{key}', item, epoch)
                
            if verbose:
                print(
                    f"test loss: {t_batch['loss']:.4f}, rloss: {t_batch['rloss']:.4f}, "  
                    f"macro f1: {t_batch['macro_f1']:.4f}, micro f1: {t_batch['micro_f1']:.4f}, "
                    f"mAP: {t_batch['mAP']:.4f}"
                )
                print('best epoch:', self.best_ep)
                
            utils.store_results(
                {**t_batch, **self.arg_dict, 'epoch': epoch, 'data_split': 'test'},
                self.metric_storing_path
            )
 
