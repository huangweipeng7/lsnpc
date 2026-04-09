"""KNN trainer for per-label classification."""

import logging
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from metrics import test
from utils import store_results

logger = logging.getLogger(__name__)


def log_metric(data_type: str, batch: Dict) -> None:
    """Log training/validation metrics."""
    logger.info(
        f"{data_type} loss: {batch['loss']:.4f}, "
        f"macro f1: {batch['macro_f1']:.4f}, "
        f"micro f1: {batch['micro_f1']:.4f}, "
        f"mAP: {batch['mAP']:.4f}, "
        f"micro mAP: {batch['micro_mAP']:.4f}"
    )


class KNNTrainer:
    """K-Nearest Neighbors trainer for per-label classification.

    Args:
        n_labels: Number of labels.
        pretrained_clf: Pretrained classifier.
        model: KNN classifier template.
        arg_dict: Arguments dictionary.
        metric_storing_path: Path to store results.
        encoder: Feature encoder.
    """

    def __init__(
        self,
        n_labels: int,
        pretrained_clf: nn.Module,
        model: KNeighborsClassifier,
        arg_dict: Dict,
        metric_storing_path: str,
        encoder: Optional[nn.Module] = None
    ):
        self.pretrained_clf = pretrained_clf
        self.model = model
        self.device = 'cpu'
        self.uid = arg_dict['uid']
        self.n_labels = n_labels
        self.arg_dict = arg_dict
        self.encoder = encoder
        self.metric_storing_path = metric_storing_path

        self.models: List[KNeighborsClassifier] = []
        for _ in range(self.n_labels):
            self.models.append(deepcopy(self.model))

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        verbose: bool = False
    ) -> None:
        """Train KNN classifiers for each label."""
        logger.info(f'Training KNN with uid: {self.uid}')

        self.pretrained_clf.to(self.device)
        embeddings = []
        preds = []

        for batch in tqdm.tqdm(train_loader):
            data = batch['data'].to(self.device)
            with torch.inference_mode():
                emb = self.encoder(data)
                res = self.pretrained_clf(data)
                if isinstance(res, tuple):
                    res = res[0]
                probs = torch.sigmoid(res)
                embeddings.extend(emb.tolist())
                preds.extend(probs.tolist())

        for batch in tqdm.tqdm(val_loader):
            data = batch['data'].to(self.device)
            with torch.inference_mode():
                emb = self.encoder(data)
                res = self.pretrained_clf(data)
                if isinstance(res, tuple):
                    res = res[0]
                probs = torch.sigmoid(res)
                embeddings.extend(emb.tolist())
                preds.extend(probs.tolist())

        embeddings = np.array(embeddings)
        preds = np.array(preds)
        logger.info(f'Embeddings shape: {embeddings.shape}, Predictions shape: {preds.shape}')

        conf_threshold = 0.1
        for i in range(self.n_labels):
            logger.info(f'Training KNN for label {i}')
            preds_i = preds[:, i]
            confident_mask = (preds_i < conf_threshold) | (preds_i > 1.0 - conf_threshold)
            logger.info(f'Confident examples: {np.sum(confident_mask)}')

            labels_i = np.round(preds_i[confident_mask]).astype(int)
            self.models[i].fit(embeddings[confident_mask], labels_i)
            logger.debug(f'Predictions: {self.models[i].predict(embeddings[confident_mask])}')

        self.eval_and_save(0, val_loader, test_loader, verbose)

    def eval(self) -> None:
        """Set to evaluation mode (no-op for KNN)."""
        pass

    @torch.inference_mode()
    def predict(self, batch: Dict, **kwargs) -> torch.Tensor:
        """Make predictions using ensemble of KNNs."""
        data = batch['data'].to(self.device)

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
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        verbose: bool = False
    ) -> None:
        """Evaluate and save results."""
        self.eval()

        if val_loader is not None:
            v_batch = test(self, val_loader, nn.BCEWithLogitsLoss())

            if verbose:
                log_metric('val', v_batch)

            store_results(
                {**v_batch, **self.arg_dict, 'epoch': ep, 'data_split': 'val'},
                self.metric_storing_path
            )

        if test_loader is not None:
            t_batch = test(self, test_loader, nn.BCEWithLogitsLoss())

            if verbose:
                log_metric('test', t_batch)

            store_results(
                {**t_batch, **self.arg_dict, 'epoch': ep, 'data_split': 'test'},
                self.metric_storing_path
            )
