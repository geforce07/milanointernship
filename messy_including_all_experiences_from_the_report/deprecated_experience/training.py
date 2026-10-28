import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

def build_dataloaders(
    dataset_dict,
    batch_size,
    train_frac,
    val_frac,
    seed,
):

    y = torch.tensor(
        dataset_dict["y"],
        dtype=torch.float32
    )

    pre_idx = torch.tensor(
        dataset_dict["pre_idx"],
        dtype=torch.long
    )

    post_idx = torch.tensor(
        dataset_dict["post_idx"],
        dtype=torch.long
    )

    tensors = [
        pre_idx,
        post_idx,
        y
    ]

    # proofreading optional

    if "pre_proof" in dataset_dict:

        pre_proof = torch.tensor(
            dataset_dict["pre_proof"],
            dtype=torch.long
        )

        post_proof = torch.tensor(
            dataset_dict["post_proof"],
            dtype=torch.long
        )

        tensors.extend([
            pre_proof,
            post_proof
        ])

    dataset = TensorDataset(*tensors)

    n_total = len(dataset)

    n_train = int(
        train_frac * n_total
    )

    n_val = int(
        val_frac * n_total
    )

    n_test = (
        n_total
        - n_train
        - n_val
    )

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_y = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = [
                x.to(device, non_blocking=True)
                for x in batch
            ]

            if len(batch) == 4:
                pi, pj, yb, pair_idx = batch
                logits = model.forward_logits(pi, pj)

            elif len(batch) == 6:
                pi, pj, yb, pp, dp, pair_idx = batch
                logits = model.forward_logits(pi, pj, pp, dp)

            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            try:
                loss = criterion(logits, yb, pair_idx)
            except TypeError:
                loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * len(yb)
            all_y.append(yb.detach().cpu().numpy())
            all_preds.append(probs.detach().cpu().numpy())

    all_y = np.concatenate(all_y)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(all_y, all_preds)

    return total_loss / len(loader.dataset), auc