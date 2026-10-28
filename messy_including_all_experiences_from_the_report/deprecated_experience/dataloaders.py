# dataloaders.py

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def build_dataloaders(
    dataset_dict,
    batch_size,
    train_frac,
    val_frac,
    seed,
):
    y = torch.tensor(dataset_dict["y"], dtype=torch.float32)
    pre_idx = torch.tensor(dataset_dict["pre_idx"], dtype=torch.long)
    post_idx = torch.tensor(dataset_dict["post_idx"], dtype=torch.long)

    pair_idx = torch.arange(len(y), dtype=torch.long)

    tensors = [pre_idx, post_idx, y]

    if "pre_proof" in dataset_dict and "post_proof" in dataset_dict:
        pre_proof = torch.tensor(dataset_dict["pre_proof"], dtype=torch.long)
        post_proof = torch.tensor(dataset_dict["post_proof"], dtype=torch.long)

        tensors.extend([pre_proof, post_proof])

    tensors.append(pair_idx)

    dataset = TensorDataset(*tensors)

    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_set, val_set, test_set