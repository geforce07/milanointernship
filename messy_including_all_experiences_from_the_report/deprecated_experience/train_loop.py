# train_loop.py

import os
import json
import numpy as np
import torch
from pathlib import Path

from evaluate import evaluate
from checkpoints import save_checkpoint, load_checkpoint
from logger import create_training_log, append_training_log
from losses import create_loss



def compute_batch_loss(criterion, logits, yb, pair_idx=None):
    try:
        return criterion(logits, yb, pair_idx)
    except TypeError:
        return criterion(logits, yb)


def run_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
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

        loss = compute_batch_loss(
            criterion,
            logits,
            yb,
            pair_idx
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)

    return total_loss / len(train_loader.dataset)


def run_training_for_dims(
    dims,
    model_factory,
    train_loader,
    val_loader,
    test_loader,
    y,
    train_set,
    output_dir,
    device,
    lr=1e-3,
    n_epochs=50,
    patience=10,
    loss_name="bce",
    distances=None,
    dist_bins=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    results_path = output_dir / "summary_results.json"

    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = []

    for dim in dims:
        print("\n====================")
        print(f"Training dim = {dim}")
        print("====================")

        model = model_factory(dim).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        train_indices = train_set.indices
        y_train = y[train_indices]

        criterion = create_loss(
            loss_name=loss_name,
            y_train=y_train,
            distances=distances,
            dist_bins=dist_bins,
            device=device,
        )

        last_ckpt_path = checkpoint_dir / f"last_dim_{dim}.pt"
        best_ckpt_path = checkpoint_dir / f"best_dim_{dim}.pt"
        log_csv_path = log_dir / f"training_log_dim_{dim}.csv"

        create_training_log(log_csv_path)

        start_epoch = 0
        best_val_loss = np.inf
        best_val_auc = -np.inf
        best_epoch = None
        epochs_without_improvement = 0

        if last_ckpt_path.exists():
            print(f"Resuming dim={dim} from {last_ckpt_path}")

            ckpt = load_checkpoint(
                last_ckpt_path,
                model,
                optimizer,
                device=device
            )

            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", np.inf)
            best_val_auc = ckpt.get("best_val_auc", -np.inf)
            best_epoch = ckpt.get("best_epoch", None)
            epochs_without_improvement = ckpt.get(
                "epochs_without_improvement",
                0
            )

        if start_epoch >= n_epochs:
            print(f"dim={dim} already completed. Skipping training.")

        else:
            for epoch in range(start_epoch, n_epochs):
                train_loss = run_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )

                val_loss, val_auc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                print(
                    f"dim={dim:02d} | epoch={epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_auc={val_auc:.4f}"
                )

                append_training_log(
                    log_csv_path,
                    dim,
                    epoch,
                    train_loss,
                    val_loss,
                    val_auc,
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    save_checkpoint(
                        best_ckpt_path,
                        model,
                        optimizer,
                        dim,
                        epoch,
                        best_val_loss,
                        best_val_auc,
                        best_epoch,
                        epochs_without_improvement,
                    )

                else:
                    epochs_without_improvement += 1

                save_checkpoint(
                    last_ckpt_path,
                    model,
                    optimizer,
                    dim,
                    epoch,
                    best_val_loss,
                    best_val_auc,
                    best_epoch,
                    epochs_without_improvement,
                )

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping at epoch {epoch} | "
                        f"best_epoch={best_epoch} | "
                        f"best_val_auc={best_val_auc:.4f}"
                    )
                    break

        if not best_ckpt_path.exists():
            raise FileNotFoundError(
                f"No best checkpoint found for dim={dim}: {best_ckpt_path}"
            )

        best_ckpt = load_checkpoint(
            best_ckpt_path,
            model,
            optimizer=None,
            device=device,
        )

        test_loss, test_auc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        results = [
            r for r in results
            if r["dim"] != dim
        ]

        result = {
            "dim": dim,
            "best_epoch": int(best_ckpt["best_epoch"]),
            "best_val_loss": float(best_ckpt["best_val_loss"]),
            "best_val_auc": float(best_ckpt["best_val_auc"]),
            "test_loss": float(test_loss),
            "test_auc": float(test_auc),
        }

        results.append(result)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print(
            f"BEST dim={dim:02d} | "
            f"epoch={result['best_epoch']} | "
            f"val_auc={result['best_val_auc']:.4f} | "
            f"test_auc={result['test_auc']:.4f}"
        )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return sorted(results, key=lambda x: x["dim"]) 