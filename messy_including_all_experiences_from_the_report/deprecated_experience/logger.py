# logger.py

import csv
from pathlib import Path


def create_training_log(log_path):

    log_path = Path(log_path)

    if not log_path.exists():

        with open(log_path, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "dim",
                "epoch",
                "train_loss",
                "val_loss",
                "val_auc",
            ])


def append_training_log(
    log_path,
    dim,
    epoch,
    train_loss,
    val_loss,
    val_auc,
):

    with open(log_path, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            dim,
            epoch,
            train_loss,
            val_loss,
            val_auc,
        ])