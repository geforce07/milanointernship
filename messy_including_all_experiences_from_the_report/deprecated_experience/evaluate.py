import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def evaluate(
    model,
    loader,
    criterion,
    device,
):
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

            # --------------------
            # Without proofreading
            # --------------------

            if len(batch) == 4:

                pi, pj, yb, pair_idx = batch

                logits = model.forward_logits(
                    pi,
                    pj
                )

            # --------------------
            # With proofreading
            # --------------------

            elif len(batch) == 6:

                pi, pj, yb, pp, dp, pair_idx = batch

                logits = model.forward_logits(
                    pi,
                    pj,
                    pp,
                    dp
                )

            else:
                raise ValueError(
                    f"Unexpected batch length: {len(batch)}"
                )

            # --------------------
            # Loss
            # --------------------

            try:
                loss = criterion(
                    logits,
                    yb,
                    pair_idx
                )

            except TypeError:

                loss = criterion(
                    logits,
                    yb
                )

            probs = torch.sigmoid(logits)

            total_loss += (
                loss.item()
                * len(yb)
            )

            all_y.append(
                yb.detach().cpu().numpy()
            )

            all_preds.append(
                probs.detach().cpu().numpy()
            )

    all_y = np.concatenate(all_y)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(
        all_y,
        all_preds
    )

    return (
        total_loss / len(loader.dataset),
        auc
    )