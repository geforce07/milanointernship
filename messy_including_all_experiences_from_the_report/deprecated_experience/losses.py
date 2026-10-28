import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, distances, dist_bins):
        super().__init__()


        distances = torch.as_tensor(distances, dtype=torch.float32)
        dist_bins = torch.as_tensor(dist_bins, dtype=torch.float32)

        bin_idx = torch.bucketize(
            distances,
            dist_bins
        ) - 1 #[80, 200, 100, 50] -> [0,2,1,0]

        bin_idx = torch.clamp(
            bin_idx,
            min=0,
            max=len(dist_bins) - 2
        ) #bring all distance in bin_idx [3] -> [2]

        pair_count = torch.bincount(
            bin_idx,
            minlength=len(dist_bins) - 1
        ).float() #[2, 1, 1]

        weights_per_bin = torch.zeros_like(pair_count)
        weights_per_bin[pair_count > 0] = 1.0 / pair_count[pair_count > 0]

        weights = weights_per_bin[bin_idx] #[1/2, 1, 1, 1/2]
        weights = weights / weights.mean()

        self.register_buffer("weights", weights)

    def forward(self, logits, targets, indices=None):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        if indices is None:
            raise ValueError("indices must be provided for distance weighted BCE")

        indices = indices.to(self.weights.device)

        w = self.weights[indices].to(logits.device)

        return (w * bce).mean()

"""alessandro distance weightloss :

% d_arr  : [80, 200, 100, 40]
% y : binary array (1 = connected, 0 = not connected)
% p_pred : predicted connection probability for each pair (from your model)

% Weight each pair by the inverse count of its distance bin, so every % bin contributes equally regardless of how many pairs it contains.

bin_idx = np.digitize(d_arr, dist_bins) - 1 #link each distance to a bins with an index [0,2,1,0]
pc = np.bincount(bin_idx, minlength=len(dist_bins) - 1)  #count the number of element by bins
wpb = np.zeros(len(dist_bins) - 1, dtype=np.float32)
wpb[pc > 0] = 1.0 / pc[pc > 0] #[1/2, 1, 1]
weights = wpb[bin_idx] 
weights /= weights.sum() #1/2+1+1+1/2

% Weighted BCE

y_f = y.astype(np.float32)
bce = -(y_f * np.log(p_pred) + (1 - y_f) * np.log(1 - p_pred))
loss = np.dot(weights, bce) #loss = sum (weigths[i] * bce[i]

"""

def create_loss(
    loss_name,
    y_train=None,
    distances=None,
    dist_bins=None,
    device="cpu",
):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "positive_weighted_bce":
        if y_train is None:
            raise ValueError("y_train is required")

        y_train = torch.as_tensor(
            y_train,
            dtype=torch.float32,
            device=device
        )

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()

        pos_weight = n_neg / n_pos
        print("pos_weight:", pos_weight)
        print("type:", type(pos_weight))
        print("is tensor:", torch.is_tensor(pos_weight))
        return nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

    if loss_name == "distance_weighted_bce":
        if distances is None or dist_bins is None:
            raise ValueError("distances and dist_bins are required")

        return DistanceWeightedBCEWithLogitsLoss(
            distances=distances,
            dist_bins=dist_bins
        )

    raise ValueError(f"Unknown loss_name: {loss_name}")