# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEmbeddingModel(nn.Module):
    def __init__(
        self,
        n_neurons,
        dim=2,
        n_pre_proof=3,
        n_post_proof=3,
        hidden=64,
        use_proofreading=True,
        shared_embedding=True,
    ):
        super().__init__()

        self.dim = dim
        self.n_pre_proof = n_pre_proof
        self.n_post_proof = n_post_proof
        self.use_proofreading = use_proofreading
        self.shared_embedding = shared_embedding

        if shared_embedding:
            self.Z = nn.Embedding(n_neurons, dim)
            nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)
        else:
            self.Z_pre = nn.Embedding(n_neurons, dim)
            self.Z_post = nn.Embedding(n_neurons, dim)

            nn.init.normal_(self.Z_pre.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.Z_post.weight, mean=0.0, std=0.1)

        input_dim = 2 * dim

        if self.use_proofreading:
            input_dim += n_pre_proof + n_post_proof

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def get_latent(self, pre_idx, post_idx):
        if self.shared_embedding:
            z_pre = self.Z(pre_idx)
            z_post = self.Z(post_idx)
        else:
            z_pre = self.Z_pre(pre_idx)
            z_post = self.Z_post(post_idx)

        return z_pre, z_post

    def forward_logits(self, pre_idx, post_idx, pre_proof=None, post_proof=None):
        z_pre, z_post = self.get_latent(pre_idx, post_idx)

        inputs = [z_pre, z_post]

        if self.use_proofreading:
            if pre_proof is None or post_proof is None:
                raise ValueError(
                    "pre_proof and post_proof must be provided when use_proofreading=True"
                )

            p_pre = F.one_hot(
                pre_proof,
                num_classes=self.n_pre_proof
            ).float()

            p_post = F.one_hot(
                post_proof,
                num_classes=self.n_post_proof
            ).float()

            inputs.extend([p_pre, p_post])

        x = torch.cat(inputs, dim=1)

        return self.mlp(x).squeeze(1)

    def forward(self, pre_idx, post_idx, pre_proof=None, post_proof=None):
        return torch.sigmoid(
            self.forward_logits(pre_idx, post_idx, pre_proof, post_proof)
        )


class PaperDistanceModel(nn.Module):
    def __init__(
        self,
        n_neurons,
        dim=2,
        n_pre_proof=3,
        n_post_proof=3,
        use_proofreading=True,
        shared_embedding=True,
    ):
        super().__init__()

        self.dim = dim
        self.n_pre_proof = n_pre_proof
        self.n_post_proof = n_post_proof
        self.use_proofreading = use_proofreading
        self.shared_embedding = shared_embedding

        if shared_embedding:
            self.Z = nn.Embedding(n_neurons, dim)
            nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)
        else:
            self.Z_pre = nn.Embedding(n_neurons, dim)
            self.Z_post = nn.Embedding(n_neurons, dim)

            nn.init.normal_(self.Z_pre.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.Z_post.weight, mean=0.0, std=0.1)

        self.log_A = nn.Parameter(torch.tensor(0.0))
        self.log_C = nn.Parameter(torch.tensor(0.0))
        self.B = nn.Parameter(torch.tensor(-6.0))

        if self.use_proofreading:
            self.proof_linear = nn.Linear(
                n_pre_proof + n_post_proof,
                1,
                bias=False,
            )
            nn.init.zeros_(self.proof_linear.weight)

    def get_latent(self, pre_idx, post_idx):
        if self.shared_embedding:
            z_pre = self.Z(pre_idx)
            z_post = self.Z(post_idx)
        else:
            z_pre = self.Z_pre(pre_idx)
            z_post = self.Z_post(post_idx)

        return z_pre, z_post

    def forward_logits(self, pre_idx, post_idx, pre_proof=None, post_proof=None):
        z_pre, z_post = self.get_latent(pre_idx, post_idx)

        dist2 = ((z_pre - z_post) ** 2).sum(dim=1) / self.dim

        A = torch.exp(self.log_A)
        C = torch.exp(self.log_C) + 1e-6

        kernel = A * torch.exp(-dist2 / (C ** 2))

        logits = kernel + self.B

        if self.use_proofreading:
            if pre_proof is None or post_proof is None:
                raise ValueError(
                    "pre_proof and post_proof must be provided when use_proofreading=True"
                )

            p_pre = F.one_hot(
                pre_proof,
                num_classes=self.n_pre_proof
            ).float()

            p_post = F.one_hot(
                post_proof,
                num_classes=self.n_post_proof
            ).float()

            proof_input = torch.cat([p_pre, p_post], dim=1)
            proof_bias = self.proof_linear(proof_input).squeeze(1)

            logits = logits + proof_bias

        return logits

    def forward(self, pre_idx, post_idx, pre_proof=None, post_proof=None):
        return torch.sigmoid(
            self.forward_logits(pre_idx, post_idx, pre_proof, post_proof)
        )


def create_model(
    model_name,
    n_neurons,
    dim,
    n_pre_proof,
    n_post_proof,
    hidden=64,
    use_proofreading=True,
    shared_embedding=True,
):
    if model_name == "mlp":
        return MLPEmbeddingModel(
            n_neurons=n_neurons,
            dim=dim,
            n_pre_proof=n_pre_proof,
            n_post_proof=n_post_proof,
            hidden=hidden,
            use_proofreading=use_proofreading,
            shared_embedding=shared_embedding,
        )

    if model_name == "paper":
        return PaperDistanceModel(
            n_neurons=n_neurons,
            dim=dim,
            n_pre_proof=n_pre_proof,
            n_post_proof=n_post_proof,
            use_proofreading=use_proofreading,
            shared_embedding=shared_embedding,
        )

    raise ValueError(f"Unknown model_name: {model_name}")