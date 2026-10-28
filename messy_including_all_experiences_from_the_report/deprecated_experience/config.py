from dataclasses import dataclass

@dataclass
class Config:
    # Experiment
    experiment_name: str = "mlp_full_pairs_E23_E23_1000_weighted"
    model_type: str = "mlp"  # "mlp" or "paper"

    # Data
    units_path: str = "units_L234.csv"
    synapses_path: str = "synapses_L234.csv"
    pathway: str = "E23_E23"
    sampling_type: str = "random_neurons"
    n_neurons_sample: int = 1000
    seed: int = 42

    # Model
    dims: tuple = (1, 2, 3)
    proof_dim: int = 2
    hidden: int = 64

    # Training
    batch_size: int = 32768
    n_epochs: int = 50
    patience: int = 10
    lr: float = 1e-3

    # Loss
    loss_type: str = "weighted_bce"  # "bce", "weighted_bce", "soft_weighted_bce"
    alpha_weight: float = 1.0        # 1.0 = full weighting, 0.5 = sqrt weighting

    # Split
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1