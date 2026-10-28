MODEL_NAME = "mlp"
LOSS_NAME = "distance_weighted_bce"   # ou "bce" si tu veux tester sans pondération

DIMS = [1,2,3]

LR = 1e-3
N_EPOCHS = 30
PATIENCE = 5
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

USE_PROOFREADING = False

SELECTION_MODE = "window"   # downsampling aléatoire
N_NEURONS = 1000           # ou 2000 si ton PC suit

AXON_PROOFREAD = "exclude"
DENDRITE_PROOFREAD = "exclude"

PATHWAY = "E23_E23"

BATCH_SIZE = 32768
SHARED_EMBEDDING = True

DIM_TO_ANALYZE = 1

from data_selection import load_synapses

synapses = load_synapses()

from pathlib import Path
import torch

from config import *

from data_selection import (
    select_neurons,
    make_populations,
    get_pathway
)

from dataset_builder import build_full_pair_dataset

from dataloaders import build_dataloaders

from models import (
    create_model,
)

from train_loop import (
    run_training_for_dims,
)

from analysis import (
    run_analysis,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

experiment_name = (
    f"{MODEL_NAME}"
    f"_loss_{LOSS_NAME}"
    f"_proof_{USE_PROOFREADING}"
    f"_shared_{SHARED_EMBEDDING}"
    f"_sampling_{SELECTION_MODE}"
    f"_n_{N_NEURONS}"
)

OUTPUT_DIR = (
    Path("results")
    / experiment_name
)

units_selected = select_neurons(
    mode=SELECTION_MODE,
    n=N_NEURONS,
    save_path= OUTPUT_DIR / "figures" / "data_selection.png",
    z_max=800,
    x_min=700,
)

pathway = make_populations(units_selected)

pre, post = get_pathway(pathway, "E23_E23")


syn = load_synapses()


dataset_dict = build_full_pair_dataset(
    pre_pop=pre,
    post_pop=post,
    synapses=syn
)

train_loader, val_loader, test_loader, train_set, val_set, test_set = build_dataloaders(
    dataset_dict,
    batch_size=BATCH_SIZE,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    seed=SEED,
)



def model_factory(dim):

    return create_model(
        model_name=MODEL_NAME,
        n_neurons=len(dataset_dict["all_ids"]),
        dim=dim,
        n_pre_proof=len(dataset_dict["axon_map"]),
        n_post_proof=len(dataset_dict["dend_map"]),
        use_proofreading=USE_PROOFREADING,
    )

dist_bins = torch.linspace(
    0,
    dataset_dict["distances"].max(),
    40
)

results = run_training_for_dims(
    dims=DIMS,

    model_factory=model_factory,

    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,

    train_set=train_set,

    y=dataset_dict["y"],

    output_dir=OUTPUT_DIR,

    device=DEVICE,

    lr=LR,
    n_epochs=N_EPOCHS,
    patience=PATIENCE,

    loss_name=LOSS_NAME,

    distances=dataset_dict["distances"],
    dist_bins=dist_bins
)

for dim in DIMS:
    run_analysis(
        model_factory=model_factory,

        dim=dim,

        checkpoint_path=(
            OUTPUT_DIR
            / "checkpoints"
            / f"best_dim_{dim}.pt"
        ),

        output_dir=OUTPUT_DIR,

        units=units_selected,

        all_ids=dataset_dict["all_ids"],

        pre_idx=dataset_dict["pre_idx"],
        post_idx=dataset_dict["post_idx"],

        y=dataset_dict["y"],

        pre_proof=dataset_dict["pre_proof"],
        post_proof=dataset_dict["post_proof"],

        device=DEVICE,
    )