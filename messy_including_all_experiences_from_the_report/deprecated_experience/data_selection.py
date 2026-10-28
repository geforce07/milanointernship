# data_selection.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load + clean
# ============================================================

def load_units(path="units_L234.csv"):

    units = pd.read_csv(path).copy()

    mask_layer = (
        (units["layer"] == "L2/3")
        |
        (units["layer"] == "L4")
    )

    mask_area = (
        units["brain_area"] == "V1"
    )

    units = units[
        mask_layer & mask_area
    ].copy()

    units = units.drop(
        columns=[
            "pvals_ori_fdr",
            "pvals_dir_fdr",
            "brain_area",
        ],
        errors="ignore"
    )

    units["pt_root_id"] = units["pt_root_id"].astype(np.int64)

    for c in [
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
    ]:
        units[c] = units[c].astype(np.float32)

    return units.reset_index(drop=True)

# ============================================================
# Proofreading filter
# ============================================================

def select_proofreading(
    units,
    axon_mode="include",
    dendrite_mode="include",
):
    """
    include : garde tout
    only    : garde uniquement les proofread
    exclude : garde uniquement les non-proofread
    """

    data = units.copy()

    if axon_mode == "only":
        data = data[
            data["strategy_axon"] != "none"
        ]

    elif axon_mode == "exclude":
        data = data[
            data["strategy_axon"] == "none"
        ]

    if dendrite_mode == "only":
        data = data[
            data["strategy_dendrite"] != "none"
        ]

    elif dendrite_mode == "exclude":
        data = data[
            data["strategy_dendrite"] == "none"
        ]

    return data.reset_index(drop=True)

# ============================================================
# Sampling
# ============================================================

def sample_neurons(
    units,
    mode="global",
    n=None,
    seed=42,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    save_path=None
):

    data = units.copy()

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    if mode == "global":

        selected = data.copy()

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    elif mode == "window":

        if x_min is not None:
            data = data[data["pt_position_x"] >= x_min]

        if x_max is not None:
            data = data[data["pt_position_x"] <= x_max]

        if y_min is not None:
            data = data[data["pt_position_y"] >= y_min]

        if y_max is not None:
            data = data[data["pt_position_y"] <= y_max]

        if z_min is not None:
            data = data[data["pt_position_z"] >= z_min]

        if z_max is not None:
            data = data[data["pt_position_z"] <= z_max]

        selected = data.copy()

    # --------------------------------------------------------
    # Random
    # --------------------------------------------------------

    elif mode == "random":

        if n is None:
            raise ValueError("n must be specified")

        n = min(n, len(data))

        selected = data.sample(
            n=n,
            random_state=seed
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    selected = selected.reset_index(drop=True)


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------

    proofread_mask = (
        (units["strategy_axon"] != "none")
        |
        (units["strategy_dendrite"] != "none")
    )

    proofread = units[proofread_mask]

    selected_ids = set(
        selected["pt_root_id"]
    )

    selected_mask = (
        units["pt_root_id"]
        .isin(selected_ids)
    )

    plt.figure(figsize=(10, 8))

    # 1. Tout le connectome
    plt.scatter(
        units["pt_position_x"],
        units["pt_position_z"],
        s=3,
        alpha=0.6,
        color="lightgray",
        label=f"All neurons ({len(units)})"
    )

    # 2. Neurones proofread
    plt.scatter(
        proofread["pt_position_x"],
        proofread["pt_position_z"],
        s=8,
        alpha=0.6,
        color="tab:green",
        label=f"Proofread ({len(proofread)})"
    )

    # 3. Neurones finalement sélectionnés
    plt.scatter(
        units.loc[selected_mask, "pt_position_x"],
        units.loc[selected_mask, "pt_position_z"],
        s=12,
        alpha=0.9,
        color="tab:blue",
        label=f"Selected ({len(selected)})"
    )

    plt.xlabel("x")
    plt.ylabel("z")

    plt.title(
        f"Neuron selection ({mode})"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()
    plt.close()
    
    return selected
    
"""units = sample_neurons(
    units,
    mode="window",
    x_min=500000,
    x_max=700000,
    y_min=100000,
    y_max=300000,
)"""


def make_populations(units):
    mask_exc = units["classification_system"] == "excitatory_neuron"
    mask_inh = units["classification_system"] == "inhibitory_neuron"

    mask_l23 = units["layer"] == "L2/3"
    mask_l4 = units["layer"] == "L4"

    return {
        "E_L23": units[mask_exc & mask_l23].copy().reset_index(drop=True),
        "I_L23": units[mask_inh & mask_l23].copy().reset_index(drop=True),
        "E_L4": units[mask_exc & mask_l4].copy().reset_index(drop=True),
        "I_L4": units[mask_inh & mask_l4].copy().reset_index(drop=True),
    }


def get_pathway(populations, pathway):
    E_L23 = populations["E_L23"]
    I_L23 = populations["I_L23"]
    E_L4 = populations["E_L4"]
    I_L4 = populations["I_L4"]

    pathways = {
        "E23_E23": (E_L23, E_L23),
        "I23_E23": (I_L23, E_L23),
        "E4_E23": (E_L4, E_L23),
        "I4_E23": (I_L4, E_L23),
        "E23_I23": (E_L23, I_L23),
        "I23_I23": (I_L23, I_L23),
        "E4_I23": (E_L4, I_L23),
        "I4_I23": (I_L4, I_L23),
    }

    if pathway not in pathways:
        raise ValueError(f"Unknown pathway: {pathway}")

    return pathways[pathway]

def load_synapses(path="synapses_L234.csv"):
    synapses = pd.read_csv(path).copy()
    return synapses

def select_neurons(
    mode="global",
    n=None,
    seed=42,
    axon_mode="include",
    dendrite_mode="include",
    save_path=None,
    **kwargs,
):
    units = load_units() #Layer : L2/3-L4, Area : V1

    units = select_proofreading(
        units,
        axon_mode=axon_mode,
        dendrite_mode=dendrite_mode,
    )

    units = sample_neurons(
        units,
        mode=mode,
        n=n,
        seed=seed,
        save_path=save_path,
        **kwargs,
    )

    return units