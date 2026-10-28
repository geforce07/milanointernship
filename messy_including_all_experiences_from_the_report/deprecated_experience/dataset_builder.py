# dataset_builder.py

import numpy as np
import pandas as pd


# ============================================================
# Orientation helpers
# ============================================================

def circular_diff_ori(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 180 - diff)


def polar_distance(a, b):
    diff = circular_diff_ori(a, b)
    return np.sin(np.deg2rad(diff))


# ============================================================
# Utilities
# ============================================================

def make_id_mapping(pre_ids, post_ids):
    all_ids = pd.Index(
        pd.concat([
            pd.Series(pre_ids),
            pd.Series(post_ids),
        ]).unique()
    )

    id_map = {
        id_: i
        for i, id_ in enumerate(all_ids)
    }

    return all_ids, id_map


def encode_proofreading(pre_pop, post_pop):
    axon_categories = sorted(
        pre_pop["strategy_axon"].astype(str).unique()
    )

    dend_categories = sorted(
        post_pop["strategy_dendrite"].astype(str).unique()
    )

    axon_map = {
        v: i
        for i, v in enumerate(axon_categories)
    }

    dend_map = {
        v: i
        for i, v in enumerate(dend_categories)
    }

    pre_proof_neuron = (
        pre_pop["strategy_axon"]
        .astype(str)
        .map(axon_map)
        .to_numpy()
    )

    post_proof_neuron = (
        post_pop["strategy_dendrite"]
        .astype(str)
        .map(dend_map)
        .to_numpy()
    )

    return pre_proof_neuron, post_proof_neuron, axon_map, dend_map

# ============================================================
# Full-pair dataset for embedding models
# ============================================================

def build_full_pair_dataset(
    pre_pop,
    post_pop,
    synapses,
):
    pre_pop = pre_pop.reset_index(drop=True)
    post_pop = post_pop.reset_index(drop=True)

    pre_ids = pre_pop["pt_root_id"].to_numpy()
    post_ids = post_pop["pt_root_id"].to_numpy()

    n_pre = len(pre_ids)
    n_post = len(post_ids)

    print("n_pre:", n_pre)
    print("n_post:", n_post)
    print("full pairs before self-removal:", n_pre * n_post)

    # --------------------------------------------------------
    # Cartesian product
    # --------------------------------------------------------

    pre_pos = np.repeat(
        np.arange(n_pre),
        n_post
    )

    post_pos = np.tile(
        np.arange(n_post),
        n_pre
    )

    pair_pre_ids = pre_ids[pre_pos]
    pair_post_ids = post_ids[post_pos]

    # --------------------------------------------------------
    # Remove self-connections
    # --------------------------------------------------------

    mask_not_self = (
        pair_pre_ids != pair_post_ids
    )

    pair_pre_ids = pair_pre_ids[mask_not_self]
    pair_post_ids = pair_post_ids[mask_not_self]

    pre_pos = pre_pos[mask_not_self]
    post_pos = post_pos[mask_not_self]

    # --------------------------------------------------------
    # Positive connections
    # --------------------------------------------------------

    syn_sub = synapses[
        synapses["pre_pt_root_id"].isin(pre_ids)
        &
        synapses["post_pt_root_id"].isin(post_ids)
    ][[
        "pre_pt_root_id",
        "post_pt_root_id",
    ]].drop_duplicates()

    syn_sub = syn_sub[
        syn_sub["pre_pt_root_id"]
        != syn_sub["post_pt_root_id"]
    ]

    positive_set = set(
        zip(
            syn_sub["pre_pt_root_id"].to_numpy(),
            syn_sub["post_pt_root_id"].to_numpy(),
        )
    )

    y = np.fromiter(
        (
            (a, b) in positive_set
            for a, b in zip(pair_pre_ids, pair_post_ids)
        ),
        dtype=np.bool_,
        count=len(pair_pre_ids),
    ).astype(np.float32)

    # --------------------------------------------------------
    # Id mapping
    # --------------------------------------------------------

    all_ids, id_map = make_id_mapping(
        pre_ids,
        post_ids,
    )

    pre_idx = np.array(
        [id_map[i] for i in pair_pre_ids],
        dtype=np.int64,
    )

    post_idx = np.array(
        [id_map[i] for i in pair_post_ids],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Proofreading
    # --------------------------------------------------------

    (
        pre_proof_neuron,
        post_proof_neuron,
        axon_map,
        dend_map,
    ) = encode_proofreading(pre_pop, post_pop)

    pre_proof = pre_proof_neuron[pre_pos].astype(np.int64)
    post_proof = post_proof_neuron[post_pos].astype(np.int64)

    pre_xyz = pre_pop[
    ["pt_position_x", "pt_position_y", "pt_position_z"]
].to_numpy()

    post_xyz = post_pop[
        ["pt_position_x", "pt_position_y", "pt_position_z"]
    ].to_numpy()

    distances = np.linalg.norm(
        pre_xyz[pre_pos] - post_xyz[post_pos],
        axis=1,
    ).astype(np.float32)

    print("n_pairs:", len(y))
    print("n_positive:", int(y.sum()))
    print("positive rate:", y.mean())
    print("axon_map:", axon_map)
    print("dend_map:", dend_map)

    return {
        "y": y,
        "pre_idx": pre_idx,
        "post_idx": post_idx,
        "pre_proof": pre_proof,
        "post_proof": post_proof,
        "all_ids": all_ids,
        "id_map": id_map,
        "distances": distances,
        "axon_map": axon_map,
        "dend_map": dend_map,
    }