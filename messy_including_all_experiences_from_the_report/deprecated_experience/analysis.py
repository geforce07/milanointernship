# analysis.py

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from checkpoints import load_checkpoint


# ============================================================
# Utilities
# ============================================================

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor_long(x):
    if torch.is_tensor(x):
        return x.long()
    return torch.as_tensor(x, dtype=torch.long)


def savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def compute_pair_distances(pre_idx, post_idx, all_ids, units):
    idx_to_id = np.array(all_ids)

    pre_idx_np = to_numpy(pre_idx)
    post_idx_np = to_numpy(post_idx)

    pre_ids = idx_to_id[pre_idx_np]
    post_ids = idx_to_id[post_idx_np]

    pos = units.set_index("pt_root_id")[
        ["pt_position_x", "pt_position_y", "pt_position_z"]
    ]

    pre_pos = pos.loc[pre_ids].values
    post_pos = pos.loc[post_ids].values

    return np.linalg.norm(pre_pos - post_pos, axis=1)


def predict_probabilities(
    model,
    pre_idx,
    post_idx,
    device,
    batch_size=32768,
    pre_proof=None,
    post_proof=None,
):
    model.eval()

    pre_idx = to_tensor_long(pre_idx)
    post_idx = to_tensor_long(post_idx)

    if pre_proof is not None:
        pre_proof = to_tensor_long(pre_proof)

    if post_proof is not None:
        post_proof = to_tensor_long(post_proof)

    probs = []

    with torch.no_grad():
        for start in range(0, len(pre_idx), batch_size):
            end = start + batch_size

            pi = pre_idx[start:end].to(device)
            pj = post_idx[start:end].to(device)

            if pre_proof is not None and post_proof is not None:
                pp = pre_proof[start:end].to(device)
                dp = post_proof[start:end].to(device)

                p = model(pi, pj, pp, dp)
            else:
                p = model(pi, pj)

            probs.append(
                p.detach().cpu().numpy()
            )

    return np.concatenate(probs)


# ============================================================
# Embeddings
# ============================================================

def extract_embedding_dataframe(model, all_ids, units):
    all_ids = np.array(all_ids)

    output = {}

    if hasattr(model, "Z"):
        Z = model.Z.weight.detach().cpu().numpy()

        df = pd.DataFrame(
            Z,
            columns=[f"z{k+1}" for k in range(Z.shape[1])]
        )

        df["neuron_idx"] = np.arange(len(df))
        df["pt_root_id"] = all_ids

        output["shared"] = df.merge(
            units,
            on="pt_root_id",
            how="left"
        )

    if hasattr(model, "Z_pre"):
        Z_pre = model.Z_pre.weight.detach().cpu().numpy()

        df_pre = pd.DataFrame(
            Z_pre,
            columns=[f"z_pre_{k+1}" for k in range(Z_pre.shape[1])]
        )

        df_pre["neuron_idx"] = np.arange(len(df_pre))
        df_pre["pt_root_id"] = all_ids

        output["pre"] = df_pre.merge(
            units,
            on="pt_root_id",
            how="left"
        )

    if hasattr(model, "Z_post"):
        Z_post = model.Z_post.weight.detach().cpu().numpy()

        df_post = pd.DataFrame(
            Z_post,
            columns=[f"z_post_{k+1}" for k in range(Z_post.shape[1])]
        )

        df_post["neuron_idx"] = np.arange(len(df_post))
        df_post["pt_root_id"] = all_ids

        output["post"] = df_post.merge(
            units,
            on="pt_root_id",
            how="left"
        )

    return output


# ============================================================
# Plots
# ============================================================

def plot_spatial_embedding(df, z_col, fig_path):
    df_plot = df.dropna(
        subset=[
            "pt_position_x",
            "pt_position_y",
            "pt_position_z",
            z_col,
        ]
    ).copy()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df_plot["pt_position_x"],
        df_plot["pt_position_y"],
        df_plot["pt_position_z"],
        c=df_plot[z_col],
        s=8,
        alpha=0.8,
        cmap="viridis",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Neuron positions colored by {z_col}")

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label(z_col)

    savefig(fig_path)


def plot_spatial_projections(df, z_col, fig_path):
    df_plot = df.dropna(
        subset=[
            "pt_position_x",
            "pt_position_y",
            "pt_position_z",
            z_col,
        ]
    ).copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        ("pt_position_x", "pt_position_y"),
        ("pt_position_x", "pt_position_z"),
        ("pt_position_y", "pt_position_z"),
    ]

    for ax, (a, b) in zip(axes, pairs):
        sc = ax.scatter(
            df_plot[a],
            df_plot[b],
            c=df_plot[z_col],
            s=8,
            alpha=0.8,
            cmap="viridis",
        )

        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{a} vs {b}, colored by {z_col}")
        ax.set_aspect("equal", adjustable="box")

    fig.colorbar(sc, ax=axes, label=z_col)

    savefig(fig_path)


def plot_connection_probability_by_distance(
    distance,
    y,
    p_pred,
    fig_path,
    n_bins=40,
):
    df = pd.DataFrame({
        "distance": distance,
        "y": to_numpy(y),
        "p_pred": p_pred,
    })

    df["bin"] = pd.cut(
        df["distance"],
        bins=n_bins
    )

    curve = (
        df.groupby("bin")
        .agg(
            empirical_prob=("y", "mean"),
            predicted_prob=("p_pred", "mean"),
            n_pairs=("y", "size"),
            n_connections=("y", "sum"),
        )
    )

    centers = np.array([
        interval.mid
        for interval in curve.index
    ])

    plt.figure(figsize=(8, 5))

    plt.plot(
        centers,
        curve["empirical_prob"],
        marker="o",
        label="Empirical P(connection)"
    )

    plt.plot(
        centers,
        curve["predicted_prob"],
        marker="o",
        label="Model predicted P(connection)"
    )

    plt.xlabel("Euclidean distance")
    plt.ylabel("Connection probability")
    plt.title("Empirical and predicted P(connection) vs distance")

    plt.legend()
    plt.grid(True)

    savefig(fig_path)

    return curve


def plot_latent_distance_vs_geo_distance(
    model,
    pre_idx,
    post_idx,
    geo_distance,
    y,
    fig_path,
    n_bins=30,
):
    pre_idx_np = to_numpy(pre_idx)
    post_idx_np = to_numpy(post_idx)

    with torch.no_grad():

        if hasattr(model, "Z"):
            Z = model.Z.weight.detach().cpu().numpy()

            z_pre = Z[pre_idx_np]
            z_post = Z[post_idx_np]

            label = "Mean |Z(pre) - Z(post)|"

        elif hasattr(model, "Z_pre") and hasattr(model, "Z_post"):
            Z_pre = model.Z_pre.weight.detach().cpu().numpy()
            Z_post = model.Z_post.weight.detach().cpu().numpy()

            z_pre = Z_pre[pre_idx_np]
            z_post = Z_post[post_idx_np]

            label = "Mean |Z_pre(pre) - Z_post(post)|"

        else:
            print("No latent embeddings found. Skipping latent distance plot.")
            return None

    latent_distance = np.linalg.norm(
        z_pre - z_post,
        axis=1,
    )

    df = pd.DataFrame({
        "geo_distance": geo_distance,
        "latent_distance": latent_distance,
        "y": to_numpy(y),
    })

    df["geo_bin"] = pd.cut(
        df["geo_distance"],
        bins=n_bins
    )

    curve = (
        df.groupby("geo_bin")
        .agg(
            mean_latent_distance=("latent_distance", "mean"),
            std_latent_distance=("latent_distance", "std"),
            n_pairs=("latent_distance", "size"),
            empirical_prob=("y", "mean"),
        )
    )

    centers = np.array([
        interval.mid
        for interval in curve.index
    ])

    plt.figure(figsize=(8, 5))

    plt.plot(
        centers,
        curve["mean_latent_distance"],
        marker="o",
        linewidth=2,
        label=label,
    )

    plt.xlabel("Euclidean distance")
    plt.ylabel("Mean latent distance")
    plt.title("Latent distance vs anatomical distance")

    plt.legend()
    plt.grid(True)

    savefig(fig_path)

    corr = np.corrcoef(
        geo_distance,
        latent_distance,
    )[0, 1]

    print(
        f"corr(geometrical distance, latent distance) = {corr:.3f}"
    )

    return curve

# ============================================================
# Main analysis function
# ============================================================

def run_analysis(
    model_factory,
    dim,
    checkpoint_path,
    output_dir,
    units,
    all_ids,
    pre_idx,
    post_idx,
    y,
    device,
    pre_proof=None,
    post_proof=None,
    batch_size=32768,
):
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    model = model_factory(dim).to(device)

    ckpt = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        device=device,
    )

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best epoch: {ckpt.get('best_epoch')}")
    print(f"Best val AUC: {ckpt.get('best_val_auc')}")

    geo_distance = compute_pair_distances(
        pre_idx,
        post_idx,
        all_ids,
        units,
    )

    p_pred = predict_probabilities(
        model,
        pre_idx,
        post_idx,
        device,
        batch_size=batch_size,
        pre_proof=pre_proof,
        post_proof=post_proof,
    )

    curve_prob = plot_connection_probability_by_distance(
        geo_distance,
        y,
        p_pred,
        fig_path=figure_dir / f"connection_probability_dim_{dim}.png",
    )

    curve_prob.to_csv(
        table_dir / f"connection_probability_dim_{dim}.csv"
    )

    curve_latent = plot_latent_distance_vs_geo_distance(
        model,
        pre_idx,
        post_idx,
        geo_distance,
        y,
        fig_path=figure_dir / f"latent_vs_geo_distance_dim_{dim}.png",
    )

    if curve_latent is not None:
        curve_latent.to_csv(
            table_dir / f"latent_vs_geo_distance_dim_{dim}.csv"
        )

    embedding_dfs = extract_embedding_dataframe(
        model,
        all_ids,
        units,
    )

    for name, df in embedding_dfs.items():
        df.to_csv(
            table_dir / f"embeddings_{name}_dim_{dim}.csv",
            index=False,
        )

        z_cols = [
            c for c in df.columns
            if c.startswith("z")
        ]

        if len(z_cols) == 0:
            continue

        z_col = z_cols[0]

        plot_spatial_embedding(
            df,
            z_col,
            figure_dir / f"spatial_3d_{name}_{z_col}_dim_{dim}.png",
        )

        plot_spatial_projections(
            df,
            z_col,
            figure_dir / f"spatial_projections_{name}_{z_col}_dim_{dim}.png",
        )

    return {
        "checkpoint": str(checkpoint_path),
        "dim": dim,
        "figures": str(figure_dir),
        "tables": str(table_dir),
    }