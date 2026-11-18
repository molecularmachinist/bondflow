import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="whitegrid")


# ==========================================
# 1. Helper Functions (Data Alignment)
# ==========================================

def _detect_frame_column(shap_df):
    candidates = {"frame", "frames", "frame_index", "frame_idx"}
    for col in shap_df.columns:
        if str(col).lower() in candidates:
            return col
    return None


def build_frame_to_cluster_map(shap_df, frames):
    if "Cluster" not in shap_df.columns:
        raise ValueError("shap_df must contain a 'Cluster' column.")

    frames = np.asarray(frames)
    frame_col = _detect_frame_column(shap_df)

    if frame_col is not None:
        mapping = dict(zip(shap_df[frame_col], shap_df["Cluster"]))
        if any(frame in mapping for frame in frames):
            return mapping

    mapping = dict(zip(shap_df.index, shap_df["Cluster"]))
    if any(frame in mapping for frame in frames):
        return mapping

    if len(shap_df) == len(frames):
        return dict(zip(frames, shap_df["Cluster"]))

    raise ValueError("Could not match shap_df rows to trajectory frames.")


def _align_y(Y, dist_df):
    if isinstance(Y, pd.Series):
        if Y.index.equals(dist_df.index):
            return Y.to_numpy()
        y_aligned = Y.reindex(dist_df.index)
        if y_aligned.notna().any():
            return y_aligned.to_numpy()
        if len(Y) == len(dist_df):
            return Y.to_numpy()
        raise ValueError("Could not align Y with dist_df frames.")
    
    y_values = np.asarray(Y)
    if len(y_values) != len(dist_df):
        raise ValueError("Y must have the same number of values as dist_df rows.")
    return y_values


# ==========================================
# 2. Single-Axis Drawing Helpers
# ==========================================

def _draw_kde_on_ax(ax, df_sub, title, points=True, bw_adjust=1.0, kde_kwargs=None, scatter_kwargs=None):
    kde_kwargs = kde_kwargs or {"fill": True, "levels": 15, "thresh": 0.05, "cmap": "viridis"}
    scatter_kwargs = scatter_kwargs or {"s": 8, "color": "black", "alpha": 0.25}

    if df_sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    can_draw_kde = (
        len(df_sub) >= 3
        and df_sub["Y"].nunique() > 1
        and df_sub["Distance"].nunique() > 1
    )

    if can_draw_kde:
        sns.kdeplot(data=df_sub, x="Y", y="Distance", ax=ax, bw_adjust=bw_adjust, **kde_kwargs)

    if points:
        ax.scatter(df_sub["Y"], df_sub["Distance"], **scatter_kwargs)

    ax.set_xlabel("Function value")
    ax.set_ylabel("Distance")
    ax.set_title(title)


def _draw_timeseries_on_ax(
    ax1, fig, plot_df, title,
    frame_col="Frame", function_col="Function Value", distance_col="Scaled Distance",
    cluster_col="Cluster", show_clusters=True, connect_points=True, show_colorbar=True
):
    if plot_df.empty:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(title)
        return

    plot_df = plot_df.sort_values(frame_col)
    x = plot_df[frame_col]
    y_func = plot_df[function_col]
    y_dist = plot_df[distance_col]

    # Left axis: Function Value
    if connect_points:
        ax1.plot(x, y_func, color="black", linewidth=1.6, label=function_col)
    else:
        ax1.scatter(x, y_func, color="black", s=18, label=function_col)

    ax1.set_xlabel("Frame")
    ax1.set_ylabel(function_col, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Optional Cluster Coloring
    if show_clusters and cluster_col in plot_df.columns:
        valid = plot_df[cluster_col].notna()
        cluster_values = plot_df.loc[valid, cluster_col]
        cluster_ids = list(pd.unique(cluster_values))

        if cluster_ids:
            cluster_to_code = {c: i for i, c in enumerate(cluster_ids)}
            codes = cluster_values.map(cluster_to_code)
            cmap_name = "tab10" if len(cluster_ids) <= 10 else "tab20"
            
            sc = ax1.scatter(
                plot_df.loc[valid, frame_col],
                plot_df.loc[valid, function_col],
                c=codes, cmap=cmap_name,
                vmin=-0.5, vmax=len(cluster_ids) - 0.5,
                s=18, alpha=0.8, zorder=3
            )
            if show_colorbar:
                cbar = fig.colorbar(sc, ax=ax1, pad=0.08)
                cbar.set_label("Cluster")
                cbar.set_ticks(range(len(cluster_ids)))
                cbar.set_ticklabels([str(c) for c in cluster_ids])

    # Right axis: Distance
    ax2 = ax1.twinx()
    if connect_points:
        ax2.plot(x, y_dist, color="C1", linewidth=1.4, alpha=0.8, label=distance_col)
    else:
        ax2.scatter(x, y_dist, color="C1", s=18, alpha=0.8, label=distance_col)

    ax2.set_ylabel(distance_col, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title(title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")


# ==========================================
# 3. Main Unified Plotting Function
# ==========================================

def plot_all_interaction_views(
    dist_df,
    df,
    interactions_df,
    Y,
    shap_df,
    trajectory=None,
    clusters_to_plot="all",
    interaction_col="Interaction",
    frame_col="Frame",
    function_col="Function Value",
    distance_col="Scaled Distance",
    cluster_col="Cluster",
    bw_adjust=1.0,
    points=True,
    connect_cluster_ts_points=False,
):
    """
    Creates one unified dashboard figure per interaction containing:
      - Row 1: Overall 2D KDE (left) | Overall Time-Series (right)
      - Row 2: 2D KDE per cluster (side-by-side)
      - Row 3: Time-Series per cluster (side-by-side)
    """
    interactions = (
        interactions_df[interaction_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not interactions:
        raise ValueError(f"No interaction names found in '{interaction_col}'.")

    # Align Y and Clusters for KDE data
    y_values = _align_y(Y, dist_df)
    frames = dist_df.index.to_numpy()
    frame_to_cluster = build_frame_to_cluster_map(shap_df, frames)
    cluster_values = pd.Series(frames).map(frame_to_cluster).to_numpy()

    # Determine clusters to plot
    available_clusters = list(pd.unique(shap_df["Cluster"].dropna()))
    if isinstance(clusters_to_plot, str) and clusters_to_plot.lower() == "all":
        clusters = available_clusters
    else:
        clusters = list(clusters_to_plot)
    n_clusters = max(1, len(clusters))

    # Filter time-series dataframe if trajectory is specified
    ts_data = df.copy()
    if trajectory is not None:
        if "Trajectory" not in ts_data.columns:
            raise ValueError("Trajectory specified, but 'Trajectory' column not in df.")
        ts_data = ts_data[ts_data["Trajectory"].astype(str) == str(trajectory)].copy()

    figures = {}

    for interaction in interactions:
        if interaction not in dist_df.columns:
            print(f"Skipping '{interaction}': not found in dist_df columns.")
            continue

        # Prepare KDE dataframe for this interaction
        kde_df = pd.DataFrame({
            "Y": y_values,
            "Distance": dist_df[interaction].to_numpy(),
            "Cluster": cluster_values
        }).dropna(subset=["Y", "Distance"])

        # Prepare Time-Series dataframe for this interaction
        ts_sub = ts_data[ts_data[interaction_col].astype(str) == interaction].copy()

        # Build GridSpec layout: 3 rows
        # Use least common multiple of 2 (top row) and n_clusters (bottom 2 rows) for clean alignment
        total_cols = math.lcm(2, n_clusters)
        half_cols = total_cols // 2
        cluster_span = total_cols // n_clusters

        fig = plt.figure(figsize=(max(14, 5.5 * n_clusters), 14))
        gs = gridspec.GridSpec(3, total_cols, figure=fig, hspace=0.4, wspace=0.35)

        # --- ROW 1: Overall Plots ---
        ax_kde_all = fig.add_subplot(gs[0, :half_cols])
        _draw_kde_on_ax(
            ax_kde_all,
            kde_df,
            title="Overall 2D KDE",
            points=points,
            bw_adjust=bw_adjust
        )

        ax_ts_all = fig.add_subplot(gs[0, half_cols:])
        _draw_timeseries_on_ax(
            ax_ts_all, fig, ts_sub,
            title="Overall Function & Distance Over Time",
            frame_col=frame_col, function_col=function_col,
            distance_col=distance_col, cluster_col=cluster_col,
            show_clusters=True, connect_points=True, show_colorbar=True
        )

        # --- ROW 2 & 3: Per-Cluster Plots ---
        for idx, cluster in enumerate(clusters):
            c_start = idx * cluster_span
            c_end = (idx + 1) * cluster_span

            # Row 2: Cluster KDE
            ax_kde_c = fig.add_subplot(gs[1, c_start:c_end])
            kde_cluster_sub = kde_df[kde_df["Cluster"] == cluster]
            _draw_kde_on_ax(
                ax_kde_c,
                kde_cluster_sub,
                title=f"KDE — Cluster {cluster}",
                points=points,
                bw_adjust=bw_adjust
            )

            # Row 3: Cluster Time-Series
            ax_ts_c = fig.add_subplot(gs[2, c_start:c_end])
            ts_cluster_sub = ts_sub[ts_sub[cluster_col] == cluster] if cluster_col in ts_sub.columns else pd.DataFrame()
            _draw_timeseries_on_ax(
                ax_ts_c, fig, ts_cluster_sub,
                title=f"Time-Series — Cluster {cluster}",
                frame_col=frame_col, function_col=function_col,
                distance_col=distance_col, cluster_col=cluster_col,
                show_clusters=False,
                connect_points=connect_cluster_ts_points,
                show_colorbar=False
            )

        main_title = f"Interaction Analysis: {interaction}"
        if trajectory is not None:
            main_title += f" | Trajectory: {trajectory}"
        fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98)

        plt.show()
        figures[interaction] = fig

    return figures