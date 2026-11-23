
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="whitegrid")

def plot_salt_bridge_distributions(df, bins=30, col_wrap=4, xlim_min = 0, xlim_max = 35, show=False):

    salt_bridges = df.columns

    n_bridges = len(salt_bridges)
    n_cols = col_wrap
    n_rows = int(np.ceil(n_bridges / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for i, salt_bridge in enumerate(salt_bridges):

        sns.histplot(
            df[salt_bridge].dropna(),
            bins=bins,
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(salt_bridge)
        axes[i].set_xlabel("Distance")
        axes[i].set_ylabel("Count")
        axes[i].set_xlim(xlim_min,xlim_max)
        

    # Remove unused axes
    for i in range(n_bridges, len(axes)):
        axes[i].remove()

    fig.tight_layout()

    if show:
        plt.show()

    else:
        plt.close(fig)

    return fig

def plot_summary_bars_from_subset(
    subset_df,
    stat="mean",
    clusters=(0, 1),
    interaction_col="Interaction",
    top_n=20,
    title=None,
    figsize=None,
    show=False,
):
    """Plot absolute central SHAP values for interactions in a selected dataframe."""

    c0, c1 = clusters
    signed0_col = f"{stat}_c{c0}"
    signed1_col = f"{stat}_c{c1}"
    abs0_col = f"abs_{stat}_c{c0}"
    abs1_col = f"abs_{stat}_c{c1}"

    needed = [interaction_col, signed0_col, signed1_col, abs0_col, abs1_col]
    missing = [col for col in needed if col not in subset_df.columns]
    if missing:
        raise ValueError(f"Missing columns in subset_df: {missing}")

    plot_df = subset_df.head(top_n).copy()

    if plot_df.empty:
        print("No interactions to plot.")
        return None

    labels = plot_df[interaction_col].astype(str).to_numpy()
    abs0 = plot_df[abs0_col].to_numpy()
    abs1 = plot_df[abs1_col].to_numpy()
    signed0 = plot_df[signed0_col].to_numpy()
    signed1 = plot_df[signed1_col].to_numpy()

    y = np.arange(len(plot_df))
    height = 0.38

    if figsize is None:
        figsize = (10, max(4, 0.45 * len(plot_df)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        y - height / 2, abs0, height=height,
        color="C0", alpha=0.7, label=f"Cluster {c0}"
    )
    ax.barh(
        y + height / 2, abs1, height=height,
        color="C1", alpha=0.7, label=f"Cluster {c1}"
    )

    # Marker colour indicates the sign of the mean/median:
    # red = positive, blue = negative, grey = zero.
    for i, value in enumerate(signed0):
        if np.isfinite(value):
            marker_color = "red" if value > 0 else "blue" if value < 0 else "grey"
            ax.scatter(
                abs(value), y[i] - height / 2,
                color=marker_color, edgecolor="black", zorder=5
            )

    for i, value in enumerate(signed1):
        if np.isfinite(value):
            marker_color = "red" if value > 0 else "blue" if value < 0 else "grey"
            ax.scatter(
                abs(value), y[i] + height / 2,
                color=marker_color, edgecolor="black", zorder=5
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Absolute {stat} SHAP value")
    ax.set_ylabel("Interaction")

    if title is None:
        title = f"Top {len(plot_df)} interactions"

    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    if show:
        plt.show()
    
    else:
        plt.close(fig)
    
    return fig
   

def plot_distributions_from_subset(
    raw_df,
    subset_df,
    value_col="SHAP Value",
    interaction_col="Interaction",
    cluster_col="Cluster",
    clusters=(0, 1),
    top_n=9,
    kind="kde",
    use_abs=False,
    ncols=3,
    bw_adjust=1.0,
    show=False,
):
    """Plot SHAP distributions for interactions listed in subset_df."""

    if interaction_col not in subset_df.columns:
        raise ValueError(f"'{interaction_col}' is missing from subset_df.")

    required = [interaction_col, cluster_col, value_col]
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw_df: {missing}")

    interactions = subset_df[interaction_col].head(top_n).astype(str).tolist()

    if not interactions:
        print("No interactions to plot.")
        return None

    plot_df = raw_df.copy()
    plot_df["_interaction_key"] = plot_df[interaction_col].astype(str)
    plot_df = plot_df[
        plot_df["_interaction_key"].isin(interactions)
        & plot_df[cluster_col].isin(clusters)
    ].copy()

    if use_abs:
        plot_df["_plot_value"] = plot_df[value_col].abs()
        x_col = "_plot_value"
        x_label = f"Absolute {value_col}"
    else:
        x_col = value_col
        x_label = value_col

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(interactions) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, interaction in enumerate(interactions):
        ax = axes[i]
        feature_data = plot_df[plot_df["_interaction_key"] == interaction]

        for j, cluster in enumerate(clusters):
            values = feature_data.loc[
                feature_data[cluster_col] == cluster, x_col
            ].dropna()

            color = f"C{j}"

            if kind == "kde":
                if len(values) > 1 and values.nunique() > 1:
                    sns.kdeplot(
                        x=values,
                        ax=ax,
                        label=f"Cluster {cluster}",
                        color=color,
                        bw_adjust=bw_adjust,
                    )
                elif len(values) > 0:
                    ax.axvline(
                        values.iloc[0],
                        color=color,
                        label=f"Cluster {cluster}",
                    )

            elif kind == "hist":
                if len(values) > 0:
                    sns.histplot(
                        x=values,
                        ax=ax,
                        label=f"Cluster {cluster}",
                        color=color,
                        stat="density",
                        bins=30,
                        alpha=0.4,
                    )
            else:
                raise ValueError("kind must be 'kde' or 'hist'.")

            if len(values) > 0:
                ax.axvline(
                    values.mean(),
                    color=color,
                    linestyle="--",
                    alpha=0.8,
                )

        ax.set_title(interaction)
        ax.set_xlabel(x_label)

        if i == 0:
            ax.legend()

    for ax in axes[len(interactions):]:
        ax.set_visible(False)

    fig.tight_layout()

    if show:
        plt.show()
    
    else:
        plt.close(fig)

    return fig
    


# ==========================================
# 1. Single-Axis Drawing Helpers
# ==========================================

def _draw_kde_on_ax(
    ax,
    df_sub,
    title,
    function_col="Function Value",
    distance_col="Scaled Distance",
    points=True,
    point_size=5,
    point_alpha=0.08,
    bw_adjust=1.0,
    x_lim=None,
    y_lim=None,
    kde_kwargs=None,
    scatter_kwargs=None,
):
    kde_kwargs = kde_kwargs or {"fill": True, "levels": 15, "thresh": 0.05, "cmap": "viridis"}
    base_scatter = {
        "s": point_size,
        "color": "black",
        "alpha": point_alpha,
        "edgecolors": "none",
    }
    if scatter_kwargs:
        base_scatter.update(scatter_kwargs)

    valid_df = df_sub[[function_col, distance_col]].dropna()

    if valid_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        return

    can_draw_kde = (
        len(valid_df) >= 3
        and valid_df[function_col].nunique() > 1
        and valid_df[distance_col].nunique() > 1
    )

    if can_draw_kde:
        sns.kdeplot(
            data=valid_df,
            x=function_col,
            y=distance_col,
            ax=ax,
            bw_adjust=bw_adjust,
            **kde_kwargs,
        )

    if points:
        ax.scatter(valid_df[function_col], valid_df[distance_col], **base_scatter)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel(function_col)
    ax.set_ylabel(distance_col)
    ax.set_title(title)


def _draw_timeseries_on_ax(
    ax1,
    fig,
    plot_df,
    title,
    frame_col="Frame",
    function_col="Function Value",
    distance_col="Scaled Distance",
    cluster_col="Cluster",
    show_clusters=False,
    connect_points=True,
    show_points=False,
    x_lim=None,
    y_func_lim=None,
    y_dist_lim=None,
):
    ax2 = ax1.twinx()

    if plot_df.empty:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(title)
    else:
        plot_df = plot_df.sort_values(frame_col)
        x = plot_df[frame_col]
        y_func = plot_df[function_col]
        y_dist = plot_df[distance_col]

        # Left axis: Function Value
        if connect_points:
            ax1.plot(
                x,
                y_func,
                color="black",
                linewidth=1.6,
                marker="o" if show_points else None,
                markersize=3.5 if show_points else 0,
                label=function_col,
                zorder=3,
            )
        elif show_points:
            ax1.scatter(x, y_func, color="black", s=18, alpha=0.85, label=function_col, zorder=3)

        # Optional cluster-colored dots + colorbar
        if show_clusters and cluster_col in plot_df.columns:
            valid = plot_df[cluster_col].notna()
            cluster_values = plot_df.loc[valid, cluster_col]
            cluster_ids = list(pd.unique(cluster_values))

            if cluster_ids:
                cluster_to_code = {c: i for i, c in enumerate(cluster_ids)}
                codes = cluster_values.map(cluster_to_code)
                cmap_name = "Set1" if len(cluster_ids) <= 9 else "tab20"

                sc = ax1.scatter(
                    plot_df.loc[valid, frame_col],
                    plot_df.loc[valid, function_col],
                    c=codes,
                    cmap=cmap_name,
                    vmin=-0.5,
                    vmax=len(cluster_ids) - 0.5,
                    s=18,
                    alpha=0.8,
                    zorder=4,
                )
                cbar = fig.colorbar(sc, ax=ax1, pad=0.08)
                cbar.set_label(cluster_col)
                cbar.set_ticks(range(len(cluster_ids)))
                cbar.set_ticklabels([str(c) for c in cluster_ids])

        # Right axis: Distance
        if connect_points:
            ax2.plot(
                x,
                y_dist,
                color="C1",
                linewidth=1.4,
                alpha=0.85,
                marker="o" if show_points else None,
                markersize=3.5 if show_points else 0,
                label=distance_col,
                zorder=2,
            )
        elif show_points:
            ax2.scatter(x, y_dist, color="C1", s=18, alpha=0.85, label=distance_col, zorder=2)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    if x_lim is not None:
        ax1.set_xlim(x_lim)
    if y_func_lim is not None:
        ax1.set_ylim(y_func_lim)
    if y_dist_lim is not None:
        ax2.set_ylim(y_dist_lim)

    ax1.set_xlabel(frame_col)
    ax1.set_ylabel(function_col, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax2.set_ylabel(distance_col, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title(title)


def _draw_cluster_function_histograms(
    ax,
    sub_df,
    clusters,
    cluster_colors,
    cluster_col="Cluster",
    function_col="Function Value",
    bins=30,
    hist_stat="density",
    show_std_band=False,
    x_lim=None,
):
    """
    Plot overlaid histograms of Function Value per cluster with vertical lines
    for each cluster's mean (and optional ±1 std shaded band).
    """
    valid_df = sub_df[[cluster_col, function_col]].dropna()
    valid_df = valid_df[valid_df[cluster_col].isin(clusters)]

    if valid_df.empty:
        ax.text(0.5, 0.5, "No cluster data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{function_col} Distribution by Cluster")
        return

    stats = valid_df.groupby(cluster_col)[function_col].agg(["mean", "std"])

    all_vals = valid_df[function_col].to_numpy()
    if all_vals.min() < all_vals.max():
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
    else:
        bin_edges = bins

    for cluster in clusters:
        c_vals = valid_df.loc[valid_df[cluster_col] == cluster, function_col]
        if c_vals.empty:
            continue

        color = cluster_colors[cluster]
        mean_val = stats.loc[cluster, "mean"]
        std_val = stats.loc[cluster, "std"]

        sns.histplot(
            c_vals,
            bins=bin_edges,
            stat=hist_stat,
            ax=ax,
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.0,
            label=f"Cluster {cluster} (n={len(c_vals)})",
        )

        std_str = f" ± {std_val:.3f}" if pd.notna(std_val) else ""
        ax.axvline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=2.4,
            zorder=5,
            label=f"Cluster {cluster} Mean: {mean_val:.3f}{std_str}",
        )

        if show_std_band and pd.notna(std_val) and std_val > 0:
            ax.axvspan(
                mean_val - std_val,
                mean_val + std_val,
                color=color,
                alpha=0.08,
                zorder=1,
            )

    if x_lim is not None:
        ax.set_xlim(x_lim)

    ax.set_xlabel(function_col)
    ax.set_ylabel(hist_stat.capitalize())
    ax.set_title(f"{function_col} Distribution & Cluster Means")
    ax.legend(loc="best", ncol=2 if len(clusters) > 2 else 1)


def _draw_2d_hist_on_ax(
    ax,
    sub_df,
    x_col,
    y_col,
    title,
    bins=30,
    cmap="viridis",
    zero_line=False,
    x_lim=None,
    y_lim=None,
):
    """Plot a 2D joint histogram of `x_col` vs. `y_col` with a frame-count colorbar."""
    valid_df = sub_df[[x_col, y_col]].dropna()

    if valid_df.empty:
        ax.text(0.5, 0.5, f"No {y_col} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    sns.histplot(
        data=valid_df,
        x=x_col,
        y=y_col,
        bins=bins,
        cmap=cmap,
        cbar=True,
        cbar_kws={"label": "Frame Count"},
        ax=ax,
    )

    if zero_line:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.3, zorder=3)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)


def _padded_limits(series, pad_frac=0.05):
    clean = series.dropna()
    if clean.empty:
        return None
    lo, hi = float(clean.min()), float(clean.max())
    span = hi - lo
    pad = span * pad_frac if span > 0 else 1.0
    return (lo - pad, hi + pad)


# ==========================================
# 2. Main Unified Plotting Function
# ==========================================

def plot_all_interaction_views(
        
    df,
    interactions,
    trajectory=None,
    clusters_to_plot="all",
    interaction_col="Interaction",
    frame_col="Frame",
    function_col="Function Value",
    distance_col="Scaled Distance",
    shap_col="SHAP Value",
    cluster_col="Cluster",
    bw_adjust=1.0,
    points=True,
    point_size=5,
    point_alpha=0.08,
    show_clusters=False,
    hist_bins=30,
    hist_stat="probability",
    show_std_band=False,
    joint_bins=30,
    joint_cmap="viridis",
    share_axes=True,
    show=False,
):
    """
    Creates one unified dashboard figure per selected interaction using a single DataFrame `df`:
      - Row 1: Overall 2D KDE (left) | Overall Time-Series (right)
      - Row 2: Overlaid Function Value Histograms per cluster with vertical Mean lines
      - Row 3: 2D Joint Histogram of Distance vs. Function (left) | 2D Joint Histogram of SHAP vs. Function (right)
    """
    required_cols = [frame_col, interaction_col, function_col, distance_col, shap_col, cluster_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    if isinstance(interactions, pd.DataFrame):
        if interaction_col not in interactions.columns:
            raise ValueError(f"'{interaction_col}' not found in interactions DataFrame.")
        target_interactions = (
            interactions[interaction_col].dropna().astype(str).drop_duplicates().tolist()
        )
    else:
        target_interactions = list(dict.fromkeys(str(i) for i in interactions if pd.notna(i)))

    if not target_interactions:
        raise ValueError("No interaction names were provided to plot.")

    data = df.copy()
    if trajectory is not None:
        if "Trajectory" not in data.columns:
            raise ValueError("Trajectory specified, but 'Trajectory' column not in df.")
        data = data[data["Trajectory"].astype(str) == str(trajectory)].copy()

    available_clusters = list(pd.unique(data[cluster_col].dropna()))
    if isinstance(clusters_to_plot, str) and clusters_to_plot.lower() == "all":
        clusters = available_clusters
    else:
        clusters = list(clusters_to_plot)

    if not clusters:
        raise ValueError("No clusters found in df or selected in clusters_to_plot.")
    n_clusters = len(clusters)

    cmap = plt.get_cmap("Set1" if n_clusters <= 9 else "tab20")
    cluster_colors = {c: cmap(i % cmap.N) for i, c in enumerate(clusters)}

    figures = {}

    for interaction in target_interactions:
        sub_df = data[data[interaction_col].astype(str) == interaction].copy()

        if sub_df.empty:
            print(f"Skipping '{interaction}': no matching rows found in df.")
            continue

        if share_axes:
            frame_lim = _padded_limits(sub_df[frame_col], pad_frac=0.02)
            func_lim = _padded_limits(sub_df[function_col], pad_frac=0.06)
            dist_lim = _padded_limits(sub_df[distance_col], pad_frac=0.06)
            shap_lim = _padded_limits(sub_df[shap_col], pad_frac=0.06)
        else:
            frame_lim = func_lim = dist_lim = shap_lim = None

        fig = plt.figure(figsize=(15, 13.5))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.28)

        # --- ROW 1: Overall Plots ---
        ax_kde_all = fig.add_subplot(gs[0, 0])
        _draw_kde_on_ax(
            ax_kde_all,
            sub_df,
            title="Overall 2D KDE",
            function_col=function_col,
            distance_col=distance_col,
            points=points,
            point_size=point_size,
            point_alpha=point_alpha,
            bw_adjust=bw_adjust,
            x_lim=func_lim,
            y_lim=dist_lim,
        )

        ax_ts_all = fig.add_subplot(gs[0, 1])
        _draw_timeseries_on_ax(
            ax_ts_all,
            fig,
            sub_df,
            title="Overall Function & Distance Over Time",
            frame_col=frame_col,
            function_col=function_col,
            distance_col=distance_col,
            cluster_col=cluster_col,
            show_clusters=show_clusters,
            connect_points=True,
            show_points=False,
            x_lim=frame_lim,
            y_func_lim=func_lim,
            y_dist_lim=dist_lim,
        )

        # --- ROW 2: Overlaid Function Value Histograms & Cluster Means ---
        ax_hist = fig.add_subplot(gs[1, :])
        _draw_cluster_function_histograms(
            ax_hist,
            sub_df,
            clusters=clusters,
            cluster_colors=cluster_colors,
            cluster_col=cluster_col,
            function_col=function_col,
            bins=hist_bins,
            hist_stat=hist_stat,
            show_std_band=show_std_band,
            x_lim=func_lim,
        )

        # --- ROW 3: Side-by-Side 2D Joint Histograms ---
        ax_dist_2d = fig.add_subplot(gs[2, 0])
        _draw_2d_hist_on_ax(
            ax_dist_2d,
            sub_df,
            x_col=function_col,
            y_col=distance_col,
            title=f"2D Histogram: {distance_col} vs. {function_col}",
            bins=joint_bins,
            cmap=joint_cmap,
            zero_line=False,
            x_lim=func_lim,
            y_lim=dist_lim,
        )

        ax_shap_2d = fig.add_subplot(gs[2, 1])
        _draw_2d_hist_on_ax(
            ax_shap_2d,
            sub_df,
            x_col=function_col,
            y_col=shap_col,
            title=f"2D Histogram: {shap_col} vs. {function_col}",
            bins=joint_bins,
            cmap=joint_cmap,
            zero_line=True,
            x_lim=func_lim,
            y_lim=shap_lim,
        )

        main_title = f"Interaction Analysis: {interaction}"
        if trajectory is not None:
            main_title += f" | Trajectory: {trajectory}"
        fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98)

        if show:
            plt.show()
        
        else:
            plt.close(fig)
        
        figures[interaction] = fig

    return figures