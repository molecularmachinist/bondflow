import numpy as np
import matplotlib.pyplot as plt
import shap
import MDAnalysis as mda
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from statsmodels.stats.diagnostic import lilliefors


def scale_features(X):
    """Scale features using standard scaling."""
    X = X.T
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def optimize_n_components(
    X,
    y,
    maxcomp=5,
    train_split=0.5,
    plateau_threshold=0.03,
    plateau_streak=2,
):

    data = {
        "proj": {
            "ncomp": [],
            "test_scores": [],
            "train_scores": [],
            "test_scores_opls": [],
            "train_scores_opls": [],
        }
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_split,
        random_state=42,
    )

    for ncomp in range(1, maxcomp + 1):

        print(f"Building models with {ncomp} components", end="\r")

        pls = PLSRegression(n_components=ncomp).fit(X_train, y_train)

        data["proj"]["ncomp"].append(ncomp)

        data["proj"]["train_scores"].append(pls.score(X_train, y_train))
        data["proj"]["test_scores"].append(pls.score(X_test, y_test))

    print(f"Done! Built models up to {maxcomp} components.")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        data["proj"]["ncomp"],
        data["proj"]["test_scores"],
        "--",
        label="PLS Test",
    )

    ax.plot(
        data["proj"]["ncomp"],
        data["proj"]["train_scores"],
        "--",
        label="PLS Train",
    )


    ax.set_xlabel("Number of Components")
    ax.set_ylabel(r"$R^2$")
    ax.legend()

    plt.tight_layout()
    plt.show()

    optimal_n_pls = data["proj"]["ncomp"][
        np.argmax(data["proj"]["test_scores"])
    ]


    def find_plateau_index(scores):
        for i in range(plateau_streak, len(scores)):
            avg_change = np.mean(
                np.abs(np.diff(scores[i - plateau_streak : i]))
            )

            if avg_change < plateau_threshold:
                return i

        return len(scores)

    plateau_index_pls = find_plateau_index(data["proj"]["test_scores"])

    optimal_n_pls_plateau = (
        data["proj"]["ncomp"][plateau_index_pls - 1]
        if plateau_index_pls > 0
        else optimal_n_pls
    )


    return (
        optimal_n_pls,
        optimal_n_pls_plateau,
    )

def pls_model(X, y, ncomp=3):
    """Fit a PLS model and preserve DataFrame feature names."""

    model = PLSRegression(
        n_components=ncomp
    ).fit(X, y)

    if hasattr(X, "columns"):
        model.feature_names = list(X.columns)
    else:
        model.feature_names = [
            f"Feature_{i}"
            for i in range(X.shape[1])
        ]

    return model


def compute_shap_values(model, X):
    """Compute SHAP values for a fitted linear model."""
    explainer = shap.LinearExplainer(model, X)
    return explainer(X)

def optimize_pca(shap_values, n_comp=6):

    if isinstance(shap_values, shap.Explanation):
        shap_values = shap_values.values

    pca = PCA(n_components=n_comp)

    shap_pca = pca.fit_transform(shap_values)

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(6, 4))
    plt.bar(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
    )
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()

    
def dimensionality_reduction_pca(shap_values, n_comp=2):

    if isinstance(shap_values, shap.Explanation):
        shap_values = shap_values.values

    pca = PCA(n_components=n_comp)

    shap_pca = pca.fit_transform(shap_values)

    return shap_pca, pca


def cluster_shap(
    shap_pca,
    shap_values,
    X_scaled,
    Y=None,
    trajectory_ids="Trajectory_0",
    feature_names=None,
    n_components=4,
    reg_covar=0.001,
    pcs_to_use=(0,),
    random_state=42,
):
    """
    Cluster frames using PCA coordinates, then return one long dataframe containing:

    Trajectory | Frame | Interaction | SHAP Value | Scaled Distance | Cluster | Function Value

    Requirements:
    - shap_values rows = frames
    - shap_values columns = interactions
    - X_scaled rows = frames
    - X_scaled columns = interactions
    """

    # SHAP values and feature names
    if isinstance(shap_values, shap.Explanation):
        shap_array = np.asarray(shap_values.values)
        if feature_names is None:
            feature_names = shap_values.feature_names
    else:
        shap_array = np.asarray(shap_values)

    n_frames, n_features = shap_array.shape

    # Use X_scaled column names if available
    if isinstance(X_scaled, pd.DataFrame):
        x_names = list(X_scaled.columns)

        if feature_names is None:
            feature_names = x_names

        missing = set(feature_names) - set(x_names)
        if missing:
            raise ValueError(f"These feature names are missing from X_scaled: {missing}")

        X_dist = X_scaled.loc[:, feature_names].to_numpy()

    else:
        X_dist = np.asarray(X_scaled)

        if feature_names is None:
            raise ValueError(
                "Feature names are required. "
                "Pass feature_names explicitly or provide X_scaled as a DataFrame with column names."
            )

    if feature_names is None:
        raise ValueError("Feature names are required.")

    if len(feature_names) != n_features:
        raise ValueError("Number of feature names must match SHAP columns.")

    if X_dist.shape != shap_array.shape:
        raise ValueError("X_scaled and shap_values must have the same shape.")

    # Trajectory ids
    if np.isscalar(trajectory_ids):
        trajectory_ids = np.repeat(trajectory_ids, n_frames)
    else:
        trajectory_ids = np.asarray(trajectory_ids)

    if len(trajectory_ids) != n_frames:
        raise ValueError("trajectory_ids must have length equal to number of frames.")

    # Function values
    if Y is not None:
        Y = np.asarray(Y)
        if len(Y) != n_frames:
            raise ValueError("Y must have length equal to number of frames.")

    # PCA data for clustering
    shap_pca = np.asarray(shap_pca)

    if shap_pca.shape[0] != n_frames:
        raise ValueError("shap_pca and shap_values must have the same number of rows.")

    X_cluster = shap_pca[:, tuple(pcs_to_use)]

    if X_cluster.ndim == 1:
        X_cluster = X_cluster.reshape(-1, 1)

    # Cluster frames
    gmm = BGMM(
        n_components=n_components,
        reg_covar=reg_covar,
        max_iter=1000,
        tol=1e-4,
        random_state=random_state,
    )

    raw_labels = gmm.fit_predict(X_cluster)

    # Reorder cluster labels along the first selected PC
    ordered = sorted(
        np.unique(raw_labels),
        key=lambda k: X_cluster[raw_labels == k, 0].mean()
    )

    label_map = {old: new for new, old in enumerate(ordered)}
    cluster_labels = np.array([label_map[x] for x in raw_labels])

    frames = np.arange(n_frames)

    df = pd.DataFrame({
        "Trajectory": np.repeat(trajectory_ids, n_features),
        "Frame": np.repeat(frames, n_features),
        "Interaction": np.tile(feature_names, n_frames),
        "SHAP Value": shap_array.ravel(),
        "Scaled Distance": X_dist.ravel(),
        "Cluster": np.repeat(cluster_labels, n_features),
    })

    if Y is not None:
        df["Function Value"] = np.repeat(Y, n_features)

    df["Trajectory"] = df["Trajectory"].astype("category")
    df["Interaction"] = df["Interaction"].astype("category")

    return gmm, cluster_labels, df

def summarize_interactions_by_cluster(
    df,
    value_col="SHAP Value",
    cluster_col="Cluster",
    interaction_col="Interaction",
    clusters=(0, 1),
    stat="median",
    mannwhitney=True,
):
    """
    Build summary statistics per interaction for two clusters.

    Adds ordering columns for:
    1. interactions high in both clusters
    2. interactions with largest change in absolute central value
    """

    if stat not in ["mean", "median"]:
        raise ValueError("stat must be 'mean' or 'median'")

    c0, c1 = clusters

    d = df[df[cluster_col].isin(clusters)].copy()

    summary = (
        d.groupby([interaction_col, cluster_col], observed=True)[value_col]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .unstack(cluster_col)
    )

    summary.columns = [
        f"{metric}_c{cluster}"
        for metric, cluster in summary.columns
    ]

    summary = summary.reset_index()

    s0 = f"{stat}_c{c0}"
    s1 = f"{stat}_c{c1}"

    if s0 not in summary.columns or s1 not in summary.columns:
        raise ValueError("One or both requested clusters are missing.")

    # Signed difference
    summary[f"{stat}_diff_c{c1}_minus_c{c0}"] = summary[s1] - summary[s0]
    summary[f"abs_signed_{stat}_diff"] = summary[
        f"{stat}_diff_c{c1}_minus_c{c0}"
    ].abs()

    # Absolute central values per cluster
    summary[f"abs_{stat}_c{c0}"] = summary[s0].abs()
    summary[f"abs_{stat}_c{c1}"] = summary[s1].abs()

    # How high both clusters are
    # Large value means both clusters have high absolute mean/median.
    summary[f"both_high_{stat}_score"] = summary[
        [f"abs_{stat}_c{c0}", f"abs_{stat}_c{c1}"]
    ].min(axis=1)

    # Maximum absolute value in either cluster
    summary[f"max_abs_{stat}"] = summary[
        [f"abs_{stat}_c{c0}", f"abs_{stat}_c{c1}"]
    ].max(axis=1)

    # Change between absolute values
    # Large value means one cluster has much larger absolute mean/median than the other.
    summary[f"abs_{stat}_change"] = (
        summary[f"abs_{stat}_c{c1}"] - summary[f"abs_{stat}_c{c0}"]
    ).abs()

    # Direction of larger absolute value
    summary["larger_abs_cluster"] = np.where(
        summary[f"abs_{stat}_c{c1}"] > summary[f"abs_{stat}_c{c0}"],
        c1,
        c0,
    )

    if mannwhitney:
        pvals = []

        for interaction, sub in d.groupby(interaction_col, observed=True):
            v0 = sub.loc[sub[cluster_col] == c0, value_col].dropna()
            v1 = sub.loc[sub[cluster_col] == c1, value_col].dropna()

            if len(v0) > 0 and len(v1) > 0:
                p = mannwhitneyu(v0, v1, alternative="two-sided").pvalue
            else:
                p = np.nan

            pvals.append((interaction, p))

        pval_df = pd.DataFrame(
            pvals,
            columns=[interaction_col, "mannwhitney_p"]
        )

        summary = summary.merge(pval_df, on=interaction_col, how="left")

    return summary

def order_summary(
    stats_df,
    stat="mean",
    order_by="both_high",
    interaction_col="Interaction",
):
    """
    order_by options:
    - "both_high": interactions high in both clusters
    - "abs_change": largest change between absolute cluster stats
    """

    if order_by == "both_high":
        sort_cols = [f"both_high_{stat}_score", f"max_abs_{stat}"]
        ascending = [False, False]

    elif order_by == "abs_change":
        sort_cols = [f"abs_{stat}_change", f"max_abs_{stat}"]
        ascending = [False, False]

    else:
        raise ValueError("order_by must be 'both_high' or 'abs_change'")

    missing = [c for c in sort_cols if c not in stats_df.columns]
    if missing:
        raise ValueError(f"Missing columns in stats_df: {missing}")

    return stats_df.sort_values(sort_cols, ascending=ascending).copy()


def plot_summary_bars(
    stats_df,
    stat="mean",
    clusters=(0, 1),
    order_by="both_high",
    top_n=20,
    interaction_col="Interaction",
    title=None,
    figsize=None,
):
    """
    Bar plot using summary statistics only.

    Bars = abs(mean/median) per cluster.
    Marker color = sign of signed mean/median.
    """

    c0, c1 = clusters

    signed0_col = f"{stat}_c{c0}"
    signed1_col = f"{stat}_c{c1}"
    abs0_col = f"abs_{stat}_c{c0}"
    abs1_col = f"abs_{stat}_c{c1}"

    needed = [signed0_col, signed1_col, abs0_col, abs1_col, interaction_col]
    missing = [c for c in needed if c not in stats_df.columns]

    if missing:
        raise ValueError(f"Missing columns in stats_df: {missing}")

    plot_df = (
        order_summary(
            stats_df,
            stat=stat,
            order_by=order_by,
            interaction_col=interaction_col,
        )
        .head(top_n)
        .copy()
    )

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
        figsize = (9, max(4, 0.45 * len(plot_df)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        y - height / 2,
        abs0,
        height=height,
        color="C0",
        alpha=0.7,
        label=f"Cluster {c0}",
    )

    ax.barh(
        y + height / 2,
        abs1,
        height=height,
        color="C1",
        alpha=0.7,
        label=f"Cluster {c1}",
    )

    for i in range(len(plot_df)):
        if not np.isnan(signed0[i]):
            color0 = "red" if signed0[i] > 0 else "blue"
            ax.scatter(
                abs(signed0[i]),
                y[i] - height / 2,
                color=color0,
                edgecolor="black",
                zorder=5,
            )

        if not np.isnan(signed1[i]):
            color1 = "red" if signed1[i] > 0 else "blue"
            ax.scatter(
                abs(signed1[i]),
                y[i] + height / 2,
                color=color1,
                edgecolor="black",
                zorder=5,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel(f"abs({stat})")
    ax.set_ylabel("Interaction")

    if title is None:
        title = f"Top {top_n} interactions ordered by {order_by}"

    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    return fig

def plot_top_distributions(
    df,
    stats_df,
    value_col="SHAP Value",
    stat="mean",
    clusters=(0, 1),
    order_by="both_high",
    top_n=9,
    interaction_col="Interaction",
    cluster_col="Cluster",
    kind="kde",
    use_abs=False,
    ncols=3,
    bw_adjust=1,
    figsize=None,
):
    """
    Plot KDEs or histograms for top interactions.

    stats_df decides the ordering.
    df provides the raw per-frame values.
    """

    ordered_stats = order_summary(
        stats_df,
        stat=stat,
        order_by=order_by,
        interaction_col=interaction_col,
    )

    interactions = ordered_stats[interaction_col].head(top_n).tolist()

    if len(interactions) == 0:
        print("No interactions to plot.")
        return None

    plot_df = df[
        df[interaction_col].isin(interactions)
        & df[cluster_col].isin(clusters)
    ].copy()

    if use_abs:
        plot_df["_plot_value"] = plot_df[value_col].abs()
        x_col = "_plot_value"
        x_label = f"abs({value_col})"
    else:
        x_col = value_col
        x_label = value_col

    n = len(interactions)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.flatten()

    for i, interaction in enumerate(interactions):
        ax = axes[i]

        sub = plot_df[plot_df[interaction_col] == interaction]

        for j, cluster in enumerate(clusters):
            vals = sub.loc[sub[cluster_col] == cluster, x_col].dropna()

            if kind == "kde":
                if len(vals) > 1:
                    sns.kdeplot(
                        x=vals,
                        ax=ax,
                        label=f"Cluster {cluster}",
                        color=f"C{j}",
                        bw_adjust=bw_adjust,
                    )

            elif kind == "hist":
                if len(vals) > 0:
                    sns.histplot(
                        x=vals,
                        ax=ax,
                        label=f"Cluster {cluster}",
                        color=f"C{j}",
                        stat="density",
                        bins=30,
                        alpha=0.45,
                    )

            else:
                raise ValueError("kind must be 'kde' or 'hist'")

            if len(vals) > 0:
                ax.axvline(
                    np.mean(vals),
                    color=f"C{j}",
                    linestyle="--",
                    alpha=0.8,
                )

        ax.set_title(str(interaction))
        ax.set_xlabel(x_label)

        if i == 0:
            ax.legend()

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    return fig
