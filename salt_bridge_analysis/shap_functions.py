
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu



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
    stat="mean",
    near_zero_threshold=0.05,
    mannwhitney=True,
):
    """
    Summarize SHAP values for each interaction in two clusters.

    Flags interactions where exactly one cluster's central SHAP value
    (mean or median) is near zero and the other cluster's value is farther
    from zero.

    The absolute signed difference can be used to rank these interactions
    regardless of whether the change is positive or negative.
    """

    if stat not in {"mean", "median"}:
        raise ValueError("stat must be 'mean' or 'median'")

    if len(clusters) != 2:
        raise ValueError("Exactly two cluster labels must be provided.")

    if near_zero_threshold < 0:
        raise ValueError("near_zero_threshold must be non-negative.")

    required_cols = {value_col, cluster_col, interaction_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {sorted(missing)}")

    c0, c1 = clusters
    d = df[df[cluster_col].isin(clusters)].copy()

    if d.empty:
        raise ValueError("No rows found for the requested clusters.")

    # Calculate statistics for each interaction within each cluster
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

    # Flatten multi-level column names, e.g. mean, cluster 0 -> mean_c0
    summary.columns = [
        f"{metric}_c{cluster}"
        for metric, cluster in summary.columns
    ]

    summary = summary.reset_index()

    s0 = f"{stat}_c{c0}"
    s1 = f"{stat}_c{c1}"

    if s0 not in summary.columns or s1 not in summary.columns:
        raise ValueError(
            "Could not find statistics for both requested clusters. "
            "Check the cluster labels in the dataframe."
        )

    # Signed difference: cluster 1 minus cluster 0
    change_col = f"{stat}_diff_c{c1}_minus_c{c0}"
    summary[change_col] = summary[s1] - summary[s0]

    # Absolute size of the signed difference, ignoring direction
    summary[f"abs_{stat}_diff"] = summary[change_col].abs()

    # Absolute central values in each cluster
    abs0_col = f"abs_{stat}_c{c0}"
    abs1_col = f"abs_{stat}_c{c1}"
    summary[abs0_col] = summary[s0].abs()
    summary[abs1_col] = summary[s1].abs()

    near0 = summary[abs0_col] <= near_zero_threshold
    near1 = summary[abs1_col] <= near_zero_threshold

    # True only when one cluster is near zero and the other is not
    summary["near_zero_one_cluster"] = near0 ^ near1

    # Record which cluster is near zero and which has the larger magnitude
    summary["near_zero_cluster"] = np.nan
    summary["higher_magnitude_cluster"] = np.nan

    summary.loc[near0 & ~near1, "near_zero_cluster"] = c0
    summary.loc[near0 & ~near1, "higher_magnitude_cluster"] = c1

    summary.loc[near1 & ~near0, "near_zero_cluster"] = c1
    summary.loc[near1 & ~near0, "higher_magnitude_cluster"] = c0

    # Optional Mann–Whitney U test comparing the SHAP-value distributions
    if mannwhitney:
        pvals = []

        for interaction, sub in d.groupby(interaction_col, observed=True):
            v0 = sub.loc[sub[cluster_col] == c0, value_col].dropna()
            v1 = sub.loc[sub[cluster_col] == c1, value_col].dropna()

            if len(v0) > 0 and len(v1) > 0:
                p_value = mannwhitneyu(
                    v0,
                    v1,
                    alternative="two-sided"
                ).pvalue
            else:
                p_value = np.nan

            pvals.append((interaction, p_value))

        pval_df = pd.DataFrame(
            pvals,
            columns=[interaction_col, "mannwhitney_p"]
        )

        summary = summary.merge(
            pval_df,
            on=interaction_col,
            how="left"
        )

    return summary


def order_summary(
    stats_df,
    stat="mean",
    order_by="both_high",
    clusters=(0, 1),
    interaction_col="Interaction",
    high_threshold=None,
):
    """
    Order interaction summaries.

    order_by:
      - "both_high": high absolute central values in both clusters;
        score is the smaller of the two absolute values.
      - "near_zero_change": exactly one cluster is near zero;
        rank by the absolute signed difference between clusters.
    """
    c0, c1 = clusters
    abs0_col = f"abs_{stat}_c{c0}"
    abs1_col = f"abs_{stat}_c{c1}"
    change_col = f"{stat}_diff_c{c1}_minus_c{c0}"
    abs_change_col = f"abs_{stat}_diff"

    required = [interaction_col, abs0_col, abs1_col]
    missing = [col for col in required if col not in stats_df.columns]
    if missing:
        raise ValueError(f"Missing columns in stats_df: {missing}")

    result = stats_df.copy()

    if order_by == "both_high":
        # Both clusters must have a high absolute central value.
        result[f"both_high_{stat}_score"] = result[
            [abs0_col, abs1_col]
        ].min(axis=1)

        result[f"max_abs_{stat}"] = result[
            [abs0_col, abs1_col]
        ].max(axis=1)

        if high_threshold is not None:
            result = result[
                result[f"both_high_{stat}_score"] >= high_threshold
            ]

        return result.sort_values(
            [f"both_high_{stat}_score", f"max_abs_{stat}"],
            ascending=[False, False],
        )

    elif order_by == "near_zero_change":
        if "near_zero_one_cluster" not in result.columns:
            raise ValueError(
                "The summary dataframe needs a 'near_zero_one_cluster' "
                "column. Create it with summarize_interactions_by_cluster."
            )

        if abs_change_col not in result.columns:
            if change_col not in result.columns:
                raise ValueError(
                    f"Need '{change_col}' or '{abs_change_col}' in stats_df."
                )
            result[abs_change_col] = result[change_col].abs()

        result = result[result["near_zero_one_cluster"]].copy()

        return result.sort_values(
            abs_change_col,
            ascending=False,
        )

    else:
        raise ValueError(
            "order_by must be 'both_high' or 'near_zero_change'"
        )

def split_interactions_by_cluster(
    shap_stats,
    clusters=(0, 1),
    stat="mean",
    high_quantile=0.90,
    interaction_col="Interaction",
    show=True,
    top_n=20,
):
    """
    Add both-high scores to shap_stats and create two ordered dataframes:

    1. both_high_ordered:
       Interactions in the top `high_quantile` fraction for which both
       clusters have large absolute central SHAP values.

    2. near_zero_ordered:
       Interactions whose central value is near zero in exactly one cluster,
       ordered by the absolute signed difference between clusters.

    Returns
    -------
    shap_stats : pandas.DataFrame
        Copy of the input summary with both-high score columns added.
    both_high_ordered : pandas.DataFrame
        Both-high interactions, ordered by both-high score.
    near_zero_ordered : pandas.DataFrame
        Near-zero interactions, ordered by absolute cluster difference.
    high_threshold : float
        Quantile cutoff applied to the both-high score.
    """

    if stat not in {"mean", "median"}:
        raise ValueError("stat must be 'mean' or 'median'")

    if len(clusters) != 2:
        raise ValueError("Provide exactly two cluster labels.")

    if not 0 <= high_quantile <= 1:
        raise ValueError("high_quantile must be between 0 and 1.")

    c0, c1 = clusters
    result = shap_stats.copy()

    abs0 = f"abs_{stat}_c{c0}"
    abs1 = f"abs_{stat}_c{c1}"
    both_high_score = f"both_high_{stat}_score"
    max_abs_score = f"max_abs_{stat}"
    abs_difference = f"abs_{stat}_diff"

    required = [
        interaction_col,
        abs0,
        abs1,
        "near_zero_one_cluster",
        abs_difference,
    ]
    missing = [col for col in required if col not in result.columns]
    if missing:
        raise ValueError(
            f"Missing columns in shap_stats: {missing}. "
            "Check that summarize_interactions_by_cluster was run with "
            "the same clusters and stat."
        )

    # A high score means both clusters have relatively large magnitudes.
    result[both_high_score] = result[[abs0, abs1]].min(axis=1)

    # Largest absolute central value in either cluster.
    result[max_abs_score] = result[[abs0, abs1]].max(axis=1)

    # Determine the top-quantile cutoff for the both-high score.
    valid_scores = result[both_high_score].dropna()
    if valid_scores.empty:
        raise ValueError("No valid central-value scores are available.")

    high_threshold = valid_scores.quantile(high_quantile)

    both_high_ordered = (
        result.loc[result[both_high_score] >= high_threshold]
        .sort_values(
            [both_high_score, max_abs_score],
            ascending=[False, False],
        )
        .copy()
    )

    # Keep interactions flagged as near zero in exactly one cluster.
    # Rank by absolute signed difference, ignoring direction.
    near_zero_ordered = (
        result.loc[result["near_zero_one_cluster"]]
        .sort_values(abs_difference, ascending=False)
        .copy()
    )

    if show:
        print(
            f"Both-high threshold "
            f"(quantile={high_quantile:.0%}): {high_threshold:.4g}"
        )
        print(f"Interactions high in both clusters: {len(both_high_ordered)}")
        print(f"Interactions near zero in one cluster: {len(near_zero_ordered)}")

        try:
            from IPython.display import display
            display(both_high_ordered.head(top_n))
            display(near_zero_ordered.head(top_n))
        except ImportError:
            print("\nBoth-high interactions:")
            print(both_high_ordered.head(top_n).to_string(index=False))
            print("\nNear-zero interactions:")
            print(near_zero_ordered.head(top_n).to_string(index=False))

    return result, both_high_ordered, near_zero_ordered, high_threshold


