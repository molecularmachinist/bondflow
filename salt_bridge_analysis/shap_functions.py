import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture as BGMM


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


def dimensionality_reduction_pca(shap_values, n_comp=2):

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

    return shap_pca, pca


def cluster_shap_values(
    shap_pca,
    n_comp=4,
    reg_covar=1e-3,
):

    gmm = BayesianGaussianMixture(
        n_components=n_comp,
        reg_covar=reg_covar,
        random_state=42,
    )

    return gmm.fit_predict(shap_pca)


def cluster_shap_values(
    shap_pca,
    shap_values,
    feature_names,
    n_components=4,
    reg_covar=0.001,
):
    """
    Cluster PCA-reduced SHAP values using a Bayesian Gaussian Mixture Model
    and compute mean SHAP values for each cluster.

    Parameters
    ----------
    shap_pca : array-like
        PCA-reduced SHAP values, shape (n_samples, n_pca_components).

    shap_values : array-like or shap.Explanation
        Original SHAP values, shape (n_samples, n_features).

    feature_names : list
        Names of the original features.

    n_components : int, default=4
        Number of mixture components/clusters.

    reg_covar : float, default=0.001
        Non-negative regularization added to the covariance matrices.

    Returns
    -------
    gmm : BayesianGaussianMixture
        Fitted Bayesian Gaussian Mixture model.

    cluster_labels : ndarray
        Cluster assignment for each observation.

    cluster_means : DataFrame
        Mean SHAP value for each feature within each cluster.
    """

    # Extract SHAP values if a SHAP Explanation object was provided
    if isinstance(shap_values, shap.Explanation):
        shap_values = shap_values.values

    # Fit Bayesian Gaussian Mixture Model
    gmm = BGMM(
        n_components=n_components,
        reg_covar=reg_covar
    )

    cluster_labels = gmm.fit_predict(shap_pca)

    # Create DataFrame with original SHAP values
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )

    # Add cluster labels
    shap_df["Cluster"] = cluster_labels

    # Compute mean SHAP value for each cluster
    cluster_means = shap_df.groupby("Cluster").mean()

    # Display cluster means
    print("Mean Feature Values per Cluster:")
    print(cluster_means)

    # Visualize clusters
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        shap_pca[:, 0],
        shap_pca[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6
    )

    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("GMM Clustering of SHAP Values (PCA-Reduced)")
    plt.show()

    return gmm, cluster_labels, cluster_means

