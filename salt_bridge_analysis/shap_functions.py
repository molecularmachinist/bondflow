import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from OPLS_MD import OPLS, OPLS_PLS, PLS


def scale_features(X):
    """Scale features using standard scaling."""
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
        opls = OPLS_PLS(pls_components=ncomp).fit(X_train, y_train)

        data["proj"]["ncomp"].append(ncomp)

        data["proj"]["train_scores"].append(pls.score(X_train, y_train))
        data["proj"]["test_scores"].append(pls.score(X_test, y_test))

        data["proj"]["train_scores_opls"].append(opls.score(X_train, y_train))
        data["proj"]["test_scores_opls"].append(opls.score(X_test, y_test))

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

    ax.plot(
        data["proj"]["ncomp"],
        data["proj"]["test_scores_opls"],
        label="OPLS Test",
    )

    ax.plot(
        data["proj"]["ncomp"],
        data["proj"]["train_scores_opls"],
        label="OPLS Train",
    )

    ax.set_xlabel("Number of Components")
    ax.set_ylabel(r"$R^2$")
    ax.legend()

    plt.tight_layout()
    plt.show()

    optimal_n_pls = data["proj"]["ncomp"][
        np.argmax(data["proj"]["test_scores"])
    ]

    optimal_n_opls = data["proj"]["ncomp"][
        np.argmax(data["proj"]["test_scores_opls"])
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
    plateau_index_opls = find_plateau_index(data["proj"]["test_scores_opls"])

    optimal_n_pls_plateau = (
        data["proj"]["ncomp"][plateau_index_pls - 1]
        if plateau_index_pls > 0
        else optimal_n_pls
    )

    optimal_n_opls_plateau = (
        data["proj"]["ncomp"][plateau_index_opls - 1]
        if plateau_index_opls > 0
        else optimal_n_opls
    )

    return (
        optimal_n_pls,
        optimal_n_opls,
        optimal_n_pls_plateau,
        optimal_n_opls_plateau,
    )


def opls_model(X, y, ncomp=3):
    """Fit an OPLS model."""
    return OPLS_PLS(pls_components=ncomp).fit(X, y)


def compute_shap_values(model, X):
    """Compute SHAP values for a fitted linear model."""
    explainer = shap.LinearExplainer(model, X)
    return explainer.shap_values(X)


def dimensionality_reduction_pca(shap_values, n_comp=5):

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

    return shap_pca


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