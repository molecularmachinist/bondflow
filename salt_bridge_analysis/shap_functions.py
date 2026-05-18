#imports

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import MDAnalysis as mda
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from OPLS_MD import OPLS, OPLS_PLS, PLS
import shap

# functions

def scale_features(X):
    # Scale features for PLS regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    return X_scaled
    
def optimize_n_components(X, y, maxcomp=5, train_split=0.5, plateau_threshold=0.03, plateau_streak=2):
    # Initialize a dictionary to store scores
    data = {
        "proj": {
            "ncomp": [],
            "test_scores": [],
            "train_scores": [],
            "test_scores_opls": [],
            "train_scores_opls": []
        }
    }

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_split, random_state=42
    )

    # Iterate over different numbers of components
    for ncomp in range(1, maxcomp + 1):
        print(f"Building models with {ncomp} components", end="\r")
        pls = PLSRegression(n_components=ncomp).fit(X_train, y_train)
        
        # Assuming OPLS_PLS is a valid class similar to PLSRegression
        opls = OPLS_PLS(pls_components=ncomp).fit(X_train, y_train)

        # Save the scores in the data dictionary
        data["proj"]["ncomp"].append(ncomp)
        data["proj"]["test_scores"].append(pls.score(X_test, y_test))
        data["proj"]["train_scores"].append(pls.score(X_train, y_train))
        data["proj"]["test_scores_opls"].append(opls.score(X_test, y_test))
        data["proj"]["train_scores_opls"].append(opls.score(X_train, y_train))
        
        print(f"Train score: {data['proj']['train_scores_opls'][-1]}")
        print(f"Test score: {data['proj']['test_scores_opls'][-1]}")

    print(f"Done! Built models up to {maxcomp} components.")
    
    # Plot scores over numbers of components
    fig, ax = plt.subplots(1)

    ax.plot(data["proj"]["ncomp"], data["proj"]["test_scores"], "--", c="C0", label="PLS Test Scores")
    ax.plot(data["proj"]["ncomp"], data["proj"]["train_scores"], "--", c="C1", label="PLS Train Scores")
    ax.plot(data["proj"]["ncomp"], data["proj"]["test_scores_opls"], c="C0", label="OPLS Test Scores")
    ax.plot(data["proj"]["ncomp"], data["proj"]["train_scores_opls"], c="C1", label="OPLS Train Scores")

    ax.legend()
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Coefficient of Determination (R²)")
    ax.set_title("Model Performance vs Number of Components")
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    plt.show()

    # Determine the optimal number of components for PLS and OPLS
    optimal_n_pls = data["proj"]["ncomp"][np.argmax(data["proj"]["test_scores"])]
    optimal_n_opls = data["proj"]["ncomp"][np.argmax(data["proj"]["test_scores_opls"])]
    

    # Check for plateauing performance
    def find_plateau_index(scores):
        for i in range(1, len(scores)):
            if i >= plateau_streak:
                # Calculate the average change over the last `plateau_streak` components
                avg_change = np.mean(np.abs(np.diff(scores[i - plateau_streak:i])))
                if avg_change < plateau_threshold:
                    return i  # Return the index where plateauing starts
        return len(scores)  # If no plateauing detected, return the length of scores

    plateau_index_pls = find_plateau_index(data["proj"]["test_scores"])
    plateau_index_opls = find_plateau_index(data["proj"]["test_scores_opls"])

    # Determine the optimal number of components based on plateauing
    optimal_n_pls_plateau = data["proj"]["ncomp"][plateau_index_pls - 1] if plateau_index_pls > 0 else optimal_n_pls
    optimal_n_opls_plateau = data["proj"]["ncomp"][plateau_index_opls - 1] if plateau_index_opls > 0 else optimal_n_opls

    return optimal_n_pls, optimal_n_opls, optimal_n_pls_plateau, optimal_n_opls_plateau


def opls_model(X, y, ncomp=3):

    # Build the PLS model
    opls = OPLS_PLS(pls_components=ncomp).fit(X, y)
   
    return opls


def shap_values(linear_model, X):
   
    # get shap values
    explainer = shap.LinearExplainer(linear_model, X)
    shap_values = explainer.shap_values(X)

    return shap_values

def dimensionality_reduction_pca(shap_values, n_comp=5):

    # Apply PCA to reduce SHAP values to 2D
    pca = PCA(n_comp)
    shap_pca = pca.fit_transform(shap_values)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot cumulative explained variance
    plt.bar(range(1, len(cumulative_variance) + 1), cumulative_variance, alpha=0.7)
    plt.xlabel("Number of Principal Components")  
    plt.ylabel("Cumulative Explained Variance Ratio") 
    plt.ylim(0, 1)  
    plt.grid(axis='y')  

    return shap_pca

def cluster_shap_values(shap_pca, n_comp = 4, reg_covar = 0.001):
    gmm = BGMM(n_comp, reg_covar)
    cluster_labels = gmm.fit_predict(shap_pca)
    return cluster_labels

