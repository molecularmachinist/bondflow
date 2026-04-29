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


def pls_model(X, y, ncomp=3):

    # Normalize the data
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    y_reshaped = np.reshape(y, (-1, 1))
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_reshaped)

    # Build the PLS model
    opls = OPLS_PLS(pls_components=ncomp).fit(X, y)
   
    return opls


def shap_values(model):
   
    # get shap values
    explainer = shap.Explainer(model.predict, dist)
    shap_values = explainer(dist)

    return shap_values

def dimensionality_reduction():

    pca = PCA(n_components=2)
    shap_pca = pca.fit_transform(shap_values)
    return shap_pca

def cluster_shap_values():
    gmm = BGMM(n_components=1, reg_covar = 0.001)
    cluster_labels = gmm.fit_predict(shap_pca)
    return cluster_labels

