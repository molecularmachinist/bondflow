#import

import shap_functions as sf

#data

#creating pls model

model = sf.pls_model(X = dist, y = projs, ncomp=3)

#calculating shap values

sf.shap_values(model)