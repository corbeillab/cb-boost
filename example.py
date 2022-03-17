from sklearn.datasets import load_breast_cancer, load_iris
from cb_boost.cb_boost import CBBoostClassifier
from self_optimizd_cb_boost import SelfOptCBBoostClassifier

import numpy as np

# Examples d'utilisation : CB-Boost

# Fonctionne avec des datasets Sk-learn
# Uniquement bi-classes
base_mat, y = load_breast_cancer(return_X_y=True)
# base_mat, y = load_iris(return_X_y=True)
clf_mat = np.genfromtxt("clf.csv")
X = base_mat.copy()
print("Labels : ", np.unique(y))
print(X.shape)
n_iters = 20

# Les trois hyper-paramêtres à modifier en cas d'optimisation de type grid search :
# n_max_iterations : nombre d'iteration de boosting
# n_stumps : nombre de stumps crees par attribut (colonne de X)
# estimators_generator: type d'estimatoeurs utilises
# si tu ne souhaites pas utiliser des stumps mets "Trees" a la place de "Stumps",
# et fais evoluer la depth en fonction des donnees.
cb_boost = CBBoostClassifier(n_max_iterations=n_iters, n_stumps=1, estimators_generator="Stumps", max_depth=1, random_start=False)

# Fit comme sklearn
cb_boost.fit(X, y)
print(cb_boost.c_bounds)
print(cb_boost.ths)
start_th, start_ind = cb_boost.ths[0]
# Pred comme sklearn
pred_y = cb_boost.predict(X)

from sklearn.metrics import zero_one_loss
print(zero_one_loss(y, pred_y))