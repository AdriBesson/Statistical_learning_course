import os
import numpy as np
import week4.utils as ut
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# Load the dataset
input_file = os.path.join(os.getcwd(), 'data', 'bodyfat.csv')
bodyfat = np.genfromtxt(input_file, delimiter=';')
bodyfat = bodyfat[1:]   # First row is nan so it is

# Select features
targets = bodyfat[:,0]
#features = [bodyfat[:,i] for i in np.arange(np.shape(bodyfat)[1]) if not(i in [0, 1, 3])]
features = bodyfat[:,1:].T
features = np.array(features)
n_samples = features.shape[0]

# Number of features considered
list_n_features = np.arange(1,10)

# Best subset selection - AIC
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
aic_bss = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.best_subset_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='AIC')
    aic_bss.append(min_score)
selected_nb_features_aic_bss = np.argmin(aic_bss)

# Forward stepwise selection - BIC
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
bic_bss = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.best_subset_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='BIC')
    bic_bss.append(min_score)
selected_nb_features_bic_bss = np.argmin(bic_bss)

# Forward stepwise selection - Ajusted R2
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
adj_r2_bss = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.best_subset_selection_with_score(model=linear_model, n_features=n_features,
                                                                      features=features, targets=targets,
                                                                      score='Adj_R2')
    adj_r2_bss.append(min_score)
selected_nb_features_r2_bss = np.argmax(adj_r2_bss)

# Forward stepwise selection - AIC
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
aic = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.forward_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='AIC')
    aic.append(min_score)
selected_nb_features_aic = np.argmin(aic)

# Forward stepwise selection - BIC
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
bic = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.forward_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='BIC')
    bic.append(min_score)
selected_nb_features_bic = np.argmin(bic)

# Forward stepwise selection - Ajusted R2
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
adj_r2 = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.forward_selection_with_score(model=linear_model, n_features=n_features,
                                                                      features=features, targets=targets,
                                                                      score='Adj_R2')
    adj_r2.append(min_score)
selected_nb_features_r2 = np.argmax(adj_r2)

# Backward stepwise selection - AIC
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
aic_bwd = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.backward_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='AIC')
    aic_bwd.append(min_score)
selected_nb_features_aic_bwd = np.argmin(aic_bwd)

# Backward stepwise selection - BIC
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
bic_bwd = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.backward_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='BIC')
    bic_bwd.append(min_score)
selected_nb_features_bic_bwd = np.argmin(bic_bwd)

# Backward stepwise selection - Adjusted R2
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6)
adj_r2_bwd = []
for n_features in list_n_features:
    best_model, best_model_features, min_score = ut.backward_selection_with_score(model=linear_model, n_features=n_features, features=features, targets=targets, score='Adj_R2')
    adj_r2_bwd.append(min_score)
selected_nb_features_r2_bwd = np.argmax(adj_r2_bwd)

# Plot the results
f, axarr = plt.subplots(3, 3)
axarr[0, 0].plot(list_n_features, aic)
axarr[0, 0].plot(list_n_features[selected_nb_features_aic], aic[selected_nb_features_aic], 'x')
axarr[0, 0].set_title('Forward stepwise selection with AIC')
axarr[0, 0].set_ylabel('AIC')
axarr[0, 0].set_xlabel('number of features')
axarr[0, 1].plot(list_n_features, bic)
axarr[0, 1].plot(list_n_features[selected_nb_features_bic], bic[selected_nb_features_bic], 'x')
axarr[0, 1].set_title('Forward stepwise selection with BIC')
axarr[0, 1].set_ylabel('BIC')
axarr[0, 1].set_xlabel('number of features')
axarr[0, 2].plot(list_n_features, adj_r2)
axarr[0, 2].plot(list_n_features[selected_nb_features_r2], adj_r2[selected_nb_features_r2], 'x')
axarr[0, 2].set_title('Forward stepwise selection with Adjusted R2')
axarr[0, 2].set_ylabel('Adjusted R2')
axarr[0, 2].set_xlabel('number of features')
axarr[1, 0].plot(list_n_features, aic_bwd)
axarr[1, 0].plot(list_n_features[selected_nb_features_aic_bwd], aic_bwd[selected_nb_features_aic_bwd], 'x')
axarr[1, 0].set_title('Backward stepwise selection with AIC')
axarr[1, 0].set_ylabel('AIC')
axarr[1, 0].set_xlabel('number of features')
axarr[1, 1].plot(list_n_features, bic_bwd)
axarr[1, 1].plot(list_n_features[selected_nb_features_bic_bwd], bic_bwd[selected_nb_features_bic_bwd], 'x')
axarr[1, 1].set_title('Backward stepwise selection with BIC')
axarr[1, 1].set_ylabel('BIC')
axarr[1, 1].set_xlabel('number of features')
axarr[1, 2].plot(list_n_features, adj_r2_bwd)
axarr[1, 2].plot(list_n_features[selected_nb_features_r2_bwd], adj_r2_bwd[selected_nb_features_r2_bwd], 'x')
axarr[1, 2].set_title('Backward stepwise selection with Adjusted R2')
axarr[1, 2].set_ylabel('Adjusted R2')
axarr[1, 2].set_xlabel('number of features')
axarr[2, 0].plot(list_n_features, aic_bss)
axarr[2, 0].plot(list_n_features[selected_nb_features_aic_bss], aic_bss[selected_nb_features_aic_bss], 'x')
axarr[2, 0].set_title('Best subset selection with AIC')
axarr[2, 0].set_ylabel('AIC')
axarr[2, 0].set_xlabel('number of features')
axarr[2, 1].plot(list_n_features, bic_bss)
axarr[2, 1].plot(list_n_features[selected_nb_features_bic_bss], bic_bss[selected_nb_features_bic_bss], 'x')
axarr[2, 1].set_title('Best subset selection with BIC')
axarr[2, 1].set_ylabel('BIC')
axarr[2, 1].set_xlabel('number of features')
axarr[2, 2].plot(list_n_features, adj_r2_bss)
axarr[2, 2].plot(list_n_features[selected_nb_features_r2_bss], adj_r2_bss[selected_nb_features_r2_bss], 'x')
axarr[2, 2].set_title('Best subset selection with Adjusted R2')
axarr[2, 2].set_ylabel('Adjusted R2')
axarr[2, 2].set_xlabel('number of features')
plt.show()