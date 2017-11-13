import os
import numpy as np
import week4.utils as ut
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error as mse
import sklearn.model_selection as model_selection
import csv

# Load the dataset
input_file = os.path.join(os.getcwd(), 'data', 'bodyfat.csv')
file = open(input_file, 'rt')
reader = csv.reader(file, delimiter=';')
bodyfat = np.array([row for row in reader])

# Extract the header
header = bodyfat[0,:]

# Extract the usefuls rows
col_to_del = [1, 3]
bodyfat = np.delete(bodyfat[1:,:], col_to_del, axis=0)

# Extract the appropriate features
targets = bodyfat[:,0].astype(np.float64)
features = np.array(bodyfat[:,1:]).astype(np.float64)
n_samples = features.shape[0]

# Create training set and test set
train_size = 152/252
test_size = 100/252
features_train, features_test, targets_train, targets_test = model_selection.train_test_split(features, targets, train_size=train_size, test_size=test_size, random_state=10, shuffle=True)

# Cosnidered number of features
list_n_features = np.arange(1, 13)

# Best subset selection
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
min_test_rss = float('inf')
test_error_bss = []
for n_features in list_n_features:
    # Best subset selection on the training set
    best_model_bss, best_index, error_bss = ut.best_subset_selection(model=linear_model, n_features=n_features, features=features_train.T, targets=targets_train)

    # Compute the prediction error on the test set
    predicted_target = best_model_bss.predict(features_test[:,best_index])
    rss_test = mse(y_true=targets_test, y_pred=predicted_target)
    test_error_bss.append(rss_test)


# Forward stepwise selection
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
min_test_rss = float('inf')
test_error_fwd = []
for n_features in list_n_features:
    # Best subset selection on the training set
    best_model_fwd, best_index, error_fwd = ut.forward_selection(model=linear_model, n_features=n_features, features=features_train.T, targets=targets_train)

    # Compute the prediction error on the test set
    predicted_target = best_model_fwd.predict(features_test[:,best_index])
    rss_test = mse(y_true=targets_test, y_pred=predicted_target)
    test_error_fwd.append(rss_test)


# Backward stepwise selection
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
min_test_rss = float('inf')
test_error_bwd = []
for n_features in list_n_features:
    # Best subset selection on the training set
    best_model_bwd, best_index, error_bwd = ut.backward_selection(model=linear_model, n_features=n_features, features=features_train.T, targets=targets_train)

    # Compute the prediction error on the test set
    predicted_target = best_model_bwd.predict(features_test[:,best_index])
    rss_test = mse(y_true=targets_test, y_pred=predicted_target)
    test_error_bwd.append(rss_test)


# Plots of the mean squared error for the three methods
plt.subplot(131)
plt.plot(list_n_features, test_error_fwd, label='Forward stepwise selection')
plt.scatter(list_n_features[np.argmin(test_error_fwd)], np.min(test_error_fwd))
plt.plot()
plt.xlabel('Number of features')
plt.ylabel('Test RSS')
plt.title('Forward stepwise selection')

plt.subplot(132)
plt.plot(list_n_features, test_error_bwd[::-1], label='backward stepwise selection')
plt.scatter(list_n_features[np.argmin(test_error_bwd[::-1])], np.min(test_error_bwd[::-1]))
plt.xlabel('Number of features')
plt.ylabel('Test RSS')
plt.title('Backward stepwise selection')

plt.subplot(133)
plt.plot(list_n_features, test_error_bss, label='Best subset selection')
plt.scatter(list_n_features[np.argmin(test_error_bss)], np.min(test_error_bss))
plt.xlabel('Number of features')
plt.ylabel('Test RSS')
plt.title('Best subset selection')

plt.show()