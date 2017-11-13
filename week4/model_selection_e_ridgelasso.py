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

# Create the two models
lasso = lm.Lasso(max_iter=1000, tol=1e-4, random_state=5)
ridge = lm.Ridge()

# Fit the model for different values of the regularization parameter
list_alphas = np.logspace(-6, 6, 200)
test_error_lasso = []
test_error_ridge = []
for alpha in list_alphas:
    # Set the value of alpha
    lasso.set_params(alpha=alpha)
    ridge.set_params(alpha=alpha)

    # Fit the LASSO model
    lasso.fit(features_train, targets_train)

    # Prediction with LASSO
    predicted_target = lasso.predict(features_test)
    test_error_lasso.append(mse(y_true=targets_test, y_pred=predicted_target))

    # Fit the ridge regression model
    ridge.fit(features_train, targets_train)

    # Prediction with Ridge
    predicted_target_ridge = ridge.predict(features_test)
    test_error_ridge.append(mse(y_true=targets_test, y_pred=predicted_target_ridge))


# Plots of the mean squared error for the three methods
plt.subplot(121)
ax = plt.gca()
ax.plot(list_alphas, test_error_lasso, label='Lasso')
ax.scatter(list_alphas[np.argmin(test_error_lasso)], np.min(test_error_lasso))
ax.set_xscale('log')
ax.set_xlabel('alpha')
ax.set_ylabel('Test error')
plt.title('LASSO')

plt.subplot(122)
ax = plt.gca()
ax.plot(list_alphas, test_error_ridge, label='Ridge')
ax.scatter(list_alphas[np.argmin(test_error_ridge)], np.min(test_error_ridge))
ax.set_xscale('log')
ax.set_xlabel('alpha')
ax.set_ylabel('Test error')
plt.title('Ridge')

plt.show()