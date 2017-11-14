import os
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error as mse
import sklearn.model_selection as model_selection
import csv

# Load the dataset
input_file = os.path.join(os.getcwd(), 'week5', 'data',  'bodyfat.csv')
file = open(input_file, 'rt')
reader = csv.reader(file, delimiter=';')
bodyfat = np.array([row for row in reader])

# Extract the header
header = bodyfat[0,:]

# Remove rows 2 and 4
col_to_del = [1, 3]
bodyfat = np.delete(bodyfat[1:,:], col_to_del, axis=0)

# Extract targets and features
targets = bodyfat[:,0].astype(np.float64)
features = np.array(bodyfat[:,1:]).T.astype(np.float64)

# Lasso model
lasso = lm.Lasso(max_iter=1000, tol=1e-4, random_state=5)
ridge = lm.Ridge()
# Sequence of alpha considered for best subset selection
list_alphas = np.logspace(-6, 6, 200)

# K-fold cross validation for Lasso with different values of the regularization parameter
K = 10
kf = model_selection.KFold(n_splits=K, shuffle=True, random_state=1)
err_k_ridge = np.zeros([len(list_alphas), K])
err_k_lasso = np.zeros([len(list_alphas), K])
it = 0
for train, test in kf.split(features.T):
    # Split the data i ntraining and test set
    features_train = features[:,train]
    features_test = features[:,test]
    targets_train = targets[train]
    targets_test = targets[test]
    for i,alpha in enumerate(list_alphas):
        # Set the value of alpha
        lasso.set_params(alpha=alpha)
        ridge.set_params(alpha=alpha)

        # Fit the lasso and ridge models on the training set
        lasso.fit(features_train.T, targets_train)
        ridge.fit(features_train.T, targets_train)

        # Predict the targets on the test set
        predicted_targets = lasso.predict(features_test.T)
        predicted_targets_ridge = ridge.predict(features_test.T)

        # Get the CV error on the test set
        err_k_lasso[i, it] += mse(y_true=targets_test, y_pred=predicted_targets)
        err_k_ridge[i, it] += mse(y_true=targets_test, y_pred=predicted_targets_ridge)
    it += 1

# Study for Lasso
cv_error_lasso = np.mean(err_k_lasso, axis=1)

# Find the best value of lambda for K-fold CV
best_alpha = list_alphas[np.argmin(cv_error_lasso)]
min_cv_error_lasso = np.min(cv_error_lasso)
print('LASSO - Best lambda - K-fold CV: {}'.format(best_alpha))
print('LASSO - Best error - K-fold CV: {}'.format(min_cv_error_lasso))

# Standard deviation of the CV error
avg_err_lasso = np.mean(err_k_lasso, axis=0)
std_cv_lasso = 1/np.sqrt(K)*np.sqrt(np.mean((err_k_lasso-avg_err_lasso)**2, axis=1))

# Best lambda - one standard error rule
list_alphas_ose = list_alphas[list_alphas > best_alpha]
cv_error_ose_lasso = cv_error_lasso[list_alphas > best_alpha]
list_alphas_ose = list_alphas_ose[cv_error_ose_lasso < min_cv_error_lasso + std_cv_lasso[np.argmin(cv_error_lasso)]]
cv_error_ose_lasso = cv_error_ose_lasso[cv_error_ose_lasso < min_cv_error_lasso + std_cv_lasso[np.argmin(cv_error_lasso)]]

print('LASSO - Standard error CV: {}'.format(std_cv_lasso[np.argmin(cv_error_lasso)]))
print('LASSO - Best lambda - K-fold CV - One standard-error rule: {}'.format(list_alphas_ose[-1]))
print('LASSO - One standard-error rule - error: {}'.format(cv_error_ose_lasso[-1]))

# Study for ridge
cv_error_ridge = np.mean(err_k_ridge, axis=1)

# Find the best value of lambda for K-fold CV
best_alpha_ridge = list_alphas[np.argmin(cv_error_ridge)]
min_cv_error_ridge = np.min(cv_error_ridge)
print('Ridge - Best lambda - K-fold CV: {}'.format(best_alpha))
print('Ridge - Best error - K-fold CV: {}'.format(min_cv_error_ridge))

# Standard deviation of the CV error
avg_err_ridge = np.mean(err_k_ridge, axis=0)
std_cv_ridge = 1/np.sqrt(K)*np.sqrt(np.mean((err_k_ridge-avg_err_ridge)**2, axis=1))

# Best lambda - one standard error rule
list_alphas_ose_ridge = list_alphas[list_alphas > best_alpha]
cv_error_ose_ridge = cv_error_ridge[list_alphas > best_alpha]
list_alphas_ose_ridge = list_alphas_ose_ridge[cv_error_ose_ridge < min_cv_error_ridge + std_cv_ridge[np.argmin(cv_error_ridge)]]
cv_error_ose_ridge = cv_error_ose_ridge[cv_error_ose_ridge < min_cv_error_ridge + std_cv_ridge[np.argmin(cv_error_ridge)]]

print('Ridge - Standard error CV: {}'.format(std_cv_ridge[np.argmin(cv_error_ridge)]))
print('Ridge - Best lambda - K-fold CV - One standard-error rule: {}'.format(list_alphas_ose_ridge[-1]))
print('Ridge - One standard-error rule - error: {}'.format(cv_error_ose_ridge[-1]))

# Display the results
plt.subplot(121)
plt.plot(list_alphas, cv_error_ridge)
plt.scatter(best_alpha_ridge, min_cv_error_ridge)
plt.scatter(list_alphas_ose_ridge[-1], cv_error_ose_ridge[-1])
ax = plt.gca()
ax = plt.gca()
ax.set_xscale('log')
ax.set_xlabel('Values of the regularization parameter')
ax.set_ylabel('Cross-validation error')
plt.title('CV error for Ridge')

plt.subplot(122)
plt.plot(list_alphas, cv_error_lasso)
plt.scatter(best_alpha, min_cv_error_lasso)
plt.scatter(list_alphas_ose[-1], cv_error_ose_lasso[-1])
ax = plt.gca()
ax = plt.gca()
ax.set_xscale('log')
ax.set_xlabel('Values of the regularization parameter')
ax.set_ylabel('Cross-validation error')
plt.title('CV error for Lasso')
plt.show()

