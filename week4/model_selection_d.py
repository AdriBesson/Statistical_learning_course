import os
import numpy as np
import week4.utils as ut
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
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

# Create the two models
lasso = lm.Lasso(max_iter=1000, tol=1e-4, random_state=5)
ridge = lm.Ridge()

# Fit the model for different values of the regularization parameter
list_alphas = np.logspace(-6, 6, 200)
rss_lasso = []
rss_ridge = []
coefs_lasso = []
coefs_ridge = []
for alpha in list_alphas:
    # Set the value of alpha
    lasso.set_params(alpha=alpha)
    ridge.set_params(alpha=alpha)

    # Fit the LASSO model
    lasso.fit(features, targets)
    coefs_lasso.append(lasso.coef_)

    # Prediction with LASSO
    predicted_target = lasso.predict(features)
    rss_lasso.append(ut.compute_RSS(orig=targets, pred=predicted_target))

    # Fit the ridge regression model
    ridge.fit(features, targets)
    coefs_ridge.append(ridge.coef_)

    # Prediction with Ridge
    predicted_target_ridge = ridge.predict(features)
    rss_ridge.append(ut.compute_RSS(orig=targets, pred=predicted_target_ridge))

plt.subplot(221)
ax = plt.gca()
ax.plot(list_alphas, coefs_ridge)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients')
plt.axis('tight')

plt.subplot(222)
ax = plt.gca()
ax.plot(list_alphas, rss_ridge)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Ridge RSS')
plt.axis('tight')

plt.subplot(223)
ax = plt.gca()
ax.plot(list_alphas, coefs_lasso)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients')
plt.axis('tight')

plt.subplot(224)
ax = plt.gca()
ax.plot(list_alphas, rss_lasso)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Lasso RSS')
plt.axis('tight')

plt.show()