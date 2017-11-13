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
features = np.array(bodyfat[:,1:]).T.astype(np.float64)
n_samples = features.shape[0]

# Number of features considered for best subset selection
list_n_features = np.arange(1,13)

# Best subset selection
linear_model = lm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
rss_bss = []
rss_fsw = []
rss_bsw = []
for n_features in list_n_features:
    best_model, best_index, error_bss = ut.best_subset_selection(model=linear_model, n_features=n_features, features=features, targets=targets)
    best_model, best_index, error_fsw = ut.forward_selection(model=linear_model, n_features=n_features, features=features,
                                                     targets=targets)
    best_model, best_index, error_bsw = ut.backward_selection(model=linear_model, n_features=n_samples-n_features, features=features,
                                                 targets=targets)
    rss_bss.append(error_bss)
    rss_fsw.append(error_fsw)
    rss_bsw.append(error_bsw)

# Display the results
plt.figure()
plt.scatter(list_n_features, rss_bss, color='C0', label='Best subset selection')
plt.scatter(list_n_features, rss_fsw, color='C1', label='Forward selection')
plt.scatter(list_n_features, rss_bsw, color='C2', label='Backward selection')
plt.xlabel('number of features')
plt.ylabel('RSS')
plt.legend()
plt.show()

