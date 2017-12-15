import os
import numpy as np
import sklearn.discriminant_analysis as lda
import sklearn.model_selection as model_selection
from sklearn import preprocessing as preproc
from sklearn import linear_model as lm
from sklearn import svm, tree, ensemble, neural_network
import csv
import project.utils as ut
import matplotlib.pyplot as plt

# Load the training set
input_file = os.path.join(os.getcwd(), 'data', 'voice.csv')
file = open(input_file, 'rt')
reader = csv.reader(file, delimiter=',')
voice = np.array([row for row in reader])

# Extract the header
header = voice[0,:]
voice_no_header = voice[1:,:]

# Remove dummy features (obvious linear combination of others)
mask = np.ones(voice_no_header.shape[1], dtype=bool)
mask[[5, 11, 18]] = False # dfrange=maxdom-mindom and IQR=Q75-Q25 and centroid = meanfreq
voice_no_header = voice_no_header[:,mask]
header = header[mask]

# Extract the genders and the features
genders = voice_no_header[1:,-1]
features = voice_no_header[1:,0:voice_no_header.shape[1]-1].astype(np.float64)
n_samples = voice_no_header.shape[0]
features = preproc.scale(features)

# Assign +1 and -1 to the label
labels = 1*(genders == 'male') + -1*(genders == 'female')

# Create a training set and a test set for cross validation
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, train_size=0.5, test_size=0.5, random_state=10)

# Forward stepwise selection
list_n_features = np.arange(1, header.shape[0]-1)
model = lda.LinearDiscriminantAnalysis()
#model = lda.QuadraticDiscriminantAnalysis()
#model = ensemble.GradientBoostingClassifier(random_state=10)
#model = ensemble.AdaBoostClassifier(random_state=10)
test_error_fwd = []
best_score_test = 0
print('**************** Forward stepwise selection ****************')
for n_features in list_n_features:
    print('******* Number of features: {0}*******'.format(n_features))
    # Forward stepwise selection on the training set
    best_model_fwd, best_index_fwd, error_fwd = ut.forward_selection(model=model, n_features=n_features, features=features_train.T, targets=labels_train)

    # Compute the prediction error on the test set
    score_test = best_model_fwd.score(X=features_test[:,best_index_fwd], y=labels_test)
    if score_test > best_score_test:
        best_model = best_model_fwd
        best_index = best_index_fwd
        best_score_test = score_test
    test_error_fwd.append(1-score_test)

print('**************** End of forward selection ****************')
# Best model
print('Best LDA model - forward stepwise selection: {0}'.format(header[best_index]))
print('Best classification accuracy - FSS: {0}'.format(best_score_test))
print('Best number of features - FSS: {0}'.format(len(best_index)))
print('Best set of features - FSS: {0}'.format(header[best_index]))


plt.figure()
plt.plot(list_n_features, test_error_fwd, label='Forward stepwise selection')
plt.scatter(list_n_features[np.argmin(test_error_fwd)], np.min(test_error_fwd))
plt.plot()
plt.xlabel('Number of features')
plt.ylabel('Classification error')
plt.title('Forward stepwise selection')
plt.show()