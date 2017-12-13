import os
import numpy as np
import sklearn.discriminant_analysis as lda
import sklearn.linear_model as lm
from sklearn import preprocessing as preproc
from sklearn import svm, tree, ensemble, neural_network, mixture
import sklearn.model_selection as model_selection
import csv
import project.utils as ut

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
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, train_size=0.8, test_size=0.2)

# Linear discriminant analysis
lda_model = lda.LinearDiscriminantAnalysis()
score_lda = model_selection.cross_val_score(estimator=lda_model, X=features, y=labels, cv=5)
classification_loss_lda = np.mean(score_lda)
print('Classification accuracy LDA: {0} %'.format(classification_loss_lda*100))

# Logistic regression with L2 regularization
list_C = np.logspace(10^-3, 10^3, 100)
logistic_reg = lm.LogisticRegression(penalty='l2', solver='liblinear', random_state=10)
max_score = 0
for C in list_C:
    logistic_reg.set_params(C=C)
    logistic_reg.fit(features_train, labels_train)
    score = logistic_reg.score(features_test, labels_test)
    if score > max_score:
        best_C = C
        max_score = score

logistic_reg.set_params(C=best_C)
score_reg_l2 = model_selection.cross_val_score(estimator=logistic_reg, X=features, y=labels, cv=5)
classification_loss_log = np.mean(score_reg_l2)
print('Classification accuracy Logistic L2: {0} - Best C = {1}'.format(classification_loss_log*100, best_C))

# Logistic regression with L1 regularization
list_C = np.logspace(10^-3, 10^3, 100)
logistic_reg_l1 = lm.LogisticRegression(penalty='l1', solver='liblinear', random_state=10)
max_score = 0
best_C = list_C[0]
for C in list_C:
    logistic_reg_l1.set_params(C=C)
    logistic_reg_l1.fit(features_train, labels_train)
    score = logistic_reg_l1.score(features_test, labels_test)
    if score > max_score:
        best_C = C
        max_score = score

logistic_reg_l1.set_params(C=best_C)
score_reg_l1 = model_selection.cross_val_score(estimator=logistic_reg_l1, X=features, y=labels, cv=5)
classification_loss_log = np.mean(score_reg_l1)
print('Classification accuracy Logistic L1: {0} - Best C = {1}'.format(classification_loss_log*100, best_C))

# SVM classification - Linear kernel
list_C = np.logspace(10^-3, 10^3, 100)
clf = svm.LinearSVC(random_state=10)
max_score = 0
best_C = list_C[0]
for C in list_C:
    clf.set_params(C=C)
    clf.fit(features_train, labels_train)
    score = clf.score(features_test, labels_test)
    if score > max_score:
        best_C = C
        max_score = score
clf.set_params(C=best_C)
score_svm = model_selection.cross_val_score(estimator=clf, X=features, y=labels, cv=5)
classification_loss_svm = np.mean(score_svm)
print('Classification accuracy - linear SVM - L2: {0} - Best C = {1} %'.format(classification_loss_svm*100, best_C))

# Decision tree classifier
dec_tree = tree.DecisionTreeClassifier(criterion='gini', random_state=10)
score_tree = model_selection.cross_val_score(estimator=dec_tree, X=features, y=labels, cv=5)
classification_loss_tree = np.mean(score_tree)
print('Classification accuracy - Decision tree: {0} %'.format(classification_loss_tree*100))

# Random forest classifier
random_forest = ensemble.RandomForestClassifier(criterion='gini', random_state=10, n_jobs=6)
score_forest = model_selection.cross_val_score(estimator=random_forest, X=features, y=labels, cv=5)
classification_loss_rf = np.mean(score_forest)
print('Classification accuracy - Random forest: {0} %'.format(classification_loss_rf*100))

# AdaBoost classifier
ada_boost = ensemble.AdaBoostClassifier(random_state=10)
score_ab = model_selection.cross_val_score(estimator=ada_boost, X=features, y=labels, cv=5)
classification_loss_ab = np.mean(score_ab)
print('Classification accuracy - AdaBoost: {0} %'.format(classification_loss_ab*100))

# Gradient boosting classifier
g_boost = ensemble.GradientBoostingClassifier(random_state=10)
score_gb = model_selection.cross_val_score(estimator=g_boost, X=features, y=labels, cv=5)
classification_loss_gb = np.mean(score_gb)
print('Classification accuracy - Gradient Boosting: {0} %'.format(classification_loss_gb*100))

# Multilayer perceptron
mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, ), alpha=5, activation='relu', solver='adam', batch_size=256, learning_rate='adaptive', max_iter=1000, learning_rate_init=0.001, random_state=10)
score_mlp = model_selection.cross_val_score(estimator=mlp, X=features, y=labels, cv=5)
classification_loss_mlp = np.mean(score_mlp)
print('Classification accuracy - Multi-layer perceptron: {0} %'.format(classification_loss_mlp*100))
