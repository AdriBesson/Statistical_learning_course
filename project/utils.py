import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
def best_subset_selection(model, n_features, features, targets):
    min_error = float('inf')
    for l in np.arange(1, n_features+1):
        indices = list(combinations(np.arange(len(features)),int(l)))
        for index in indices:
            best_features_k = features[list(index)]
            if best_features_k.ndim == 1:
                array_best_features = np.array(best_features_k).T.reshape(-1, 1)
            else:
                array_best_features = np.array(best_features_k).T

            # Fit the model with the additional feature
            model.fit(array_best_features, targets)

            # Predict the target value and compute the RSS
            predicted_target = model.predict(array_best_features)
            error = compute_classification_loss(targets, predicted_target)

            # Identify the best column in terms of RSS
            if error <= min_error:
                best_index = list(index)
                min_error = error
        best_model = model.fit(features[best_index].T, targets)
    return best_model, best_index, min_error

def forward_selection(model, n_features, features, targets):
    best_features = []
    min_error = float('inf')
    best_index = []
    for l in np.arange(1, n_features+1):
        best_i = []
        for i,col in enumerate(features):
            if i in best_index:
                continue
            # Add a feature to the list of features
            if best_features == []:
                best_features_k = col
                array_best_features = np.array(best_features_k).T.reshape(-1, 1)
            else:
                best_features_k = best_features + [col]
                array_best_features = np.array(best_features_k).T

            # Fit the model with the additional feature
            model.fit(array_best_features, targets)

            # Predict the target value and compute the RSS
            predicted_target = model.predict(array_best_features)
            error = compute_classification_loss(y_true=targets,y_pred=predicted_target)

            # Identify the best column in terms of RSS
            if error <= min_error:
                best_col = col
                min_error = error
                best_i = i
        # Append the best feature to the current list and remove the best feature from the list of features
        if not(best_i==[]):
            best_index.append(best_i)
            best_features = best_features + [best_col]


    # Fit the best model
    best_model = model.fit(np.array(best_features).T, targets)
    return best_model, best_index, min_error

def backward_selection(model, n_features, features, targets):
    best_features = list(features)
    best_index = np.arange(0, np.shape(features)[0])
    best_ind = []
    for l in np.arange(1, n_features+1):
        min_error = float('inf')
        for i, col in enumerate(best_features):
            best_features_k = list(best_features)
            # Add a feature to the list of features
            del best_features_k[i]
            if len(best_features_k) <= 1:
                array_best_features = np.array(best_features_k).T.reshape(-1,1)
            else:
                array_best_features = np.array(best_features_k).T

            # Fit the model with the additional feature
            model.fit(array_best_features, targets)

            # Predict the target value and compute the RSS
            predicted_target = model.predict(array_best_features)
            error = compute_classification_loss(targets,predicted_target)

            # Identify the best column in terms of RSS
            if error <= min_error:
                min_error = error
                best_i = i

        # Append the best feature to the current list and remove the best feature from the list of features
        best_ind.append([ind for ind in np.arange(np.shape(features)[0]) if np.sum(np.array(features[ind,:])-best_features[best_i]) == 0])
        del best_features[best_i]
    best_index = np.delete(best_index, best_ind)
    best_model = model.fit(np.array(best_features).T, targets)
    return best_model, best_index, min_error

def compute_classification_loss(y_true, y_pred):
    return 1 - accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True, sample_weight=None)

def my_lda_fit(features, labels):
    prior_class1 = np.sum(labels == 1) / labels.shape[0]
    prior_class2 = np.sum(labels == -1) / labels.shape[0]
    mu_class1 = np.mean(features[labels == 1, :], axis=0)
    mu_class2 = np.mean(features[labels == -1, :], axis=0)
    covariance_matrix = (np.matmul((features[labels == 1, :] - mu_class1).T,
                                   (features[labels == 1, :] - mu_class1)) + np.matmul(
        (features[labels == -1, :] - mu_class2).T, (features[labels == -1, :] - mu_class2))) / (
                        labels.shape[0] - 2)
    inv_covariance_matrix = np.linalg.pinv(covariance_matrix)
    return mu_class1, mu_class2, prior_class1, prior_class2, inv_covariance_matrix

def my_lda_predict(features, mu_class1, mu_class2, prior_class1, prior_class2, inv_covariance_matrix):
    decision_function_1 = np.matmul(mu_class1, np.matmul(inv_covariance_matrix, features.T)) - 0.5 * np.matmul(
        np.matmul(mu_class1, inv_covariance_matrix), mu_class1) + np.log(prior_class1)
    decision_function_2 = np.matmul(mu_class2, np.matmul(inv_covariance_matrix, features.T)) - 0.5 * np.matmul(
        np.matmul(mu_class2, inv_covariance_matrix), mu_class2) + np.log(prior_class2)

    return 1 * (decision_function_1 >= decision_function_2) + -1 * (decision_function_1 < decision_function_2)