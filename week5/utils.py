import numpy as np
from itertools import combinations

def compute_RSS(orig, pred):
    return np.sum((orig-pred)**2)

def compute_ms_score(proj_matrix, orig, pred, var, score='Cp'):
    scores = {'Cp', 'AIC', 'BIC', 'Adj_R2'}
    if not(score in scores):
        raise IOError('The proposed score is not yet implemented')
    N = orig.shape[0]
    rss = compute_RSS(orig, pred)
    d = np.trace(proj_matrix)
    if score == 'Cp':
        return rss + 2.0*d/N*var
    elif score == 'AIC':
        return 1.0/N/var*(rss + 2*d*var)
    elif score == 'BIC':
        return 1/N/var*(rss + np.log(N)*d*var)
    elif score == 'Adj_R2':
        tss = np.sum((orig - np.mean(orig))**2)
        return 1 - (rss/(N-d-1))/(tss/(N-1))

def get_projection_matrix(x):
    # Compute X'X
    cov_mat = np.matmul(np.transpose(x), x)
    # Compute ((X'X)^-1)*X'
    if isinstance(cov_mat, np.float):
        inv_cov_mat = 1.0 / cov_mat
        ls_matrix = inv_cov_mat * np.transpose(x)
    else:
        inv_cov_mat = np.linalg.inv(cov_mat)
        ls_matrix = np.matmul(inv_cov_mat, np.transpose(x))
    return np.matmul(x, ls_matrix)


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
            error = compute_RSS(targets, predicted_target)

            # Identify the best column in terms of RSS
            if error < min_error:
                best_index = list(index)
                min_error = error
        best_model = model.fit(features[best_index].T, targets)
    return best_model, best_index, min_error

def forward_selection(model, n_features, features, targets):
    best_features = []
    min_error = float('inf')
    best_index = []
    for l in np.arange(1, n_features+1):
        for i,col in enumerate(features):
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
            error = compute_RSS(targets,predicted_target)

            # Identify the best column in terms of RSS
            if error < min_error:
                best_col = col
                min_error = error
                best_i = i
        # Append the best feature to the current list and remove the best feature from the list of features
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
            error = compute_RSS(targets,predicted_target)

            # Identify the best column in terms of RSS
            if error < min_error:
                min_error = error
                best_i = i

        # Append the best feature to the current list and remove the best feature from the list of features
        best_ind.append([ind for ind in np.arange(np.shape(features)[0]) if np.sum(np.array(features[ind,:])-best_features[best_i]) == 0])
        del best_features[best_i]
    best_index = np.delete(best_index, best_ind)
    best_model = model.fit(np.array(best_features).T, targets)
    return best_model, best_index, min_error


def best_subset_selection_with_score(model, n_features, features, targets, score='Cp'):
    best_features = []
    min_error = float('inf')
    if not (score is 'Adj_R2'):
        min_score = float('inf')
    else:
        min_score = float('-inf')

    # Compute the variance of the model (used to calculate the score)
    model.fit(features.T, targets)
    predicted_target = model.predict(features.T)
    var = np.var(targets - predicted_target)

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
            error = compute_RSS(targets, predicted_target)

            # Identify the best column in terms of RSS
            if error < min_error:
                best_index = list(index)
                min_error = error
        best_features = features[best_index]

        # Fit the model
        array_best_features = np.array(best_features).T
        model.fit(array_best_features, targets)

        # Compute the model selection metric
        predicted_target = model.predict(array_best_features)
        proj_matrix = get_projection_matrix(array_best_features)
        score_val = compute_ms_score(proj_matrix=proj_matrix, orig=targets, pred=predicted_target, var=var, score=score)
        if not (score is 'Adj_R2'):
            if score_val < min_score:
                best_model = model
                best_model_features = best_features
                min_score = score_val
        else:
            if score_val > min_score:
                best_model = model
                best_model_features = best_features
                min_score = score_val

    return best_model, best_model_features, min_score

def forward_selection_with_score(model, n_features, features, targets, score='Cp'):
    best_features = []
    min_error = float('inf')
    if not (score is 'Adj_R2'):
        min_score = float('inf')
    else:
        min_score = float('-inf')

    # Compute the variance of the model (used to calculate the score)
    model.fit(features.T, targets)
    predicted_target = model.predict(features.T)
    var = np.var(targets - predicted_target)

    for l in np.arange(1, n_features+1):
        for col in features:
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
            error = compute_RSS(targets,predicted_target)

            # Identify the best column in terms of RSS
            if error < min_error:
                best_col = col
                min_error = error

        # Append the best feature to the current list and remove the best feature from the list of features
        best_features.append(best_col)
        features = features - [best_col]

        # Fit the model
        array_best_features = np.array(best_features).T
        model.fit(array_best_features, targets)

        # Compute the model selection metric
        predicted_target = model.predict(array_best_features)
        proj_matrix = get_projection_matrix(array_best_features)
        score_val = compute_ms_score(proj_matrix=proj_matrix, orig=targets, pred=predicted_target, var=var, score=score)
        if not (score is 'Adj_R2'):
            if score_val < min_score:
                best_model = model
                best_model_features = best_features
                min_score = score_val
        else:
            if score_val > min_score:
                best_model = model
                best_model_features = best_features
                min_score = score_val


    return best_model, best_model_features, min_score

def backward_selection_with_score(model, n_features, features, targets, score='Cp'):
    best_features = list(features)
    min_error = float('inf')
    if not (score is 'Adj_R2'):
        min_score = float('inf')
    else:
        min_score = float('-inf')

    # Compute the variance of the model (used to calculate the score)
    model.fit(features.T, targets)
    predicted_target = model.predict(features.T)
    var = np.var(targets - predicted_target)

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
            error = compute_RSS(targets,predicted_target)

            # Identify the best column in terms of RSS
            if error < min_error:
                min_error = error
                best_i = i
        # Append the best feature to the current list and remove the best feature from the list of features
        del best_features[best_i]

        # Fit the model
        if len(best_features) <= 1:
            array_best_features = np.array(best_features).T.reshape(-1, 1)
        else:
            array_best_features = np.array(best_features).T
        model.fit(array_best_features, targets)

        # Compute the model selection metric
        predicted_target = model.predict(array_best_features)
        proj_matrix = get_projection_matrix(array_best_features)
        score_val = compute_ms_score(proj_matrix=proj_matrix, orig=targets, pred=predicted_target, var=var, score=score)
        if not (score is 'Adj_R2'):
            if score_val < min_score:
                best_model = model
                best_model_features = best_features
                min_score = score_val
        else:
            if score_val > min_score:
                best_model = model
                best_model_features = best_features
                min_score = score_val


    return best_model, best_model_features, min_score
