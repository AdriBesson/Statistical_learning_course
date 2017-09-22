import numpy as np

def compute_my_ls_estimate(x, y):

    cov_mat = np.matmul(np.transpose(x), x)

    if isinstance(cov_mat, np.float):
        inv_cov_mat = 1.0/cov_mat
        ls_matrix = inv_cov_mat*np.transpose(x)
    else:
        inv_cov_mat = np.linalg.inv(cov_mat)
        ls_matrix = np.matmul(inv_cov_mat, np.transpose(x))

    return np.matmul(ls_matrix, y)

def compute_training_error(y, y_hat):
    return np.linalg.norm(y-y_hat)**2

def compute_empirical_test_error(y, y_hat):
    return np.linalg.norm(y-y_hat, 1)