import numpy as np
def compute_my_ls_estimate(x, y):

    # Compute X'X
    cov_mat = np.matmul(np.transpose(x), x)

    # Compute ((X'X)^-1)*X'
    if isinstance(cov_mat, np.float):
        inv_cov_mat = 1.0/cov_mat
        ls_matrix = inv_cov_mat*np.transpose(x)
    else:
        inv_cov_mat = np.linalg.inv(cov_mat)
        ls_matrix = np.matmul(inv_cov_mat, np.transpose(x))

    return np.matmul(ls_matrix, y)

def compute_l2_error(y, y_hat):
    N = np.prod(np.shape(y))
    return 1/N*np.linalg.norm(y-y_hat)**2

def compute_classification_error(y, y_hat):
    # zero-one norm which can be calculated using the l1-norm of the difference between labels
    return np.mean(np.abs(y-y_hat))

def compute_knn_dist(xk, x1, dist_type='l2'):
    # Calculate the distance of xk from each coordinate of x1
    if dist_type is 'l2':
            dist_xx1=[np.linalg.norm(xk-x1k) for x1k in x1]
    else:
        raise NotImplementedError("Other norms than l2 should be implemented")

    return dist_xx1

def knn_estimate(x, x1, y1, neighbour_size=1, dist_type='l2'):
    y_nn = []
    for xk in x:
        # Compute the distance between xk and x1
        dist_xx1 = compute_knn_dist(xk, x1)

        # Sort the element according to their distance to the coordinate of x
        I = np.argsort(dist_xx1)

        # Calculate the corresponding value
        y_knn = np.mean(y1[I[0:neighbour_size]])

        # Give the label to the corresponding value
        label = y_knn>=0.5
        y_nn.append(label)
    return np.asarray(y_nn)

