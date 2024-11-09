import numpy as np
import scipy.stats as st


def construct_cov_mat(theta_arr, var_arr):
    cov_mat = np.zeros((theta_arr.size, theta_arr.size))

    for i, theta_1 in enumerate(theta_arr):
        for j, theta_2 in enumerate(theta_arr):
            cov_mat[i, j] = np.sqrt(var_arr[i] * var_arr[j])(1 + 15 * np.abs(theta_1 - theta_2)) * np.exp(
                -15 * np.abs(theta_1 - theta_2))

    np.fill_diagonal(cov_mat, var_arr)
    return cov_mat


