import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def construct_covariance_matrix(
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    variance: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Create a matrix of theta_A with suitable dimensions
    theta_A_mat = np.repeat(theta_A[:, np.newaxis], theta_A.size, axis=1)
    # Create a matrix of theta_B with suitable dimensions
    theta_B_mat = np.repeat(theta_B[:, np.newaxis], theta_B.size, axis=1)
    # Compute the distances between the points in A
    H_A = np.abs(theta_A_mat - theta_A_mat.T)
    # Compute the distances between the points in B
    H_B = np.abs(theta_B_mat - theta_B_mat.T)
    # Compute the distances between the points in A and B by
    # extending theta_A, theta_B to matrices of suitable dimensions
    H_AB = np.abs(
        np.repeat(theta_A[:, np.newaxis], theta_B.size, axis=1)
        - np.repeat(theta_B[:, np.newaxis], theta_A.size, axis=1).T
    )

    # Plug into formula
    S_A = variance * (1 + k * H_A) * np.exp(-k * H_A)
    S_B = variance * (1 + k * H_B) * np.exp(-k * H_B)
    S_AB = variance * (1 + k * H_AB) * np.exp(-k * H_AB)

    return S_A, S_B, S_AB


def get_conditional_expectance(
    mu_A: np.ndarray,
    mu_B: np.ndarray,
    x_B: np.ndarray,
    S_AB: np.ndarray,
    S_B_inverse: np.ndarray,
):
    # Compute the conditional expectance
    return mu_A + S_AB @ S_B_inverse @ (x_B - mu_B)


def get_conditional_variance(
    S_A: np.ndarray,
    S_AB: np.ndarray,
    S_B_inverse: np.ndarray,
):
    # Compute the conditional covariance
    return S_A - S_AB @ S_B_inverse @ S_AB.T


def CI(
    E_AIB: np.ndarray, V_AIB: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    # Compute confidence interval
    alpha = (1 - confidence) / 2
    z = st.norm.ppf(alpha)
    lower = E_AIB + z * np.sqrt(V_AIB.diagonal())
    upper = E_AIB - z * np.sqrt(V_AIB.diagonal())
    return lower, upper


def P(a: float, E_AIB: np.ndarray, V_AIB: np.ndarray) -> np.ndarray:
    return st.norm.cdf((a - E_AIB) / np.sqrt(V_AIB.diagonal()))
