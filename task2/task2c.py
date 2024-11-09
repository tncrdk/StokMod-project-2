import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import utils as u


def task2c():
    theta_A_arr = np.linspace(0.25, 0.5, 51)
    theta_B_arr = np.array([0.3, 0.33, 0.35, 0.39, 0.41, 0.45])
    x_B_arr = np.array([0.5, 0.4, 0.32, 0.40, 0.35, 0.6])
    mu_A = np.full_like(theta_A_arr, 0.5)
    mu_B = np.full_like(theta_B_arr, 0.5)
    variance = 0.5**2
    k = 15

    S_A, S_B, S_AB = u.construct_covariance_matrix(
        theta_A_arr, theta_B_arr, variance, k
    )

    S_B_inv = np.linalg.inv(S_B)
    E_AIB = u.get_conditional_expectance(mu_A, mu_B, x_B_arr, S_AB, S_B_inv)
    V_AIB = u.get_conditional_variance(S_A, S_AB, S_B_inv)
    # Some indices were negative but effectively zero
    V_AIB[np.diag_indices_from(V_AIB)] = np.abs(V_AIB[np.diag_indices_from(V_AIB)])

    lower, upper = u.CI(E_AIB, V_AIB, 0.9)

    plt.plot(theta_A_arr, E_AIB, label="Y")
    plt.plot(theta_A_arr, lower, label="Lower")
    plt.plot(theta_A_arr, upper, label="Upper")
    plt.legend()
    plt.savefig("./task2c_CI.pdf")
    plt.show()
    plt.clf()

    p_theta = u.P(0.3, E_AIB, V_AIB)
    plt.plot(theta_A_arr, p_theta)
    plt.savefig("./task2c_p.pdf")
    plt.show()
    plt.clf()


task2c()
