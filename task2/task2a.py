import numpy as np
import matplotlib.pyplot as plt
import utils as u


def task2a():
    #Initialization of needed arrays and values
    theta_A_arr = np.linspace(0.25, 0.5, 51)
    #Known evaluations points
    theta_B_arr = np.array([0.3, 0.35, 0.39, 0.41, 0.45])
    x_B_arr = np.array([0.5, 0.32, 0.40, 0.35, 0.6])
    #Expectation values
    mu_A = np.full_like(theta_A_arr, 0.5)
    mu_B = np.full_like(theta_B_arr, 0.5)
    variance = 0.5**2
    #Correlation function parameter
    k = 15

    #Covariance matrices
    S_A, S_B, S_AB = u.construct_covariance_matrix(
        theta_A_arr, theta_B_arr, variance, k
    )

    #Conditional expectation and conditional variance
    S_B_inv = np.linalg.inv(S_B)
    E_AIB = u.get_conditional_expectance(mu_A, mu_B, x_B_arr, S_AB, S_B_inv)
    V_AIB = u.get_conditional_variance(S_A, S_AB, S_B_inv)

    #Confidence intervals
    lower, upper = u.CI(E_AIB, V_AIB, 0.9)

    plt.title(r"Prediction of $Y(\theta)$ and $90$% confidence intervals")
    plt.plot(theta_A_arr, E_AIB, label=r"$Y(\theta)$")
    plt.plot(theta_A_arr, lower, label="Lower")
    plt.plot(theta_A_arr, upper, label="Upper")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$Y$")
    plt.savefig("./task2a.pdf")
    plt.show()
    plt.clf()

    p_theta = u.P(0.3, E_AIB, V_AIB)

    plt.title(r"$P(Y(\theta) < 0.3)$ conditioned on the evaluation points")
    plt.plot(theta_A_arr, p_theta, label=r"$P(Y(\theta) < 0.3)$")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$P$")
    plt.savefig("./task2b.pdf")
    plt.show()
    plt.clf()

    #Print max probability and for which theta
    max_theta = np.argmax(p_theta)
    max_p = p_theta[max_theta]
    print(f"\nMax: ({theta_A_arr[max_theta]:.3f}, {max_p:.3f})")


task2a()
