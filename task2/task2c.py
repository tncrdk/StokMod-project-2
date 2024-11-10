import numpy as np
import matplotlib.pyplot as plt
import utils as u


def task2c():
    #Initialization of arrays and variables
    theta_A_arr = np.linspace(0.25, 0.5, 51)
    theta_B_arr = np.array([0.3, 0.33, 0.35, 0.39, 0.41, 0.45])
    #Reordering of theta_A to not include theta values from theta_B
    theta_A_arr = theta_A_arr[~np.isin(theta_A_arr, theta_B_arr)]
    x_B_arr = np.array([0.5, 0.4, 0.32, 0.40, 0.35, 0.6])
    mu_A = np.full_like(theta_A_arr, 0.5)
    mu_B = np.full_like(theta_B_arr, 0.5)
    variance = 0.5**2
    k = 15

    #Get covariance matrices
    S_A, S_B, S_AB = u.construct_covariance_matrix(
        theta_A_arr, theta_B_arr, variance, k
    )

    #Get conditional expectation and variance
    S_B_inv = np.linalg.inv(S_B)
    E_AIB = u.get_conditional_expectance(mu_A, mu_B, x_B_arr, S_AB, S_B_inv)
    V_AIB = u.get_conditional_variance(S_A, S_AB, S_B_inv)

    #Confidence intervals
    lower, upper = u.CI(E_AIB, V_AIB, 0.9)

    #Add back the values from theta_B to theta_A and other arrays where necessary
    theta_A_arr_plot = np.concatenate((theta_A_arr, theta_B_arr))
    E_AIB_plot = np.concatenate((E_AIB, x_B_arr))
    lower = np.concatenate((lower, x_B_arr))
    upper = np.concatenate((upper, x_B_arr))

    #Reordering of the arrays to get correct plots
    sort_indices = np.argsort(theta_A_arr_plot)
    theta_A_arr_plot = theta_A_arr_plot[sort_indices]
    E_AIB_plot = E_AIB_plot[sort_indices]
    lower = lower[sort_indices]
    upper = upper[sort_indices]

    plt.title(r"Prediction of $Y(\theta)$ and $90$% confidence intervals")
    plt.plot(theta_A_arr_plot, E_AIB_plot, label="Y")
    plt.plot(theta_A_arr_plot, lower, label="Lower")
    plt.plot(theta_A_arr_plot, upper, label="Upper")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$Y$")
    plt.savefig("./task2c_CI.pdf")
    plt.show()
    plt.clf()

    #P < 0.3
    p_theta = u.P(0.3, E_AIB, V_AIB)
    plt.title(r"$P(Y(\theta) < 0.3)$ conditioned on the evaluation points")
    plt.plot(theta_A_arr, p_theta)
    plt.grid()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$P$")
    plt.savefig("./task2c_p.pdf")
    plt.show()
    plt.clf()

    #Get max theta and probability
    max_theta = np.argmax(p_theta)
    max_p = p_theta[max_theta]
    print(f"\nMax: ({theta_A_arr[max_theta]:.3f}, {max_p:.3f})")


task2c()
