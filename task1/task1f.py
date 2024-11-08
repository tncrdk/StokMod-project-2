import numpy as np
import matplotlib.pyplot as plt


def W_U(p: np.ndarray, lam: float, mu: float) -> np.ndarray:
    return 1 / (mu - p * lam)


def W_N(p: np.ndarray, lam: float, mu: float) -> np.ndarray:
    return mu / ((mu - lam) * (mu - lam * p))


def task1f():
    lam = 5
    mu = 6
    N = 100

    p = np.linspace(0, 1, N)
    w_u = W_U(p, lam, mu)
    w_n = W_N(p, lam, mu)

    plt.plot(p, w_u, label=r"$W_U$")
    plt.plot(p, w_n, label=r"$W_N$")
    plt.legend()
    plt.title(r"Average time spent in the UCC, $W_{\{U,N\}}(p)$")
    plt.xlabel("Probability of being an urgent patient")
    plt.ylabel("W [h]")
    plt.grid()
    plt.savefig("./task1f.pdf")
    plt.show()

task1f()
