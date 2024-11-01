import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


lam = 5 / 60  # Scale so both lamda and mu are in minutes^-1.
mu = 1 / 10

# Number of minutes in 50 days
T = 50 * 24 * 60


def simulate_realization():
    # Get how many customers arrived in 50 days
    n_customers = np.random.poisson(lam * T)
    # We know how many customers arrived in the 50 days, so the arrival times
    # are uniformly distributed.
    arrival_times = np.sort(np.random.uniform(0, T, size=n_customers))
    # We need at maximum n_customers treating times
    treatment_times = np.random.exponential(1 / mu, size=n_customers)
    release_time = arrival_times[0] + treatment_times[0]

    # At most 2*n_customers events will happen
    X = np.zeros(2 * n_customers)
    time = np.zeros(2 * n_customers)

    i_a = 1  # Arrival index
    i_r = 0  # Releasetime index

    # Init
    time[0] = arrival_times[0]
    X[0] = 1

    while i_a < n_customers and time[i_a + i_r] <= T:
        # If someone arrives before current service is done
        if arrival_times[i_a] < release_time:
            X[i_a + i_r] = X[i_a + i_r - 1] + 1
            time[i_a + i_r] = arrival_times[i_a]
            i_a += 1
        # Else the next event is that the service is done
        else:
            X[i_a + i_r] = X[i_a + i_r - 1] - 1
            time[i_a + i_r] = release_time
            i_r += 1
            if X[i_a + i_r - 1] == 0:
                # If X == 0, the next release time is the arrival_time of the next customer + its service time
                release_time = arrival_times[i_r] + treatment_times[i_r]
            else:
                # Else the next release time is the current time + next service time
                release_time = release_time + treatment_times[i_r]

    # Only return the relevant parts of the array, the rest is still zero
    return time[: i_a + i_r], X[: i_a + i_r]


def average_time_spent():
    t, x = simulate_realization()
    # Construct weights for the weighted average
    delta_t = np.zeros_like(t)
    delta_t[:-1] = t[1:] - t[:-1]
    delta_t[-1] = T - t[-1]

    # L is the average of X weighted by the time X stayed constant
    L = np.average(x, 0, delta_t)
    # We start a bit out in the array so the state has "stabilized"
    # L = np.average(x[100:], 0, delta_t[100:])

    # By Little's law
    return L / lam


def CI():
    L = np.zeros(30)
    for i in range(30):
        L[i] = average_time_spent()

    avg = np.average(L)
    N = L.size
    stdev = np.std(L, ddof=1)
    t_alpha = stats.t.ppf(0.975, N - 1)
    lower = avg - t_alpha * stdev / np.sqrt(N)
    upper = avg + t_alpha * stdev / np.sqrt(N)
    return avg, lower, upper


def plot_steps(axs, x: np.ndarray, y: np.ndarray, end: float, color, **kwargs):
    for i in range(len(x)):
        if i < len(x) - 1:
            line = axs.hlines(y[i], x[i], x[i + 1], linewidth=3, colors=color)
        else:
            line = axs.hlines(y[i], x[i], end, linewidth=3, colors=color)
    return line


def task1b():
    t, x = simulate_realization()
    t /= 60  # Convert to hours
    stop_time = 12  # minutes in 12 hours
    (indices,) = np.where(t <= stop_time)

    fig, axs = plt.subplots()
    line1 = plot_steps(axs, t[indices], x[indices], stop_time, "blue")
    axs.legend([line1], ["X"])
    axs.grid()
    fig.tight_layout()
    fig.savefig("./plots/task1b.png")

    avg, lower, upper = CI()
    print()
    print("Avg: ", avg)
    print(f"CI: [{lower:.3f}, {upper:.3f}]")


task1b()
