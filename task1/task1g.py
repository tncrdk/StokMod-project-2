import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


lam = 5 / 60  # Scale so both lamda and mu are in minutes^-1.
mu = 1 / 10
p = 0.8  # Probability that a patient is urgent

# Number of minutes in 50 days
T = 50 * 24 * 60


class Queue:
    def __init__(self, max_length: int) -> None:
        # Array with treatment_times
        self.p_queue = np.zeros(max_length, dtype=np.int64)
        self.p_front = 0
        self.p_back = 0

        # Array with treatment_times
        self.queue = np.zeros(max_length, dtype=np.int64)
        self.front = 0
        self.back = 0

        self.service = 0  # 0: None, 1: priority, 2: normal
        self.service_end = 0

    def append(self, arrival_time: float, treatment_time: float, priority: bool):
        if priority:
            self.p_queue[self.p_back] = treatment_time
            self.p_back += 1
            if self.p_back == len(self.p_queue):
                self.p_back = 0
        else:
            self.queue[self.back] = treatment_time
            self.back += 1
            if self.back == len(self.queue):
                self.back = 0

        # Start treatment
        if self.service == 0:
            self.start_service(arrival_time, treatment_time, priority)
        elif self.service == 2 and priority:
            self.start_service(arrival_time, treatment_time, priority)

    def start_service(self, current_time, treatment_time, priority):
        if priority:
            self.service = 1
        else:
            self.service = 2
        self.service_end = current_time + treatment_time

    def end_service(self):
        pass

    def pop(self, priority: bool):
        pass


def simulate_realization():
    # Get how many customers arrived in 50 days
    n_patients = np.random.poisson(lam * T)
    n_urgent_patients = np.random.binomial(n_patients, p)
    n_normal_patients = n_patients - n_urgent_patients
    # We know how many customers arrived in the 50 days, so the arrival times
    # are uniformly distributed.
    urgent_arrival_times = np.sort(np.random.uniform(0, T, size=n_urgent_patients))
    normal_arrival_times = np.sort(np.random.uniform(0, T, size=n_normal_patients))
    # We need at maximum n_patients**2 treating times
    urgent_treatment_times = np.random.exponential(1 / mu, size=n_urgent_patients)
    normal_treatment_times = np.random.exponential(1 / mu, size=n_patients)

    # At most 2*n_customers events will happen
    N = np.zeros(2 * n_normal_patients)
    U = np.zeros(2 * n_urgent_patients)
    X = np.zeros(2 * n_patients)
    time_U = np.zeros(2 * n_urgent_patients)
    time_N = np.zeros(2 * n_normal_patients)
    time_X = np.zeros(2 * n_patients)

    u_arrival = 1  # Arrival index
    u_treatment = 0  # Releasetime index

    i_arrival = 1  # Arrival index
    i_treatment = 0  # Releasetime index

    # Init
    X[0] = 1
    if urgent_arrival_times[0] <= normal_arrival_times[0]:
        time[0] = urgent_arrival_times[0]
        U[0] = 1
    else:
        pass

    while i_a < n_patients and time[i_a + i_r] <= T:
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
