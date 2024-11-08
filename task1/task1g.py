import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class Queue:
    def __init__(self, max_length) -> None:
        self.max_length = max_length

        # Array with treatment_times
        self.queue = np.zeros(max_length)
        self.head = 0
        self.back = 0

    def pop(self) -> float:
        treatment_time = self.queue[self.head]
        self.head += 1
        return treatment_time

    def update_treatment_time(self, treatment_time: float):
        self.queue[self.head] = treatment_time

    def append(self, treatment_time: float):
        self.queue[self.back] = treatment_time
        self.back += 1

    @property
    def treatment_time(self):
        return self.queue[self.head]

    @property
    def empty(self):
        return self.head == self.back

    @property
    def length(self):
        if self.head <= self.back:
            return self.back - self.head
        return self.back - self.head + self.max_length


class UCC:
    def __init__(self, m: float, l: float, p: float, T: float) -> None:
        self.m = m
        self.l = l
        self.p = p
        self.T = T
        self.current_time = 0

        # Init random variables
        self.n_patients = np.random.poisson(self.l * T)
        self.classification = np.random.binomial(1, self.p, size=self.n_patients)
        self.n_urgent_patients = np.count_nonzero(self.classification)
        self.n_normal_patients = self.n_patients - self.n_urgent_patients

        # We know how many customers arrived in the 50 days, so the arrival times
        # are uniformly distributed
        self.arrival_times = np.sort(np.random.uniform(0, T, size=self.n_patients))

        # The treatment times are exponentially distributed
        self.treatment_times = np.random.exponential(1 / self.m, size=self.n_patients)

        # Array with treatment_times
        self.priority_queue = Queue(self.n_urgent_patients)
        self.normal_queue = Queue(self.n_normal_patients)

        # Index into how far we have many patients have arrived
        self.i_patients = 0

        self.service = 0  # 0: None, 1: priority, 2: normal
        self.service_endtime = np.inf

    def next_step(self) -> bool:
        if (
            self.i_patients < self.n_patients - 1
            and self.arrival_times[self.i_patients + 1] <= self.service_endtime
        ):
            self.current_time = self.arrival_times[self.i_patients + 1]
            self.new_arrival()
        else:
            self.current_time = self.service_endtime
            # If the current_time exceed T, we are done with the simulation
            if self.current_time > self.T:
                return True
            self.complete_service()
        return False

    def new_arrival(self):
        self.i_patients += 1
        priority = self.classification[self.i_patients]
        treatment_time = self.treatment_times[self.i_patients]
        if priority:
            self.priority_queue.append(treatment_time)
        else:
            self.normal_queue.append(treatment_time)

        if self.service == 0:
            self.start_service()
        elif self.service == 2 and priority:
            self.interrupt_service()

    def interrupt_service(self):
        # Create new treatment time to the interrupted patient
        self.normal_queue.update_treatment_time(np.random.exponential(1 / self.m))
        self.start_service()

    def start_service(self):
        assert not (
            self.priority_queue.empty and self.normal_queue.empty
        ), "We can not start a service if the queue is empty."

        if not self.priority_queue.empty:
            self.service = 1
            treatment_time = self.priority_queue.treatment_time
        else:
            self.service = 2
            treatment_time = self.normal_queue.treatment_time
        self.service_endtime = self.current_time + treatment_time

    def complete_service(self):
        assert not (
            self.priority_queue.empty and self.normal_queue.empty
        ), "Can not complete a service if none are being served."
        # One patient walk out
        if self.service == 1:
            self.priority_queue.pop()
        else:
            self.normal_queue.pop()

        # One new patient starts recieving treatment if there is a patient in the queue
        if not (self.priority_queue.empty and self.normal_queue.empty):
            self.start_service()
        else:
            # If there are no patients, none are treated
            self.service = 0
            self.service_endtime = np.inf

    @property
    def N(self):
        return self.normal_queue.length

    @property
    def U(self):
        return self.priority_queue.length

    @property
    def X(self):
        return self.N + self.U


def simulate_realization(m, l, p, T) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    ucc = UCC(m, l, p, T)

    # At most 2*n_customers events will happen
    N = np.zeros(2 * ucc.n_normal_patients + 1)
    U = np.zeros(2 * ucc.n_urgent_patients + 1)
    X = np.zeros(2 * ucc.n_patients + 1)
    time_N = np.zeros_like(N)
    time_U = np.zeros_like(U)
    time_X = np.zeros_like(X)

    iN, iU, iX = 1, 1, 1

    done = False
    while not done:
        done = ucc.next_step()
        if done:
            break
        n, u, x = ucc.N, ucc.U, ucc.X
        current_time = ucc.current_time
        if N[iN - 1] != n:
            N[iN] = n
            time_N[iN] = current_time
            iN += 1
        if U[iU - 1] != u:
            U[iU] = u
            time_U[iU] = current_time
            iU += 1
        if X[iX - 1] != x:
            X[iX] = x
            time_X[iX] = current_time
            iX += 1

    # Only return the relevant parts of the array, the rest is still zero
    return (
        (time_N[:iN], N[:iN]),
        (time_U[:iU], U[:iU]),
        (time_X[:iX], X[:iX]),
    )


def average_time_spent(lam, mu, p, T):
    n, u, x = simulate_realization(mu, lam, p, T)

    # N
    # Construct weights for the weighted average
    n_delta_t = np.zeros_like(n[0])
    n_delta_t[:-1] = n[0][1:] - n[0][:-1]
    n_delta_t[-1] = T - n[0][-1]

    # nL is the average of N weighted by the time X stayed constant
    nL = np.average(n[1][20:], 0, n_delta_t[20:])

    # U
    u_delta_t = np.zeros_like(u[0])
    u_delta_t[:-1] = u[0][1:] - u[0][:-1]
    u_delta_t[-1] = T - u[0][-1]

    uL = np.average(u[1][20:], 0, u_delta_t[20:])

    # X
    x_delta_t = np.zeros_like(x[0])
    x_delta_t[:-1] = x[0][1:] - x[0][:-1]
    x_delta_t[-1] = T - x[0][-1]

    xL = np.average(x[1][20:], 0, x_delta_t[20:])

    # By Little's law
    return nL / (lam * (1 - p)), uL / (lam * p), xL / lam


def CI(lam, mu, p, T):
    N = 30
    W = np.zeros((3, N))
    for i in range(N):
        W[:, i] = average_time_spent(lam, mu, p, T)
        print(W[:, i])

    avg = np.average(W, axis=1)
    stdev = np.std(W, ddof=1, axis=1)
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


def task1g():
    lam = 5 / 60  # Scale so both lamda and mu are in minutes^-1.
    mu = 1 / 10
    p = 0.8  # Probability that a patient is urgent

    # Number of minutes in 50 days
    T = 50 * 24 * 60

    stop_time = 12  # minutes in 12 hours
    n, u, x = simulate_realization(mu, lam, p, stop_time * 60)

    fig, axs = plt.subplots()
    line1 = plot_steps(axs, u[0] / 60, u[1], stop_time, "blue")
    line2 = plot_steps(axs, n[0] / 60, n[1], stop_time, "red")
    axs.legend([line1, line2], ["U(t)", "N(t)"])
    axs.grid()
    axs.set_xlabel("Time [h]")
    axs.set_ylabel("Number of patients in the system")
    fig.suptitle(r"Time evolution of $X(t)$ and $N(t)$, $p=0.8$")
    fig.tight_layout()
    fig.savefig("./task1g.pdf")
    plt.show()

    print()
    avg, lower, upper = CI(lam, mu, p, T)
    parameters = ["W_N", "W_U", "W_X"]
    exact_values = [
        mu / ((mu - lam) * (mu - p * lam)),
        1 / (mu - p * lam),
        1 / (mu - lam),
    ]
    print("=" * 105)
    print(
        f"{'Parameter':^20}|{'Avg':^20}|{'Lower':^20}|{'Upper':^20}|{'Exact value':^20}|"
    )
    print("-" * 105)
    for i in range(3):
        print(
            f"{parameters[i]:^20}|{avg[i]:^20.2f}|{lower[i]:^20.2f}|{upper[i]:^20.2f}|{exact_values[i]:^20.2f}|"
        )


task1g()
