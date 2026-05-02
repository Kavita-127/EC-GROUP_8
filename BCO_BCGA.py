import numpy as np
import random
import time
import matplotlib.pyplot as plt

# =============================
# PARAMETERS
# =============================
NUM_VALIDATORS = 25
POP_SIZE = 60
MAX_GEN = 100
RUNS = 20

v, M = 5, 20
T_MIN, T_MAX = 200, 400

# System Parameters
O = 0.5
rd = 1.2
ru = 1.3
K = 100
psi = 0.001

# =============================
# SETTINGS
# =============================
stake = np.array([45,60,72,55,80,66,49,77,68,59,62,74,53,70,65,58,61,79,67,52,73,69,64,71,56])
latency = np.array([0.5,0.7,0.6,0.8,0.4,0.55,0.9,0.45,0.5,0.65,0.6,0.52,0.7,0.48,0.58,0.62,0.66,0.47,0.59,0.73,0.51,0.57,0.63,0.49,0.68])
reputation = np.array([0.82,0.76,0.88,0.74,0.91,0.85,0.79,0.9,0.83,0.77,0.81,0.86,0.75,0.89,0.84,0.78,0.8,0.92,0.87,0.73,0.88,0.85,0.82,0.9,0.76])

stake_2 = stake + 5
latency_2 = latency - 0.05
reputation_2 = reputation + 0.03

# =============================
# FITNESS FUNCTION
# =============================
def evaluate(solution, stake, latency_arr, reputation_arr):
    indices = np.where(solution > 0.5)[0]
    m = len(indices)

    if m == 0:
        return -1, 0, 0, []

    n = int(200 + m * 5)

    if m < v or m > M or n < T_MIN or n > T_MAX:
        return -1, 0, n, indices

    network_delay = (O / rd) + (O / ru)
    total_latency = network_delay + np.mean(latency_arr[indices])

    cost = psi * m * K

    security = (np.mean(stake[indices]) / 100) * 0.5 + np.mean(reputation_arr[indices]) * 0.5

    validator_bonus = m / NUM_VALIDATORS

    U = (0.5 * security) + (0.3 * validator_bonus) - (0.1 * (total_latency / 2)) - (0.1 * cost)

    return U, U, n, indices

# =============================
# BCGA SINGLE RUN
# =============================
def run_bcga_once(stake, latency, reputation, print_gen=False):

    pop = np.random.rand(POP_SIZE, NUM_VALIDATORS)
    best_fit = float('-inf')

    fitness_history = []

    for gen in range(1, MAX_GEN + 1):
        for i in range(POP_SIZE):
            fit, U, n, idx = evaluate(pop[i], stake, latency, reputation)

            if fit > best_fit:
                best_fit, best_U, best_n, best_idx = fit, U, n, idx

        fitness_history.append(best_fit)

        if print_gen and gen % 20 == 0:
            print(f"  Gen {gen}: best fitness = {best_fit:.6f}")

        new_pop = []

        while len(new_pop) < POP_SIZE:
            p1 = pop[random.randint(0, POP_SIZE-1)]
            p2 = pop[random.randint(0, POP_SIZE-1)]

            alpha = random.random()
            child = alpha * p1 + (1 - alpha) * p2

            for j in range(NUM_VALIDATORS):
                if random.random() < 0.08:
                    child[j] = random.random()

            new_pop.append(child)

        pop = np.array(new_pop)

    return best_U, len(best_idx), best_n, [int(i) for i in best_idx], fitness_history

# =============================
# MULTI RUN (CUSTOM OUTPUT)
# =============================
def run_bcga_multiple(stake, latency, reputation, name):
    utilities = []
    all_histories = []

    print("\n==================================================")
    print(name)
    print("==================================================")

    for r in range(RUNS):
        if r == 0:
            U, m, n, idx, history = run_bcga_once(stake, latency, reputation, print_gen=True)

            print("\nOptimal configuration found (Run 1):")
            print(f"  Utility U = {U:.6f}")
            print(f"  Number of validators m = {m}")
            print(f"  Transactions per block n = {n}")
            print(f"  Selected validator indices (0-based): {idx}")
        else:
            U, m, n, idx, history = run_bcga_once(stake, latency, reputation)

        utilities.append(U)
        all_histories.append(history)

    utilities = np.array(utilities)
    all_histories = np.array(all_histories)

    # ONLY REQUIRED PRINTS
    print(f"\nAverage Utility = {np.mean(utilities):.6f}")
    print(f"Best Utility = {np.max(utilities):.6f}")

    # =============================
    # CONVERGENCE PLOT
    # =============================
    avg_curve = np.mean(all_histories, axis=0)

    plt.figure()

    for i in range(RUNS):
        plt.plot(all_histories[i], alpha=0.3)

    plt.plot(avg_curve, linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"BCGA Convergence ({name})")
    plt.grid(True)
    plt.show()

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    run_bcga_multiple(stake, latency, reputation, "Setting 1")
    run_bcga_multiple(stake_2, latency_2, reputation_2, "Setting 2")
