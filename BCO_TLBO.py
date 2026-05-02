import numpy as np
import random
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

# TLBO has NO algorithm-specific parameters (parameter-free algorithm)

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
# Encoding: Real-valued vector [0,1] per validator
# Threshold 0.5: value > 0.5 → validator selected
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
# TLBO SINGLE RUN
# Encoding: Real-valued [0,1] vector (parameter-free algorithm)
#
# Two Phases per generation:
# TEACHER PHASE: Learners move toward the best solution (teacher)
#   new_X = X + r * (Teacher - TF * Mean)
#   TF (Teaching Factor) = round(1 + rand()) ∈ {1, 2}
#
# LEARNER PHASE: Learners learn from random peers
#   If fitness(Xi) < fitness(Xj): new_X = X + r * (Xj - Xi)
#   Else:                          new_X = X + r * (Xi - Xj)
# =============================
def run_tlbo_once(stake, latency, reputation, print_gen=False):

    # Initialize population with real values in [0, 1]
    pop = np.random.rand(POP_SIZE, NUM_VALIDATORS)
    best_fit = float('-inf')
    fitness_history = []

    for gen in range(1, MAX_GEN + 1):

        # Evaluate all individuals
        fitnesses = []
        for i in range(POP_SIZE):
            fit, U, n, idx = evaluate(pop[i], stake, latency, reputation)
            fitnesses.append(fit)
            if fit > best_fit:
                best_fit, best_U, best_n, best_idx = fit, U, n, idx

        fitness_history.append(best_fit)

        if print_gen and gen % 20 == 0:
            print(f"  Gen {gen}: best fitness = {best_fit:.6f}")

        fitnesses = np.array(fitnesses)

        # Identify teacher: individual with best fitness
        teacher_idx = np.argmax(fitnesses)
        teacher = pop[teacher_idx]

        # Mean of the current population
        mean_pop = np.mean(pop, axis=0)

        # Teaching factor TF ∈ {1, 2} chosen randomly
        TF = round(1 + random.random())

        # ---- TEACHER PHASE ----
        new_pop = []
        for i in range(POP_SIZE):
            r = np.random.rand(NUM_VALIDATORS)
            # Learner moves toward teacher, away from scaled mean
            candidate = pop[i] + r * (teacher - TF * mean_pop)
            candidate = np.clip(candidate, 0, 1)

            # Greedy selection: keep candidate only if better
            fit_candidate, _, _, _ = evaluate(candidate, stake, latency, reputation)
            if fit_candidate > fitnesses[i]:
                new_pop.append(candidate)
                fitnesses[i] = fit_candidate
            else:
                new_pop.append(pop[i])

        pop = np.array(new_pop)

        # ---- LEARNER PHASE ----
        new_pop = []
        for i in range(POP_SIZE):
            # Pick a random peer (different from i)
            j = random.choice([x for x in range(POP_SIZE) if x != i])

            r = np.random.rand(NUM_VALIDATORS)

            if fitnesses[i] < fitnesses[j]:
                # Learn from the better peer
                candidate = pop[i] + r * (pop[j] - pop[i])
            else:
                # Move away from the worse peer
                candidate = pop[i] + r * (pop[i] - pop[j])

            candidate = np.clip(candidate, 0, 1)

            # Greedy selection
            fit_candidate, U, n, idx = evaluate(candidate, stake, latency, reputation)
            if fit_candidate > fitnesses[i]:
                new_pop.append(candidate)
                if fit_candidate > best_fit:
                    best_fit, best_U, best_n, best_idx = fit_candidate, U, n, idx
            else:
                new_pop.append(pop[i])

        pop = np.array(new_pop)

    return best_U, len(best_idx), best_n, [int(i) for i in best_idx], fitness_history

# =============================
# MULTI RUN
# =============================
def run_tlbo_multiple(stake, latency, reputation, name):
    utilities = []
    all_histories = []

    print("\n==================================================")
    print(f"TLBO - {name}")
    print("==================================================")

    for r in range(RUNS):
        if r == 0:
            U, m, n, idx, history = run_tlbo_once(stake, latency, reputation, print_gen=True)
            print("\nOptimal configuration found (Run 1):")
            print(f"  Utility U        = {U:.6f}")
            print(f"  Validators m     = {m}")
            print(f"  Transactions n   = {n}")
            print(f"  Selected indices : {idx}")
        else:
            U, m, n, idx, history = run_tlbo_once(stake, latency, reputation)

        utilities.append(U)
        all_histories.append(history)

    utilities = np.array(utilities)
    all_histories = np.array(all_histories)

    print(f"\nAverage Utility = {np.mean(utilities):.6f}")
    print(f"Best Utility    = {np.max(utilities):.6f}")

    # =============================
    # CONVERGENCE PLOT
    # =============================
    avg_curve = np.mean(all_histories, axis=0)

    plt.figure()
    for i in range(RUNS):
        plt.plot(all_histories[i], alpha=0.3, color='mediumpurple')
    plt.plot(avg_curve, linewidth=2, color='indigo', label='Average')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"TLBO Convergence ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    run_tlbo_multiple(stake, latency, reputation, "Setting 1")
    run_tlbo_multiple(stake_2, latency_2, reputation_2, "Setting 2")