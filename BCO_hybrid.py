# BCGA + RCGA

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

# Phase split: first PHASE_SWITCH generations use BCGA, rest use RCGA
PHASE_SWITCH = 50

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
# Encoding: Real-valued [0,1]; threshold 0.5 for selection
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
# BCGA CROSSOVER
# Blend crossover (BLX-alpha): standard operator for binary-coded GA
# Produces child as convex combination of two parents
# =============================
def bcga_crossover(p1, p2):
    alpha = random.random()
    child = alpha * p1 + (1 - alpha) * p2
    return child

# =============================
# BCGA MUTATION
# Uniform random reset: flips gene to a random value in [0,1]
# =============================
def bcga_mutate(individual, mut_prob=0.08):
    mutated = individual.copy()
    for j in range(len(mutated)):
        if random.random() < mut_prob:
            mutated[j] = random.random()
    return mutated

# =============================
# SBX CROSSOVER (RCGA Phase)
# Simulated Binary Crossover: mimics single-point crossover on real values
# =============================
def sbx_crossover(p1, p2, eta_c=2.0):
    child1 = np.zeros(len(p1))
    child2 = np.zeros(len(p1))
    for i in range(len(p1)):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta_c + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
        child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(child1, 0, 1), np.clip(child2, 0, 1)

# =============================
# POLYNOMIAL MUTATION (RCGA Phase)
# Perturbation using polynomial probability distribution
# =============================
def poly_mutate(individual, eta_m=20.0, mut_prob=0.08):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mut_prob:
            u = random.random()
            delta = (2*u)**(1/(eta_m+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(eta_m+1))
            mutated[i] = np.clip(mutated[i] + delta, 0, 1)
    return mutated

# =============================
# TOURNAMENT SELECTION
# =============================
def tournament_select(pop, fitnesses, k=2):
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda idx: fitnesses[idx])
    return pop[best]

# =============================
# HYBRID BCGA + RCGA SINGLE RUN
# Phase 1 (gen 1 to PHASE_SWITCH)   → BCGA: broad exploration with blend crossover
# Phase 2 (gen PHASE_SWITCH+1 to end) → RCGA: fine-tuned exploitation with SBX + poly mutation
# =============================
def run_hybrid_once(stake, latency, reputation, print_gen=False):

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
            phase = "BCGA" if gen <= PHASE_SWITCH else "RCGA"
            print(f"  Gen {gen} [{phase}]: best fitness = {best_fit:.6f}")

        new_pop = []

        if gen <= PHASE_SWITCH:
            # ---- PHASE 1: BCGA ----
            # Blend crossover + uniform mutation for global search
            while len(new_pop) < POP_SIZE:
                p1 = pop[random.randint(0, POP_SIZE - 1)]
                p2 = pop[random.randint(0, POP_SIZE - 1)]
                child = bcga_crossover(p1, p2)
                child = bcga_mutate(child, mut_prob=0.08)
                new_pop.append(child)
        else:
            # ---- PHASE 2: RCGA ----
            # SBX crossover + polynomial mutation for local refinement
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop, fitnesses)
                p2 = tournament_select(pop, fitnesses)
                child1, child2 = sbx_crossover(p1, p2, eta_c=2.0)
                child1 = poly_mutate(child1, eta_m=20.0, mut_prob=0.05)
                child2 = poly_mutate(child2, eta_m=20.0, mut_prob=0.05)
                new_pop.append(child1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(child2)

        pop = np.array(new_pop)

    return best_U, len(best_idx), best_n, [int(i) for i in best_idx], fitness_history

# =============================
# MULTI RUN
# =============================
def run_hybrid_multiple(stake, latency, reputation, name):
    utilities = []
    all_histories = []

    print("\n==================================================")
    print(f"Hybrid BCGA+RCGA - {name}")
    print("==================================================")

    for r in range(RUNS):
        if r == 0:
            U, m, n, idx, history = run_hybrid_once(stake, latency, reputation, print_gen=True)
            print("\nOptimal configuration found (Run 1):")
            print(f"  Utility U        = {U:.6f}")
            print(f"  Validators m     = {m}")
            print(f"  Transactions n   = {n}")
            print(f"  Selected indices : {idx}")
        else:
            U, m, n, idx, history = run_hybrid_once(stake, latency, reputation)

        utilities.append(U)
        all_histories.append(history)

    utilities = np.array(utilities)
    all_histories = np.array(all_histories)

    print(f"\nAverage Utility = {np.mean(utilities):.6f}")
    print(f"Best Utility    = {np.max(utilities):.6f}")

    # =============================
    # CONVERGENCE PLOT
    # Phase boundary marked with vertical dashed line
    # =============================
    avg_curve = np.mean(all_histories, axis=0)

    plt.figure()
    for i in range(RUNS):
        plt.plot(all_histories[i], alpha=0.3, color='mediumseagreen')
    plt.plot(avg_curve, linewidth=2, color='darkgreen', label='Average')
    plt.axvline(x=PHASE_SWITCH, color='red', linestyle='--', linewidth=1.5, label=f'Phase switch (gen {PHASE_SWITCH})')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Hybrid BCGA+RCGA Convergence ({name})")
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

    run_hybrid_multiple(stake, latency, reputation, "Setting 1")
    run_hybrid_multiple(stake_2, latency_2, reputation_2, "Setting 2")