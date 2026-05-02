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

# ABC Specific Parameters
# Colony size = POP_SIZE; split equally into employed and onlooker bees
NUM_EMPLOYED  = POP_SIZE // 2   # Employed bees: one per food source
NUM_ONLOOKER  = POP_SIZE // 2   # Onlooker bees: select sources by roulette
LIMIT         = NUM_VALIDATORS * NUM_EMPLOYED  # Abandonment limit for scout phase

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
# Encoding: Real-valued food source position [0,1] per validator
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
# NEIGHBOUR SOLUTION GENERATOR
# Standard ABC position update:
# v_ij = x_ij + phi_ij * (x_ij - x_kj)
# phi ∈ [-1, 1], k ≠ i, j chosen randomly
# =============================
def generate_neighbour(source, all_sources, source_idx):
    k = random.choice([x for x in range(len(all_sources)) if x != source_idx])
    j = random.randint(0, NUM_VALIDATORS - 1)     # Randomly chosen dimension
    phi = random.uniform(-1, 1)

    neighbour = source.copy()
    neighbour[j] = source[j] + phi * (source[j] - all_sources[k][j])
    neighbour[j] = np.clip(neighbour[j], 0, 1)

    return neighbour

# =============================
# ABC SINGLE RUN
# Encoding: Real-valued [0,1] food source position
#
# Three Bee Types:
# EMPLOYED BEES: Exploit current food sources, generate neighbours
# ONLOOKER BEES: Select sources via roulette (fitness-proportional), exploit further
# SCOUT BEES: Abandon exhausted sources (trial > LIMIT), initialize new random sources
# =============================
def run_abc_once(stake, latency, reputation, print_gen=False):

    # Initialize food sources (employed bee positions) randomly
    sources  = np.random.rand(NUM_EMPLOYED, NUM_VALIDATORS)
    fitnesses = np.full(NUM_EMPLOYED, float('-inf'))
    trials    = np.zeros(NUM_EMPLOYED)  # Counter for failed improvement attempts

    # Evaluate initial food sources
    best_extras = {}
    for i in range(NUM_EMPLOYED):
        fit, U, n, idx = evaluate(sources[i], stake, latency, reputation)
        fitnesses[i] = fit
        best_extras[i] = (U, n, idx)

    best_fit = np.max(fitnesses)
    best_src = np.argmax(fitnesses)
    best_U, best_n, best_idx = best_extras[best_src]

    fitness_history = []

    for gen in range(1, MAX_GEN + 1):

        # ---- EMPLOYED BEE PHASE ----
        # Each employed bee searches neighbourhood of its food source
        for i in range(NUM_EMPLOYED):
            neighbour = generate_neighbour(sources[i], sources, i)
            fit_n, U, n, idx = evaluate(neighbour, stake, latency, reputation)

            if fit_n > fitnesses[i]:
                # Greedy selection: move to better neighbour
                sources[i]  = neighbour
                fitnesses[i] = fit_n
                trials[i]   = 0
                best_extras[i] = (U, n, idx)
            else:
                # Source not improved; increment trial counter
                trials[i] += 1

        # ---- ONLOOKER BEE PHASE ----
        # Select food sources proportional to fitness (roulette wheel)
        # Negative fitnesses handled by shifting all values above zero
        shifted = fitnesses - np.min(fitnesses) + 1e-10
        probs = shifted / np.sum(shifted)

        for _ in range(NUM_ONLOOKER):
            # Roulette wheel selection
            i = np.random.choice(NUM_EMPLOYED, p=probs)
            neighbour = generate_neighbour(sources[i], sources, i)
            fit_n, U, n, idx = evaluate(neighbour, stake, latency, reputation)

            if fit_n > fitnesses[i]:
                sources[i]   = neighbour
                fitnesses[i] = fit_n
                trials[i]    = 0
                best_extras[i] = (U, n, idx)
            else:
                trials[i] += 1

        # ---- SCOUT BEE PHASE ----
        # Abandon exhausted sources and reinitialize randomly
        for i in range(NUM_EMPLOYED):
            if trials[i] > LIMIT:
                sources[i]   = np.random.rand(NUM_VALIDATORS)
                fitnesses[i], U, n, idx = evaluate(sources[i], stake, latency, reputation)[:4]
                trials[i]    = 0
                best_extras[i] = (fitnesses[i], n, idx) if fitnesses[i] > float('-inf') else (0, 0, [])

        # Track global best across all sources
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fit:
            best_fit = fitnesses[current_best_idx]
            best_U, best_n, best_idx = best_extras[current_best_idx]

        fitness_history.append(best_fit)

        if print_gen and gen % 20 == 0:
            print(f"  Gen {gen}: best fitness = {best_fit:.6f}")

    return best_U, len(best_idx), best_n, [int(i) for i in best_idx], fitness_history

# =============================
# MULTI RUN
# =============================
def run_abc_multiple(stake, latency, reputation, name):
    utilities = []
    all_histories = []

    print("\n==================================================")
    print(f"ABC - {name}")
    print("==================================================")

    for r in range(RUNS):
        if r == 0:
            U, m, n, idx, history = run_abc_once(stake, latency, reputation, print_gen=True)
            print("\nOptimal configuration found (Run 1):")
            print(f"  Utility U        = {U:.6f}")
            print(f"  Validators m     = {m}")
            print(f"  Transactions n   = {n}")
            print(f"  Selected indices : {idx}")
        else:
            U, m, n, idx, history = run_abc_once(stake, latency, reputation)

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
        plt.plot(all_histories[i], alpha=0.3, color='gold')
    plt.plot(avg_curve, linewidth=2, color='goldenrod', label='Average')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"ABC Convergence ({name})")
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

    run_abc_multiple(stake, latency, reputation, "Setting 1")
    run_abc_multiple(stake_2, latency_2, reputation_2, "Setting 2")