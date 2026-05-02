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

# PSO Specific Parameters
W       = 0.7    # Inertia weight: controls momentum of particle movement
C1      = 1.5    # Cognitive coefficient: attraction toward personal best
C2      = 1.5    # Social coefficient: attraction toward global best
V_MAX   = 0.5    # Maximum velocity clamp to prevent explosion

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
# Encoding: Real-valued position vector [0,1] per validator
# Threshold 0.5: position > 0.5 → validator selected
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
# PSO SINGLE RUN
# Encoding: Real-valued position in [0,1] for each validator gene
# Each particle has a position (solution) and a velocity
# Velocity update: v = W*v + C1*r1*(pbest-x) + C2*r2*(gbest-x)
# Position update: x = x + v, then clipped to [0,1]
# =============================
def run_pso_once(stake, latency, reputation, print_gen=False):

    # Initialize positions and velocities randomly
    positions  = np.random.rand(POP_SIZE, NUM_VALIDATORS)
    velocities = np.random.uniform(-V_MAX, V_MAX, (POP_SIZE, NUM_VALIDATORS))

    # Personal best position and fitness for each particle
    pbest_pos = positions.copy()
    pbest_fit = np.full(POP_SIZE, float('-inf'))

    # Global best
    gbest_pos = None
    gbest_fit = float('-inf')

    fitness_history = []

    for gen in range(1, MAX_GEN + 1):

        for i in range(POP_SIZE):
            fit, U, n, idx = evaluate(positions[i], stake, latency, reputation)

            # Update personal best
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i].copy()

            # Update global best
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = positions[i].copy()
                best_U, best_n, best_idx = U, n, idx

        fitness_history.append(gbest_fit)

        if print_gen and gen % 20 == 0:
            print(f"  Gen {gen}: best fitness = {gbest_fit:.6f}")

        # ---- Velocity and Position Update ----
        for i in range(POP_SIZE):
            r1 = np.random.rand(NUM_VALIDATORS)   # Random cognitive factor
            r2 = np.random.rand(NUM_VALIDATORS)   # Random social factor

            # Standard PSO velocity update equation
            velocities[i] = (W * velocities[i]
                             + C1 * r1 * (pbest_pos[i] - positions[i])
                             + C2 * r2 * (gbest_pos   - positions[i]))

            # Clamp velocity to [-V_MAX, V_MAX]
            velocities[i] = np.clip(velocities[i], -V_MAX, V_MAX)

            # Update position
            positions[i] += velocities[i]

            # Clip position to valid range [0, 1]
            positions[i] = np.clip(positions[i], 0, 1)

    return best_U, len(best_idx), best_n, [int(i) for i in best_idx], fitness_history

# =============================
# MULTI RUN
# =============================
def run_pso_multiple(stake, latency, reputation, name):
    utilities = []
    all_histories = []

    print("\n==================================================")
    print(f"PSO - {name}")
    print("==================================================")

    for r in range(RUNS):
        if r == 0:
            U, m, n, idx, history = run_pso_once(stake, latency, reputation, print_gen=True)
            print("\nOptimal configuration found (Run 1):")
            print(f"  Utility U        = {U:.6f}")
            print(f"  Validators m     = {m}")
            print(f"  Transactions n   = {n}")
            print(f"  Selected indices : {idx}")
        else:
            U, m, n, idx, history = run_pso_once(stake, latency, reputation)

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
        plt.plot(all_histories[i], alpha=0.3, color='coral')
    plt.plot(avg_curve, linewidth=2, color='darkred', label='Average')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"PSO Convergence ({name})")
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

    run_pso_multiple(stake, latency, reputation, "Setting 1")
    run_pso_multiple(stake_2, latency_2, reputation_2, "Setting 2")