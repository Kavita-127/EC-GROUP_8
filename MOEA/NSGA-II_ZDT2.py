# ==========================================================
# IMPORTS
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================================
# PARAMETERS
# ==========================================================
POP_SIZE = 100
GENS     = 200
DIM      = 30

LOW  = 0
HIGH = 1

CROSS_PROB = 0.9
MUT_PROB   = 0.1

ETA_C = 15
ETA_M = 20


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def create_population():
    return np.random.uniform(LOW, HIGH, size=(POP_SIZE, DIM))


# ==========================================================
# OBJECTIVE FUNCTION (ZDT2)
# ==========================================================
def evaluate(ind):
    ind = np.array(ind)
    n   = len(ind)

    f1 = ind[0]
    g  = 1 + (9 / (n - 1)) * np.sum(ind[1:])
    h  = 1 - (f1 / g) ** 2
    f2 = g * h

    return f1, f2


# ==========================================================
# DOMINATION CHECK
# ==========================================================
def dominates(a, b):
    """Return True if a dominates b (minimization)."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


# ==========================================================
# FAST NON-DOMINATED SORT
# ==========================================================
def non_dominated_sort(F):
    N = len(F)

    S    = [[] for _ in range(N)]
    n    = [0] * N
    rank = [0] * N

    fronts = [[]]

    # Stage 1
    for p in range(N):
        S[p] = []
        n[p] = 0

        for q in range(N):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    # Stage 2
    i = 0
    while fronts[i]:
        Q = []

        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)

        i += 1
        fronts.append(Q)

    fronts.pop()
    return fronts, rank


# ==========================================================
# CROWDING DISTANCE
# ==========================================================
def crowding_dist(front, F):
    if len(front) == 0:
        return np.array([])

    front  = np.array(front)
    values = F[front]
    r, M   = values.shape

    if r == 1:
        return np.array([np.inf])

    dist = np.zeros(r)

    for m in range(M):
        sorted_idx = np.argsort(values[:, m])
        sorted_v   = values[sorted_idx]

        dist[sorted_idx[0]]  = np.inf
        dist[sorted_idx[-1]] = np.inf

        f_min = sorted_v[0,  m]
        f_max = sorted_v[-1, m]

        if f_max == f_min:
            continue

        prev_vals  = sorted_v[:-2, m]
        next_vals  = sorted_v[2:,  m]
        increments = (next_vals - prev_vals) / (f_max - f_min)

        dist[sorted_idx[1:-1]] += increments

    return dist


# ==========================================================
# CROWDED BINARY TOURNAMENT
# ==========================================================
def crowded_binary_tournament(pop_size, ranks, cd):
    N = len(ranks)

    i_arr = np.random.randint(0, N, pop_size)
    j_arr = np.random.randint(0, N, pop_size)

    collisions = i_arr == j_arr
    j_arr[collisions] = (j_arr[collisions] + np.random.randint(1, N, collisions.sum())) % N

    winners = np.where(
        ranks[i_arr] < ranks[j_arr], i_arr,
        np.where(
            ranks[j_arr] < ranks[i_arr], j_arr,
            np.where(
                cd[i_arr] > cd[j_arr], i_arr,
                np.where(
                    cd[j_arr] > cd[i_arr], j_arr,
                    np.where(np.random.rand(pop_size) < 0.5, i_arr, j_arr)
                )
            )
        )
    )

    return winners


# ==========================================================
# SELECTION
# ==========================================================
def tournament_select(population, F):
    fronts, rank = non_dominated_sort(F)

    cd = np.zeros(len(population))
    for front in fronts:
        cd[front] = crowding_dist(front, F)

    selected_indices = crowded_binary_tournament(
        pop_size=len(population),
        ranks=np.array(rank),
        cd=cd
    )

    pool = population[selected_indices]
    return pool


# ==========================================================
# CROSSOVER (SBX)
# ==========================================================
def crossover(p1, p2):
    if np.random.rand() >= CROSS_PROB:
        return p1[:], p2[:]

    child1 = []
    child2 = []

    for x1, x2 in zip(p1, p2):
        u = np.random.rand()

        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (ETA_C + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1.0 / (ETA_C + 1))

        c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

        child1.append(c1)
        child2.append(c2)

    child1 = np.clip(child1, LOW, HIGH)
    child2 = np.clip(child2, LOW, HIGH)

    return np.array(child1), np.array(child2)


# ==========================================================
# MUTATION
# ==========================================================
def mutate(ind):
    ind = ind.copy()

    if np.random.rand() >= MUT_PROB:
        return ind
    else:
        r = np.random.rand(DIM)
        for i in range(len(ind)):
            if r[i] < 0.5:
                delta = (2 * r[i]) ** (1.0 / (ETA_M + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r[i])) ** (1.0 / (ETA_M + 1))

            ind[i] = ind[i] + delta * (HIGH - LOW)

    ind = np.clip(ind, LOW, HIGH)
    return ind


# ==========================================================
# CREATE OFFSPRING
# ==========================================================
def create_offspring(pool):
    children = []

    for i in range(0, len(pool) - 1, 2):
        p1 = pool[i]
        p2 = pool[i + 1]

        c1, c2 = crossover(p1, p2)
        children.append(mutate(c1))
        children.append(mutate(c2))

    return np.array(children[:POP_SIZE])


# ==========================================================
# NEXT GENERATION SELECTION
# ==========================================================
def next_generation(population, F, children, F_child):
    combined_population = np.vstack((population, children))
    combined_F          = np.vstack((F, F_child))

    fronts, _ = non_dominated_sort(combined_F)

    new_population = []
    new_F          = []

    for front in fronts:
        if len(new_population) + len(front) <= POP_SIZE:
            new_population.extend(combined_population[front])
            new_F.extend(combined_F[front])
        else:
            cd_front   = crowding_dist(front, combined_F)
            sorted_idx = np.argsort(-cd_front)
            remaining  = POP_SIZE - len(new_population)
            selected   = [front[i] for i in sorted_idx[:remaining]]

            new_population.extend(combined_population[selected])
            new_F.extend(combined_F[selected])
            break

    return np.array(new_population), np.array(new_F)


# ==========================================================
# NSGA-II MAIN
# ==========================================================
def run_nsga2():
    population = create_population()
    F          = np.array([evaluate(ind) for ind in population])

    initial_F = F.copy()

    for gen in range(GENS):
        pool     = tournament_select(population, F)
        children = create_offspring(pool)
        F_child  = np.array([evaluate(ind) for ind in children])

        population, F = next_generation(population, F, children, F_child)

    return initial_F, F


# ==========================================================
# PLOT RESULTS
# ==========================================================
def plot_results(initial_F, final_F):

    fronts, _ = non_dominated_sort(final_F)
    pareto    = final_F[fronts[0]]

    # True Pareto front (ZDT2)
    x = np.linspace(0, 1, 200)
    y = 1 - x ** 2

    plt.figure(figsize=(12, 5))

    # LEFT
    plt.subplot(1, 2, 1)
    plt.scatter(initial_F[:, 0], initial_F[:, 1], s=15, alpha=0.6, label='Initial population')
    plt.plot(x, y, 'r-', linewidth=2, label='True Pareto front')
    plt.title("Initial Population")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(alpha=0.3)
    plt.legend()

    # RIGHT
    plt.subplot(1, 2, 2)
    plt.scatter(pareto[:, 0], pareto[:, 1], s=20, c='green', alpha=0.7, label='NSGA-II Pareto front (non-dominated)')
    plt.plot(x, y, 'r-', linewidth=2, label='True Pareto front')
    plt.title("Obtained Non-Dominated Solutions")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(alpha=0.3)
    plt.legend()

    # MAIN TITLE
    plt.suptitle("NSGA-II Performance on ZDT2", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==========================================================
# RUN
# ==========================================================
def run_nsga2_analysis():
    num_runs = 1

    for run in range(num_runs):
        start_time = time.perf_counter()

        initial_F, final_F = run_nsga2()

        end_time = time.perf_counter()
        run_time = end_time - start_time

        plot_results(initial_F, final_F)
        print(f"Run {run + 1}: Time = {run_time:.4f} sec")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    run_nsga2_analysis()