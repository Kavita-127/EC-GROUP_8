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
GENS = 200
DIM = 30

LOW, HIGH = 0, 1

CROSS_PROB = 0.9
MUT_PROB = 0.1

ETA_C = 15
ETA_M = 20


# ==========================================================
# INITIAL POPULATION
# ==========================================================
def create_population():
    return np.random.uniform(LOW, HIGH, (POP_SIZE, DIM))


# ==========================================================
# OBJECTIVE FUNCTION (ZDT1)
# ==========================================================
def evaluate(ind):
    f1 = ind[0]
    g = 1 + (9 / (DIM - 1)) * np.sum(ind[1:])
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])


# ==========================================================
# DOMINATION CHECK
# ==========================================================
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


# ==========================================================
# FAST NON-DOMINATED SORT
# ==========================================================
def non_dominated_sort(F):
    N = len(F)
    S = [[] for _ in range(N)]
    n = [0] * N
    rank = [0] * N

    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, rank


# ==========================================================
# CROWDING DISTANCE
# ==========================================================
def crowding_dist(front, F):
    if len(front) == 0:
        return np.array([])

    front = np.array(front)
    values = F[front]
    n, m = values.shape

    dist = np.zeros(n)

    for i in range(m):
        idx = np.argsort(values[:, i])
        dist[idx[0]] = dist[idx[-1]] = np.inf

        min_v = values[idx[0], i]
        max_v = values[idx[-1], i]

        if max_v == min_v:
            continue

        for j in range(1, n - 1):
            dist[idx[j]] += (values[idx[j + 1], i] - values[idx[j - 1], i]) / (max_v - min_v)

    return dist


# ==========================================================
# SELECTION (TOURNAMENT)
# ==========================================================
def tournament_select(pop, rank, dist):
    N = len(pop)
    selected = []

    for _ in range(N):
        i, j = np.random.randint(0, N, 2)

        if rank[i] < rank[j]:
            selected.append(pop[i])
        elif rank[j] < rank[i]:
            selected.append(pop[j])
        else:
            selected.append(pop[i] if dist[i] > dist[j] else pop[j])

    return np.array(selected)


# ==========================================================
# CROSSOVER (SBX)
# ==========================================================
def crossover(p1, p2):
    if np.random.rand() > CROSS_PROB:
        return p1.copy(), p2.copy()

    c1, c2 = [], []

    for x, y in zip(p1, p2):
        u = np.random.rand()

        if u <= 0.5:
            beta = (2 * u) ** (1 / (ETA_C + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (ETA_C + 1))

        c1.append(0.5 * ((1 + beta) * x + (1 - beta) * y))
        c2.append(0.5 * ((1 - beta) * x + (1 + beta) * y))

    return np.clip(c1, LOW, HIGH), np.clip(c2, LOW, HIGH)


# ==========================================================
# MUTATION
# ==========================================================
def mutate(ind):
    ind = ind.copy()

    if np.random.rand() > MUT_PROB:
        return ind

    for i in range(DIM):
        r = np.random.rand()

        if r < 0.5:
            delta = (2 * r) ** (1 / (ETA_M + 1)) - 1
        else:
            delta = 1 - (2 * (1 - r)) ** (1 / (ETA_M + 1))

        ind[i] += delta

    return np.clip(ind, LOW, HIGH)


# ==========================================================
# CREATE OFFSPRING
# ==========================================================
def create_offspring(pool):
    children = []

    for i in range(0, len(pool) - 1, 2):
        c1, c2 = crossover(pool[i], pool[i + 1])
        children.append(mutate(c1))
        children.append(mutate(c2))

    return np.array(children[:POP_SIZE])


# ==========================================================
# NEXT GENERATION SELECTION
# ==========================================================
def next_generation(pop, F, children, F_child):
    combined = np.vstack((pop, children))
    combined_F = np.vstack((F, F_child))

    fronts, _ = non_dominated_sort(combined_F)

    new_pop, new_F = [], []

    for front in fronts:
        if len(new_pop) + len(front) <= POP_SIZE:
            new_pop.extend(combined[front])
            new_F.extend(combined_F[front])
        else:
            dist = crowding_dist(front, combined_F)
            order = np.argsort(-dist)
            remain = POP_SIZE - len(new_pop)

            selected = [front[i] for i in order[:remain]]

            new_pop.extend(combined[selected])
            new_F.extend(combined_F[selected])
            break

    return np.array(new_pop), np.array(new_F)


# ==========================================================
# NSGA-II MAIN
# ==========================================================
def run_nsga2():
    pop = create_population()
    F = np.array([evaluate(ind) for ind in pop])

    initial_F = F.copy()

    for _ in range(GENS):
        fronts, rank = non_dominated_sort(F)

        dist = np.zeros(len(pop))
        for front in fronts:
            dist[front] = crowding_dist(front, F)

        pool = tournament_select(pop, rank, dist)
        children = create_offspring(pool)
        F_child = np.array([evaluate(ind) for ind in children])

        pop, F = next_generation(pop, F, children, F_child)

    return initial_F, F


# ==========================================================
# PLOT RESULTS
# ==========================================================
def plot_results(initial_F, final_F):

    fronts, _ = non_dominated_sort(final_F)
    pareto = final_F[fronts[0]]

    x = np.linspace(0, 1, 200)
    y = 1 - np.sqrt(x)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 5))

    # LEFT — Initial Population
    plt.subplot(1, 2, 1)
    plt.scatter(initial_F[:, 0], initial_F[:, 1],
                color="#4C72B0", alpha=0.6,
                edgecolors='black',
                label="Initial Population")
    plt.plot(x, y, '--',
             color="#DD8452",
             linewidth=2,
             label="True Pareto Front")

    plt.title("Initial Population")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend()


    # RIGHT — Final Pareto Front
    plt.subplot(1, 2, 2)
    plt.scatter(pareto[:, 0], pareto[:, 1],
                color="#55A868", alpha=0.8,
                edgecolors='black',
                label="NSGA-II Pareto Front")

    plt.plot(x, y, '--',
             color="#C44E52",
             linewidth=2,
             label="True Pareto Front")

    plt.title("Final Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend()


    plt.suptitle("NSGA-II on ZDT1", fontsize=14)
    plt.tight_layout()
    plt.show()

# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    start = time.perf_counter()

    init_F, final_F = run_nsga2()

    end = time.perf_counter()
    print(f"Execution Time: {end - start:.4f} sec")

    plot_results(init_F, final_F)