# BCGA Repair
import numpy as np
import random
import time
import matplotlib.pyplot as plt


# -------------------------------
# SETTINGS
# -------------------------------
SEED           = 42
POP_SIZE       = 100
GENERATIONS    = 800
CROSSOVER_RATE = 0.8
MUTATION_RATE  = 0.15
NUM_RUNS       = 20


# -------------------------------
# READ FILE
# -------------------------------
def read_gap_file(filepath):
    with open(filepath, 'r') as f:
        data = list(map(int, f.read().split()))
    idx = 0; P = data[idx]; idx += 1
    problems = []
    for _ in range(P):
        m, n = data[idx], data[idx + 1]; idx += 2
        cost     = np.array(data[idx:idx + m * n]).reshape(m, n); idx += m * n
        resource = np.array(data[idx:idx + m * n]).reshape(m, n); idx += m * n
        capacity = np.array(data[idx:idx + m]); idx += m
        problems.append((m, n, cost, resource, capacity))
    return problems


# ---------------------------------------------------------------
# FEASIBILITY CHECK
# ---------------------------------------------------------------
def is_feasible(assignment, m, n, resource, capacity):
    used = np.zeros(m)
    for j in range(n):
        used[assignment[j]] += resource[assignment[j]][j]
    violations = [i for i in range(m) if used[i] > capacity[i]]
    return len(violations) == 0, used, violations


# ---------------------------------------------------------------
# FEASIBLE INITIAL INDIVIDUAL
# ---------------------------------------------------------------
def create_feasible_individual(m, n, cost, resource, capacity):
    used      = np.zeros(m)
    assign    = [0] * n
    job_order = list(range(n))
    random.shuffle(job_order)
    for j in job_order:
        feasible = [i for i in range(m)
                    if used[i] + resource[i][j] <= capacity[i]]
        if feasible:
            i = max(feasible,
                    key=lambda i: cost[i][j] + random.uniform(0, 1e-3))
        else:
            i = min(range(m), key=lambda i: used[i] / (capacity[i] + 1e-9))
        assign[j] = i
        used[i]  += resource[i][j]
    return assign


def repair_individual(individual, m, n, cost, resource, capacity):
    used = np.zeros(m)
    for j in range(n):
        used[individual[j]] += resource[individual[j]][j]
    for _ in range(2000):
        overloaded = [i for i in range(m) if used[i] > capacity[i]]
        if not overloaded: break
        i    = random.choice(overloaded)
        jobs = [j for j in range(n) if individual[j] == i]
        if not jobs: continue
        jobs.sort(key=lambda j: cost[i][j])  # move lowest-profit jobs first
        moved = False
        for j in jobs:
            for k in range(m):
                if k != i and used[k] + resource[k][j] <= capacity[k]:
                    used[i] -= resource[i][j]; individual[j] = k
                    used[k] += resource[k][j]; moved = True; break
            if moved: break
        if not moved:
            j = random.choice(jobs); k = random.randint(0, m - 1)
            used[i] -= resource[i][j]; individual[j] = k
            used[k] += resource[k][j]
    return individual


# ---------------------------------------------------------------
# FITNESS
# ---------------------------------------------------------------
def fitness_raw(individual, m, n, cost):
    return sum(cost[individual[j]][j] for j in range(n))


def fitness_true(individual, m, n, cost, resource, capacity):
    total = 0; used = np.zeros(m)
    for j in range(n):
        i = individual[j]; total += cost[i][j]; used[i] += resource[i][j]
    if np.any(used > capacity): return -float('inf')
    return total


# ---------------------------------------------------------------
# GA OPERATORS
# ---------------------------------------------------------------
def tournament(population, fit_func):
    sel = random.sample(population, 3)
    return max(sel, key=fit_func)


def crossover(p1, p2, n):
    if random.random() < CROSSOVER_RATE:
        pt = random.randint(1, n - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]


def mutate_and_repair(individual, m, n, cost, resource, capacity):
    ind = individual[:]
    for j in range(n):
        if random.random() < MUTATION_RATE:
            ind[j] = random.randint(0, m - 1)
    return repair_individual(ind, m, n, cost, resource, capacity)


# ---------------------------------------------------------------
# RUN BCGA — REPAIR METHOD
# ---------------------------------------------------------------
def run_bcga_repair(m, n, cost, resource, capacity):

    sel_fit = lambda x: fitness_raw(x, m, n, cost)

    # Requirement 7: all initial individuals are feasible
    population = [create_feasible_individual(m, n, cost, resource, capacity)
                  for _ in range(POP_SIZE)]

    true0    = [fitness_true(p, m, n, cost, resource, capacity) for p in population]
    bi       = int(np.argmax(true0))
    best_fit = true0[bi]; best_sol = population[bi][:]

    history = []

    for gen in range(GENERATIONS):
        # Apply repair to entire population each generation
        population = [repair_individual(ind[:], m, n, cost, resource, capacity)
                      for ind in population]

        new_pop = [best_sol[:]]   # elitism

        while len(new_pop) < POP_SIZE:
            p1 = tournament(population, sel_fit)
            p2 = tournament(population, sel_fit)
            c1, c2 = crossover(p1, p2, n)
            c1 = mutate_and_repair(c1, m, n, cost, resource, capacity)
            c2 = mutate_and_repair(c2, m, n, cost, resource, capacity)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE: new_pop.append(c2)

        population = new_pop[:POP_SIZE]

        # Requirement 8: best tracked via true fitness (penalty=0)
        for ind in population:
            ft = fitness_true(ind, m, n, cost, resource, capacity)
            if ft > best_fit: best_fit = ft; best_sol = ind[:]

        history.append(best_fit)

    return best_sol, best_fit, history


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":

    filepath = "gap12.txt"
    SEP  = "=" * 62
    LINE = "-" * 62

    problems = read_gap_file(filepath)
    m, n, cost, resource, capacity = problems[0]

    print(SEP)
    print("  Generalized Assignment Problem — BCGA [REPAIR]")
    print(SEP)
    print(f"  File               : {filepath}")
    print(f"  Problem sets       : {len(problems)}")
    print(f"  Agents             : {m}   Jobs : {n}")
    print(f"  Runs               : {NUM_RUNS}")
    print(f"  Generations        : {GENERATIONS}")
    print(f"  Population size    : {POP_SIZE}")
    print(f"  Crossover rate     : {CROSSOVER_RATE}")
    print(f"  Mutation rate      : {MUTATION_RATE}")
    print(f"  Constraint method  : repair (greedy, moves lowest-profit jobs first)")
    print(LINE)

    all_histories = []; best_values = []; run_times = []
    best_overall  = -float('inf'); best_solution = None

    total_start = time.time()

    for run in range(NUM_RUNS):
        random.seed(SEED + run); np.random.seed(SEED + run)
        t0 = time.time()
        sol, val, history = run_bcga_repair(m, n, cost, resource, capacity)
        rt = time.time() - t0
        all_histories.append(history); best_values.append(val); run_times.append(rt)
        print(f"  Run {run+1:>2}  |  Best Fitness : {val:>8}  |  Time : {rt:.2f}s")
        if val > best_overall: best_overall = val; best_solution = sol

    total_time = time.time() - total_start

    print(LINE)
    print("  Summary")
    print(f"    Best Fitness       : {best_overall}")
    print(f"    Average Fitness    : {np.mean(best_values):.2f}")
    print(f"    Worst Fitness      : {np.min(best_values)}")
    print(f"    Std Deviation      : {np.std(best_values):.2f}")
    print(LINE)
    print("  Runtime Summary")
    print(f"    Total time ({NUM_RUNS} runs) : {total_time:.2f}s")
    print(f"    Avg time per run     : {np.mean(run_times):.2f}s")
    print(f"    Min run time         : {np.min(run_times):.2f}s")
    print(f"    Max run time         : {np.max(run_times):.2f}s")
    print(LINE)

    feasible, used, violations = is_feasible(best_solution, m, n, resource, capacity)
    print("  Feasibility Check on Best Solution")
    print(f"    Feasible : {'YES - All constraints satisfied' if feasible else 'NO - Constraints violated'}")
    if not feasible:
        for v in violations:
            print(f"    Agent {v+1:>2} : Used={used[v]:.1f}  Capacity={capacity[v]}  "
                  f"Exceeded by {used[v]-capacity[v]:.1f}")
    else:
        print(f"    All {m} agent capacity constraints satisfied.")
        print(f"    Penalty in best solution : 0  (pure profit reported)")
    print(LINE)

    print("  Best Assignment (job -> agent):")
    for j, agent in enumerate(best_solution):
        print(f"    Job {j+1:>3}  ->  Agent {agent+1}")
    print(SEP)

    avg_history = np.mean(all_histories, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(avg_history, color='green', linewidth=2)
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Avg Best Feasible Fitness")
    axes[0].set_title(f"BCGA [REPAIR] Convergence\n(avg over {NUM_RUNS} runs, {filepath})")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(range(1, NUM_RUNS+1), run_times,
                 color='green', linewidth=2, marker='o', markersize=5)
    axes[1].axhline(np.mean(run_times), color='crimson', linestyle='--',
                    linewidth=1.5, label=f"Avg: {np.mean(run_times):.2f}s")
    axes[1].set_xlabel("Run Number"); axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title(f"BCGA [REPAIR] Runtime per Run\n({GENERATIONS} generations, pop={POP_SIZE})")
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(); plt.show()