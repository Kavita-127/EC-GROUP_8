
# gap_base.py
# Shared utilities used by all algorithm files

import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# SHARED PARAMETERS
# ---------------------------------------------------------------
SEED           = 42
POP_SIZE       = 100
ITERATIONS     = 800
CROSSOVER_RATE = 0.8
MUTATION_RATE  = 0.15
NUM_RUNS       = 20
ETA_C          = 15     # SBX distribution index  (RCGA only)
ETA_M          = 20     # poly-mutation index      (RCGA only)


# ---------------------------------------------------------------
# READ FILE
# ---------------------------------------------------------------
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
    """
    Parameters
    ----------
    assignment : list of int  — assignment[j] = agent index for job j
    m, n       : int          — number of agents, number of jobs
    resource   : 2D array     — resource[i][j] used if agent i takes job j
    capacity   : 1D array     — capacity[i] = max resource for agent i

    Returns
    -------
    feasible   : bool   — True if no capacity is exceeded
    used       : array  — resource consumed per agent
    violations : list   — agent indices that exceed capacity
    """
    used = np.zeros(m)
    for j in range(n):
        used[assignment[j]] += resource[assignment[j]][j]
    violations = [i for i in range(m) if used[i] > capacity[i]]
    return len(violations) == 0, used, violations


def is_feasible_bool(assignment, m, n, resource, capacity):
    # Quick bool-only feasibility check
    used = np.zeros(m)
    for j in range(n):
        used[assignment[j]] += resource[assignment[j]][j]
    return np.all(used <= capacity)


# ---------------------------------------------------------------
# COMPUTE AUTO-SCALED PENALTY
# ---------------------------------------------------------------
def compute_penalty(m, n, cost):
    return int(sum(cost[:, j].max() for j in range(n))) + 1


# ---------------------------------------------------------------
# FITNESS FUNCTIONS
# ---------------------------------------------------------------
def fitness_true(assignment, m, n, cost, resource, capacity):
    total = 0; used = np.zeros(m)
    for j in range(n):
        i = assignment[j]; total += cost[i][j]; used[i] += resource[i][j]
    if np.any(used > capacity):
        return -float('inf')
    return total


def fitness_guided_penalty(assignment, m, n, cost, resource, capacity, penalty):
    total = 0; used = np.zeros(m)
    for j in range(n):
        i = assignment[j]; total += cost[i][j]; used[i] += resource[i][j]
    pen = sum(penalty * (used[i] - capacity[i])
              for i in range(m) if used[i] > capacity[i])
    return total - pen


def fitness_raw(assignment, m, n, cost):
    return sum(cost[assignment[j]][j] for j in range(n))


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
            # fallback: least-loaded agent
            i = min(range(m), key=lambda i: used[i] / (capacity[i] + 1e-9))
        assign[j] = i
        used[i]  += resource[i][j]
    # Always repair to fix any capacity violations from fallback assignments
    assign = repair_individual(assign, m, n, cost, resource, capacity)
    return assign


# ---------------------------------------------------------------
# REPAIR
# ---------------------------------------------------------------
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
        jobs.sort(key=lambda j: cost[i][j])  # cheapest job moved first
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
# OUTPUT HELPERS
# ---------------------------------------------------------------
SEP  = "=" * 64
LINE = "-" * 64


def print_header(algo_name, method, filepath, problems,
                 m, n, extra_params=""):
    print(SEP)
    print(f"  Generalized Assignment Problem — {algo_name} [{method.upper()}]")
    print(SEP)
    print(f"  File               : {filepath}")
    print(f"  Problem sets       : {len(problems)}")
    print(f"  Agents             : {m}   Jobs : {n}")
    print(f"  Runs               : {NUM_RUNS}")
    print(f"  Iterations/Gens    : {ITERATIONS}")
    print(f"  Population size    : {POP_SIZE}")
    print(f"  Crossover rate     : {CROSSOVER_RATE}")
    print(f"  Mutation rate      : {MUTATION_RATE}")
    print(f"  Constraint method  : {method}")
    if extra_params:
        print(f"  {extra_params}")
    print(LINE)


def print_summary(best_overall, best_values, run_times, total_time):
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


def print_feasibility(best_solution, m, n, resource, capacity):
    feasible, used, violations = is_feasible(
        best_solution, m, n, resource, capacity)
    print("  Feasibility Check on Best Solution")
    print(f"    Feasible : "
          f"{'YES - All constraints satisfied' if feasible else 'NO - Constraints violated'}")
    if not feasible:
        for v in violations:
            print(f"    Agent {v+1:>2} : Used={used[v]:.1f}  "
                  f"Capacity={capacity[v]}  "
                  f"Exceeded by {used[v]-capacity[v]:.1f}")
    else:
        print(f"    All {m} agent capacity constraints satisfied.")
        print(f"    Penalty in best solution : 0  (pure profit reported)")
    print(LINE)
    return feasible


def print_assignment(best_solution):
    print("  Best Assignment (job -> agent):")
    for j, agent in enumerate(best_solution):
        print(f"    Job {j+1:>3}  ->  Agent {agent+1}")
    print(SEP)


def plot_results(avg_history, run_times, algo_name, method,
                 filepath, color):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(avg_history, color=color, linewidth=2)
    axes[0].set_xlabel("Iteration / Generation")
    axes[0].set_ylabel("Avg Best Feasible Fitness (penalty=0)")
    axes[0].set_title(
        f"{algo_name} [{method.upper()}] Convergence\n"
        f"(avg over {NUM_RUNS} runs, {filepath})")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].plot(range(1, NUM_RUNS + 1), run_times,
                 color=color, linewidth=2, marker='o', markersize=5)
    axes[1].axhline(np.mean(run_times), color='crimson', linestyle='--',
                    linewidth=1.5,
                    label=f"Avg: {np.mean(run_times):.2f}s")
    axes[1].set_xlabel("Run Number")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title(
        f"{algo_name} [{method.upper()}] Runtime per Run\n"
        f"({ITERATIONS} iterations, pop={POP_SIZE})")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()