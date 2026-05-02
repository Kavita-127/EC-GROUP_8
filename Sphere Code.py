# Metaheuristics Comparison on Sphere Function 

import numpy as np
import random
import matplotlib.pyplot as plt

DIM = 10
LB, UB = -20, 20
POP_SIZE = 30
GEN = 100
RUNS = 20

# Sphere function
def sphere(x):
    return np.sum(x**2)

# ------------------ BCGA ------------------
def bcga():
    BITS = 10
    CHROM_LEN = DIM * BITS
    Pc = 0.8
    Pm = 0.02

    pop = np.random.randint(0, 2, (POP_SIZE, CHROM_LEN))

    def decode(ind):
        real = []
        for i in range(DIM):
            bits = ind[i*BITS:(i+1)*BITS]
            decimal = int("".join(map(str, bits)), 2)
            value = LB + (decimal / (2**BITS - 1)) * (UB - LB)
            real.append(value)
        return np.array(real)

    def tournament(pop, fitness):
        i, j = np.random.choice(len(pop), 2)
        return pop[i] if fitness[i] < fitness[j] else pop[j]

    best_list = []

    for g in range(GEN):
        decoded = np.array([decode(ind) for ind in pop])
        fitness = np.array([sphere(ind) for ind in decoded])
        best_list.append(np.min(fitness))

        new_pop = []
        new_pop.append(pop[np.argmin(fitness)].copy())  # elitism

        while len(new_pop) < POP_SIZE:
            p1 = tournament(pop, fitness)
            p2 = tournament(pop, fitness)

            if random.random() < Pc:
                point = random.randint(1, CHROM_LEN - 1)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in [c1, c2]:
                for i in range(CHROM_LEN):
                    if random.random() < Pm:
                        c[i] = 1 - c[i]

                new_pop.append(c)
                if len(new_pop) >= POP_SIZE:
                    break

        pop = np.array(new_pop)

    return best_list

# ------------------ RCGA ------------------
def rcga():
    Pc, Pm = 0.9, 0.05
    pop = np.random.uniform(LB, UB, (POP_SIZE, DIM))
    best_list = []

    def tournament(pop, fitness):
        i, j = np.random.choice(len(pop), 2)
        return pop[i] if fitness[i] < fitness[j] else pop[j]

    for g in range(GEN):
        fitness = np.array([sphere(ind) for ind in pop])
        best_list.append(np.min(fitness))

        new_pop = [pop[np.argmin(fitness)].copy()]

        while len(new_pop) < POP_SIZE:
            p1 = tournament(pop, fitness)
            p2 = tournament(pop, fitness)

            if random.random() < Pc:
                alpha = random.random()
                c1 = alpha*p1 + (1-alpha)*p2
                c2 = alpha*p2 + (1-alpha)*p1
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in [c1, c2]:
                for i in range(DIM):
                    if random.random() < Pm:
                        c[i] += np.random.normal(0, 1)
                c = np.clip(c, LB, UB)
                new_pop.append(c)
                if len(new_pop) >= POP_SIZE:
                    break

        pop = np.array(new_pop)

    return best_list

# ------------------ PSO ------------------
def pso():
    w, c1, c2 = 0.7, 1.5, 1.5

    pos = np.random.uniform(LB, UB, (POP_SIZE, DIM))
    vel = np.random.uniform(-1, 1, (POP_SIZE, DIM))

    pbest = pos.copy()
    pbest_val = np.array([sphere(x) for x in pos])
    gbest = pbest[np.argmin(pbest_val)]

    best_list = []

    for g in range(GEN):
        for i in range(POP_SIZE):
            r1, r2 = random.random(), random.random()
            vel[i] = (w*vel[i] +
                      c1*r1*(pbest[i] - pos[i]) +
                      c2*r2*(gbest - pos[i]))

            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], LB, UB)

            fit = sphere(pos[i])
            if fit < pbest_val[i]:
                pbest[i] = pos[i]
                pbest_val[i] = fit

        gbest = pbest[np.argmin(pbest_val)]
        best_list.append(sphere(gbest))

    return best_list

# ------------------ DE ------------------
def de():
    F, CR = 0.8, 0.9
    pop = np.random.uniform(LB, UB, (POP_SIZE, DIM))
    best_list = []

    for g in range(GEN):
        new_pop = []

        for i in range(POP_SIZE):
            idxs = list(range(POP_SIZE))
            idxs.remove(i)
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = np.clip(a + F*(b - c), LB, UB)
            trial = pop[i].copy()
            j_rand = random.randint(0, DIM-1)

            for j in range(DIM):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            new_pop.append(trial if sphere(trial) < sphere(pop[i]) else pop[i])

        pop = np.array(new_pop)
        best_list.append(np.min([sphere(x) for x in pop]))

    return best_list

# ------------------ TLBO ------------------
def tlbo():
    pop = np.random.uniform(LB, UB, (POP_SIZE, DIM))
    best_list = []

    for g in range(GEN):
        fitness = np.array([sphere(x) for x in pop])
        teacher = pop[np.argmin(fitness)]
        mean = np.mean(pop, axis=0)

        TF = random.choice([1, 2])
        for i in range(POP_SIZE):
            r = np.random.random(DIM)           # per-dimension random vector
            new = np.clip(pop[i] + r*(teacher - TF*mean), LB, UB)
            if sphere(new) < sphere(pop[i]):
                pop[i] = new

        for i in range(POP_SIZE):
            j = random.randint(0, POP_SIZE-1)
            while j == i:
                j = random.randint(0, POP_SIZE-1)

            r = np.random.random(DIM)           # per-dimension random vector
            if sphere(pop[i]) < sphere(pop[j]):
                new = pop[i] + r*(pop[i] - pop[j])
            else:
                new = pop[i] + r*(pop[j] - pop[i])

            new = np.clip(new, LB, UB)
            if sphere(new) < sphere(pop[i]):
                pop[i] = new

        best_list.append(np.min([sphere(x) for x in pop]))

    return best_list

# ------------------ ABC ------------------
def abc():
    limit = 10
    pop = np.random.uniform(LB, UB, (POP_SIZE, DIM))
    trial = np.zeros(POP_SIZE)
    best_list = []

    for g in range(GEN):
        for i in range(POP_SIZE):
            k = random.randint(0, POP_SIZE-1)
            while k == i:
                k = random.randint(0, POP_SIZE-1)

            new = np.clip(pop[i] + random.uniform(-1,1)*(pop[i] - pop[k]), LB, UB)

            if sphere(new) < sphere(pop[i]):
                pop[i], trial[i] = new, 0
            else:
                trial[i] += 1

        fitness = np.array([1/(1+sphere(x)) for x in pop])
        prob = fitness / np.sum(fitness)

        for i in range(POP_SIZE):
            if random.random() < prob[i]:
                k = random.randint(0, POP_SIZE-1)
                new = np.clip(pop[i] + random.uniform(-1,1)*(pop[i] - pop[k]), LB, UB)

                if sphere(new) < sphere(pop[i]):
                    pop[i], trial[i] = new, 0
                else:
                    trial[i] += 1

        for i in range(POP_SIZE):
            if trial[i] > limit:
                pop[i] = np.random.uniform(LB, UB, DIM)
                trial[i] = 0

        best_list.append(np.min([sphere(x) for x in pop]))

    return best_list

# ------------------ RUN ALL ------------------
algorithms = {
    "BCGA": bcga,
    "RCGA": rcga,
    "PSO": pso,
    "DE": de,
    "TLBO": tlbo,
    "ABC": abc
}

results = {}
final_stats = {}

# ---- ADDED: also store all 20 run curves per algorithm ----
all_runs = {}
# -----------------------------------------------------------

for name, algo in algorithms.items():
    avg_curve = np.zeros(GEN)
    final_vals = []

    # ---- ADDED: collect individual run curves ----
    run_curves = []
    # ----------------------------------------------

    for _ in range(RUNS):
        curve = np.array(algo())
        avg_curve += curve
        final_vals.append(curve[-1])

        # ---- ADDED ----
        run_curves.append(curve)
        # ---------------

    avg_curve /= RUNS
    results[name] = avg_curve

    # ---- ADDED ----
    all_runs[name] = run_curves
    # ---------------

    final_stats[name] = (np.min(final_vals), np.mean(final_vals))

# ------------------ PRINT RESULTS ------------------
print("\n===== FINAL PERFORMANCE COMPARISON =====\n")
for name, (best_val, avg_val) in final_stats.items():
    print(f"{name}: Best = {best_val:.6f}, Average = {avg_val:.6f}")

best_algo = min(final_stats, key=lambda x: final_stats[x][1])
print(f"\n>>> Best Performing Algorithm: {best_algo}")

# ------------------ PLOT 1: Separate graph for each algorithm (20 runs) ------------------
colors_20 = plt.cm.tab20.colors

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, name in enumerate(algorithms):
    ax = axes[idx]
    for r in range(RUNS):
        ax.semilogy(range(1, GEN+1), np.maximum(all_runs[name][r], 1e-12),
                    color=colors_20[r], alpha=0.55, linewidth=0.9)
    ax.semilogy(range(1, GEN+1), np.maximum(results[name], 1e-12),
                color='black', linewidth=2.0, linestyle='--', label='Average')
    ax.set_title(f"{name} – 20 Runs", fontsize=11, fontweight='bold')
    ax.set_xlabel("Generations", fontsize=9)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

plt.suptitle("Convergence – 20 Individual Runs per Algorithm (Sphere Function, Dim=10)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("convergence_separate_20runs.png", dpi=150, bbox_inches='tight')
plt.show()

# ------------------ PLOT 2: Comparison graph of all algorithms (average curve) ------------------
colors_main = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']

plt.figure(figsize=(10, 6))
for (name, curve), col in zip(results.items(), colors_main):
    plt.semilogy(range(1, GEN+1), np.maximum(curve, 1e-12),
                 label=name, color=col, linewidth=1.8)

plt.xlabel("Generations", fontsize=12)
plt.ylabel("Average Best Fitness (log scale)", fontsize=12)
plt.title("Convergence Comparison – All Algorithms\n(Sphere Function, Dim=10, Avg of 20 Runs)",
          fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("convergence_comparison_all.png", dpi=150, bbox_inches='tight')
plt.show()