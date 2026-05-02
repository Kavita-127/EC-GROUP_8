"""
========================================================
Blockchain Configuration Optimization (BCO)
Metaheuristic Algorithms 
========================================================
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL PARAMETERS
# ============================================================
POP_SIZE = 150       # Population / swarm size (100–200 range)
MAX_GEN  = 300       # Maximum generations
RUNS     = 20        # Independent runs
CR       = 0.8       # Crossover rate
MUT_RATE = 0.1       # Mutation rate

CONVERGENCE_TOL  = 1e-6   # Tolerance to detect convergence
CONVERGENCE_ITERS = 20    # Stop if no improvement for this many generations

# System Parameters
q   = 4
theta = 1
O   = 0.5        # Mb  – verification feedback size
rd  = 1.2        # Mb/s – downlink rate
ru  = 1.3        # Mb/s – uplink rate
K   = 100        # computational resources for block verification
B   = 0.5/1024   # Mb  – block size (0.5 kb converted to Mb)
psi = 0.001

# Weighting coefficients (alpha + beta + gamma = 1)
ALPHA = 1/3
BETA  = 1/3
GAMMA = 1/3

# ============================================================
# SETTINGS 
# ============================================================

# Setting 1: M = 25 validators
S1 = {
    'name' : 'Setting 1 (M=25)',
    'v'    : 5,
    'M'    : 25,
    't'    : 50,
    'X'    : 500,
    'xi'   : np.array([250.74, 187.36, 138.91, 245.20, 271.70,
                       276.80, 119.32, 194.45, 213.02, 138.59,
                       207.21, 215.94, 236.32, 203.08, 145.41,
                       264.83, 286.12, 123.30, 242.89, 121.47,
                       198.76, 254.19, 167.43, 289.05, 132.88])
}

# Setting 2: M = 40 validators
S2 = {
    'name' : 'Setting 2 (M=40)',
    'v'    : 5,
    'M'    : 40,
    't'    : 50,
    'X'    : 500,
    'xi'   : np.array([121.12, 259.89, 198.40, 234.00, 272.30,
                       134.32, 157.55, 288.65, 210.58, 255.98,
                       249.03, 201.28, 298.93, 157.16, 125.26,
                       167.22, 261.86, 211.60, 184.22, 153.01,
                       116.73, 241.46, 201.56, 290.00, 218.31,
                       232.87, 104.61, 268.79, 185.63, 115.13,
                       236.16, 102.55, 259.15, 226.24, 100.15,
                       142.94, 199.64, 221.28, 150.30, 104.61])
}

# Setting 3: M = 60 validators (generated with fixed seed for reproducibility)
rng_s3 = np.random.default_rng(seed=0)
S3 = {
    'name' : 'Setting 3 (M=60)',
    'v'    : 5,
    'M'    : 60,
    't'    : 50,
    'X'    : 500,
    'xi'   : np.round(rng_s3.uniform(100, 300, 60), 2)
}

SETTINGS = [S1, S2, S3]

# Transaction values
TRANSACTION_VALUES = [20, 40, 60, 80, 100]

# ============================================================
# BCO OBJECTIVE FUNCTIONS
# ============================================================

def compute_min_max(s, n_fixed=None):
    """
    Pre-compute Cmin, Cmax, eta_min, eta_max, Lmin, Lmax for a setting.
    n_fixed: if provided, use this as the fixed n value;
             otherwise falls back to s['t'] (default = 50).
    """
    xi   = s['xi']
    v, M = s['v'], s['M']
    t, X = s['t'], s['X']

    # Use fixed n for bounds if provided, else use t and X for range
    # Bounds are still computed over the full feasible range [t, X]
    # so normalisation is consistent regardless of chosen n_fixed.

    # --- Cost bounds ---
    Cmax = np.sum(xi) / t
    Cmin = np.sum(np.sort(xi)[:v]) / X

    # --- Security bounds ---
    eta_max = theta * (M ** q)
    eta_min = theta * (v ** q)

    # --- Latency bounds ---
    Lmax = (X * B / rd) + (K / np.min(xi)) + psi * (X * B) ** M + (O / ru)
    v_th_highest = np.sort(xi)[-v]
    Lmin = (t * B / rd) + (K / v_th_highest) + psi * (t * B) ** v + (O / ru)

    return dict(Cmin=Cmin, Cmax=Cmax,
                eta_min=eta_min, eta_max=eta_max,
                Lmin=Lmin, Lmax=Lmax)


def log_minmax_norm(val, vmin, vmax):
    """Log min-max normalisation """
    if vmax == vmin:
        return 0.0
    return (np.log(max(val, 1e-12)) - np.log(max(vmin, 1e-12))) / \
           (np.log(max(vmax, 1e-12)) - np.log(max(vmin, 1e-12)))


def compute_utility(m_sel, n, s, bounds):
    """
    Compute BCO utility U = alpha*L_norm + beta*eta_norm + gamma*C_norm.
    m_sel : sorted array of SELECTED validator indices (0-based)
    n     : fixed number of transactions per block
    s     : setting dict
    bounds: pre-computed min/max bounds
    Returns (U, L, eta, C) or None if infeasible.
    """
    xi   = s['xi']
    v, M = s['v'], s['M']
    t, X = s['t'], s['X']
    m = len(m_sel)

    # ---- Constraint check ----
    if not (v <= m <= M):
        return None
    if not (t <= n <= X):
        return None

    # ---- Cost C = sum(xi[selected]) / n ----
    C = np.sum(xi[m_sel]) / n

    # ---- Security eta = theta * m^q ----
    eta = theta * (m ** q)

    # ---- Latency L ----
    L = (n * B / rd) + np.max(K / xi[m_sel]) + psi * (n * B) ** m + (O / ru)

    # ---- Log min-max normalisation ----
    L_norm   = log_minmax_norm(L,   bounds['Lmin'],   bounds['Lmax'])
    C_norm   = log_minmax_norm(C,   bounds['Cmin'],   bounds['Cmax'])
    eta_norm = 1.0 - log_minmax_norm(eta, bounds['eta_min'], bounds['eta_max'])

    U = ALPHA * L_norm + BETA * eta_norm + GAMMA * C_norm
    return U, L, eta, C


# ============================================================
# SOLUTION ENCODING HELPERS
# n is FIXED — chromosome length = M (validators only)
# ============================================================

def decode_solution(chromosome, s):
    """
    Decode a real-valued chromosome of length M.
    Values > 0.5 → validator selected.
    n is fixed at s['t'] = 50 (set externally via global N_FIXED).
    """
    M = s['M']
    indices = np.where(chromosome[:M] > 0.5)[0]
    m = len(indices)
    return indices, m


def random_solution(s):
    """
    Random chromosome of length M (validator genes only).
    n_gene removed — n is fixed externally.
    """
    M = s['M']
    chrom = np.random.rand(M)       # <-- length M, not M+1
    return chrom


def repair_solution(chrom, s):
    """
    Repair chromosome so that selected validators count is in [v, M].
    """
    M     = s['M']
    v_min = s['v']
    idx   = np.where(chrom[:M] > 0.5)[0]
    m     = len(idx)

    if m < v_min:
        unsel = np.where(chrom[:M] <= 0.5)[0]
        need  = v_min - m
        if len(unsel) >= need:
            flip = np.random.choice(unsel, need, replace=False)
            chrom[flip] = 0.6
    elif m > M:
        need = m - M
        flip = np.random.choice(idx, need, replace=False)
        chrom[flip] = 0.4

    return chrom


# Global fixed n — set before each algorithm run
N_FIXED = 50

def evaluate(chrom, s, bounds):
    """
    Full evaluation of a chromosome.
    n is taken from global N_FIXED (default 50).
    Returns (fitness, U, L, eta, C, m_sel, n).
    """
    global N_FIXED
    n = N_FIXED

    indices, m = decode_solution(chrom, s)
    if m == 0:
        return np.inf, np.inf, 0, 0, 0, [], n

    result = compute_utility(indices, n, s, bounds)
    if result is None:
        return np.inf, np.inf, 0, 0, 0, indices, n

    U, L, eta, C = result
    return U, U, L, eta, C, indices, n


def convergence_check(history, tol=CONVERGENCE_TOL, window=CONVERGENCE_ITERS):
    if len(history) < window:
        return False
    recent = history[-window:]
    return (recent[0] - recent[-1]) < tol


# ============================================================
# PRINT HELPERS
# ============================================================

def print_run1_result(name, sname, U, m, n, idx, xi):
    print(f"\n  Best solution found (Run 1) – {name} | {sname} | n={n}:")
    print(f"    Utility U         = {U:.6f}")
    print(f"    Validators m      = {m}")
    print(f"    Transactions n    = {n}")
    sel_xi = xi[idx]
    print(f"    Selected xi vals  = {np.round(sel_xi, 2).tolist()}")
    print(f"    Selected indices  = {[int(i) for i in idx]}")


def print_stats(name, sname, utilities, times, stopped_gens, n=None):
    n_str = f" | n={n}" if n is not None else ""
    print(f"\n  {name} | {sname}{n_str} – Summary over {RUNS} runs:")
    print(f"    Best  Utility = {np.min(utilities):.6f}")
    print(f"    Avg   Utility = {np.mean(utilities):.6f}")
    print(f"    Worst Utility = {np.max(utilities):.6f}")
    print(f"    Std   Utility = {np.std(utilities):.6f}")
    print(f"    Avg Time      = {np.mean(times)*1000:.2f} ms")
    print(f"    Avg Conv Gen  = {np.mean(stopped_gens):.1f}")


def plot_convergence(all_histories, name, sname, color_ind, color_avg,
                     phase_switch=None, n=None):
    avg_curve = np.mean([h for h in all_histories], axis=0)
    n_str = f" | n={n}" if n is not None else ""
    plt.figure(figsize=(8, 4))
    for h in all_histories:
        plt.plot(h, alpha=0.25, color=color_ind, linewidth=0.8)
    plt.plot(avg_curve, linewidth=2, color=color_avg, label='Average')
    if phase_switch:
        plt.axvline(x=phase_switch, color='red', linestyle='--',
                    linewidth=1.2, label=f'Phase switch (gen {phase_switch})')
    plt.xlabel("Generation")
    plt.ylabel("Best Utility (minimize)")
    plt.title(f"{name} Convergence – {sname}{n_str}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 1.  BINARY-CODED GENETIC ALGORITHM  (BCGA)
# ============================================================

def run_bcga_once(s, bounds, print_gen=False):
    M = s['M']
    pop = np.array([repair_solution(random_solution(s), s) for _ in range(POP_SIZE)])
    best_fit = np.inf
    best_sol = (np.inf, 0, N_FIXED, [], 0)
    history  = []

    for gen in range(1, MAX_GEN + 1):
        fitnesses = []
        for chrom in pop:
            fit, U, L, eta, C, idx, n = evaluate(chrom, s, bounds)
            fitnesses.append(fit)
            if fit < best_fit:
                best_fit = fit
                best_sol = (U, len(idx), n, [int(i) for i in idx], fit)

        history.append(best_fit)
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d}: best U = {best_fit:.6f}")

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

        fitnesses = np.array(fitnesses)
        new_pop = []
        while len(new_pop) < POP_SIZE:
            t1 = min(random.sample(range(POP_SIZE), 3), key=lambda i: fitnesses[i])
            t2 = min(random.sample(range(POP_SIZE), 3), key=lambda i: fitnesses[i])
            p1, p2 = pop[t1], pop[t2]

            if random.random() < CR:
                alpha = np.random.rand(M)       # <-- length M
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            for j in range(M):                  # <-- iterate over M genes
                if random.random() < MUT_RATE:
                    child[j] = random.random()

            child = repair_solution(child, s)
            new_pop.append(child)

        pop = np.array(new_pop)

    U, m, n, idx, _ = best_sol
    return U, m, n, idx, history


def run_bcga(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  BCGA  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_bcga_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("BCGA", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("BCGA", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "BCGA", s['name'], 'steelblue', 'navy', n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# 2.  REAL-CODED GENETIC ALGORITHM  (RCGA)
# ============================================================

def sbx_crossover(p1, p2, eta_c=15.0):
    child1, child2 = np.zeros(len(p1)), np.zeros(len(p2))
    for i in range(len(p1)):
        u = random.random()
        beta = (2*u)**(1/(eta_c+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta_c+1))
        child1[i] = 0.5*((1+beta)*p1[i] + (1-beta)*p2[i])
        child2[i] = 0.5*((1-beta)*p1[i] + (1+beta)*p2[i])
    return np.clip(child1, 0, 1), np.clip(child2, 0, 1)


def poly_mutation(ind, eta_m=20.0):
    out = ind.copy()
    for i in range(len(out)):
        if random.random() < MUT_RATE:
            u = random.random()
            delta = (2*u)**(1/(eta_m+1))-1 if u < 0.5 else 1-(2*(1-u))**(1/(eta_m+1))
            out[i] = np.clip(out[i] + delta, 0, 1)
    return out


def tournament_select(pop, fitnesses, k=3):
    sel  = random.sample(range(len(pop)), k)
    best = min(sel, key=lambda i: fitnesses[i])
    return pop[best]


def run_rcga_once(s, bounds, print_gen=False):
    M = s['M']
    pop = np.array([repair_solution(random_solution(s), s) for _ in range(POP_SIZE)])
    best_fit = np.inf
    best_sol = (np.inf, 0, N_FIXED, [], 0)
    history  = []

    for gen in range(1, MAX_GEN + 1):
        fitnesses = []
        for chrom in pop:
            fit, U, L, eta, C, idx, n = evaluate(chrom, s, bounds)
            fitnesses.append(fit)
            if fit < best_fit:
                best_fit = fit
                best_sol = (U, len(idx), n, [int(i) for i in idx], fit)

        history.append(best_fit)
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d}: best U = {best_fit:.6f}")

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

        fitnesses = np.array(fitnesses)
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)
            if random.random() < CR:
                c1, c2 = sbx_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = repair_solution(poly_mutation(c1), s)
            c2 = repair_solution(poly_mutation(c2), s)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        pop = np.array(new_pop)

    U, m, n, idx, _ = best_sol
    return U, m, n, idx, history


def run_rcga(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  RCGA  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_rcga_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("RCGA", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("RCGA", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "RCGA", s['name'], 'mediumseagreen', 'darkgreen', n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# 3.  HYBRID GA  (BCGA phase → RCGA phase)
# ============================================================
PHASE_SWITCH = 150

def run_hybrid_once(s, bounds, print_gen=False):
    M = s['M']
    pop = np.array([repair_solution(random_solution(s), s) for _ in range(POP_SIZE)])
    best_fit = np.inf
    best_sol = (np.inf, 0, N_FIXED, [], 0)
    history  = []

    for gen in range(1, MAX_GEN + 1):
        fitnesses = []
        for chrom in pop:
            fit, U, L, eta, C, idx, n = evaluate(chrom, s, bounds)
            fitnesses.append(fit)
            if fit < best_fit:
                best_fit = fit
                best_sol = (U, len(idx), n, [int(i) for i in idx], fit)

        history.append(best_fit)
        phase = "BCGA" if gen <= PHASE_SWITCH else "RCGA"
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d} [{phase}]: best U = {best_fit:.6f}")

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

        fitnesses = np.array(fitnesses)
        new_pop = []

        if gen <= PHASE_SWITCH:
            while len(new_pop) < POP_SIZE:
                t1 = min(random.sample(range(POP_SIZE), 3), key=lambda i: fitnesses[i])
                t2 = min(random.sample(range(POP_SIZE), 3), key=lambda i: fitnesses[i])
                p1, p2 = pop[t1], pop[t2]
                alpha  = np.random.rand(M)      # <-- length M
                child  = alpha * p1 + (1 - alpha) * p2 if random.random() < CR else p1.copy()
                for j in range(M):              # <-- iterate over M genes
                    if random.random() < MUT_RATE:
                        child[j] = random.random()
                new_pop.append(repair_solution(child, s))
        else:
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop, fitnesses)
                p2 = tournament_select(pop, fitnesses)
                c1, c2 = sbx_crossover(p1, p2) if random.random() < CR else (p1.copy(), p2.copy())
                c1 = repair_solution(poly_mutation(c1, eta_m=20.0), s)
                c2 = repair_solution(poly_mutation(c2, eta_m=20.0), s)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2)

        pop = np.array(new_pop)

    U, m, n, idx, _ = best_sol
    return U, m, n, idx, history


def run_hybrid(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  Hybrid GA  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_hybrid_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("Hybrid GA", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("Hybrid GA", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "Hybrid GA", s['name'],
                     'mediumorchid', 'indigo', phase_switch=PHASE_SWITCH, n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# 4.  PARTICLE SWARM OPTIMIZATION  (PSO)
# ============================================================
W_START = 0.9
W_END   = 0.4
C1      = 1.5
C2      = 1.5
V_MAX   = 0.4

def run_pso_once(s, bounds, print_gen=False):
    M    = s['M']
    dim  = M                        # <-- dim is now M only

    pos  = np.array([repair_solution(random_solution(s), s) for _ in range(POP_SIZE)])
    vel  = np.random.uniform(-V_MAX, V_MAX, (POP_SIZE, dim))

    pbest_pos = pos.copy()
    pbest_fit = np.full(POP_SIZE, np.inf)
    gbest_pos = None
    gbest_fit = np.inf
    gbest_sol = (np.inf, 0, N_FIXED, [], 0)
    history   = []

    for gen in range(1, MAX_GEN + 1):
        w = W_START - (W_START - W_END) * gen / MAX_GEN

        for i in range(POP_SIZE):
            fit, U, L, eta, C, idx, n = evaluate(pos[i], s, bounds)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = pos[i].copy()
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_pos = pos[i].copy()
                gbest_sol = (U, len(idx), n, [int(i2) for i2 in idx], fit)

        history.append(gbest_fit)
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d}: best U = {gbest_fit:.6f}")

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

        for i in range(POP_SIZE):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            vel[i] = (w * vel[i]
                      + C1 * r1 * (pbest_pos[i] - pos[i])
                      + C2 * r2 * (gbest_pos   - pos[i]))
            vel[i] = np.clip(vel[i], -V_MAX, V_MAX)
            pos[i] = np.clip(pos[i] + vel[i], 0, 1)
            pos[i] = repair_solution(pos[i], s)

    U, m, n, idx, _ = gbest_sol
    return U, m, n, idx, history


def run_pso(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  PSO  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_pso_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("PSO", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("PSO", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "PSO", s['name'], 'coral', 'darkred', n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# 5.  TEACHING-LEARNING-BASED OPTIMIZATION  (TLBO)
# ============================================================

def run_tlbo_once(s, bounds, print_gen=False):
    M   = s['M']
    dim = M                         # <-- dim is now M only
    pop = np.array([repair_solution(random_solution(s), s) for _ in range(POP_SIZE)])
    best_fit = np.inf
    best_sol = (np.inf, 0, N_FIXED, [], 0)
    history  = []

    for gen in range(1, MAX_GEN + 1):
        fitnesses = np.array([evaluate(c, s, bounds)[0] for c in pop])

        best_idx_local = np.argmin(fitnesses)
        if fitnesses[best_idx_local] < best_fit:
            best_fit = fitnesses[best_idx_local]
            fit, U, L, eta, C, idx, n = evaluate(pop[best_idx_local], s, bounds)
            best_sol = (U, len(idx), n, [int(i) for i in idx], fit)

        history.append(best_fit)
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d}: best U = {best_fit:.6f}")

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

        teacher  = pop[np.argmin(fitnesses)]
        mean_pop = np.mean(pop, axis=0)
        TF       = round(1 + random.random())

        new_pop = []
        for i in range(POP_SIZE):
            r = np.random.rand(dim)
            cand = np.clip(pop[i] + r * (teacher - TF * mean_pop), 0, 1)
            cand = repair_solution(cand, s)
            fit_c = evaluate(cand, s, bounds)[0]
            new_pop.append(cand if fit_c < fitnesses[i] else pop[i])
            if fit_c < fitnesses[i]:
                fitnesses[i] = fit_c
        pop = np.array(new_pop)

        new_pop = []
        for i in range(POP_SIZE):
            j = random.choice([x for x in range(POP_SIZE) if x != i])
            r = np.random.rand(dim)
            if fitnesses[i] < fitnesses[j]:
                cand = np.clip(pop[i] + r * (pop[i] - pop[j]), 0, 1)
            else:
                cand = np.clip(pop[i] + r * (pop[j] - pop[i]), 0, 1)
            cand = repair_solution(cand, s)
            fit_c, U, L, eta, C, idx, n = evaluate(cand, s, bounds)
            if fit_c < fitnesses[i]:
                new_pop.append(cand)
                fitnesses[i] = fit_c
                if fit_c < best_fit:
                    best_fit = fit_c
                    best_sol = (U, len(idx), n, [int(k) for k in idx], fit_c)
            else:
                new_pop.append(pop[i])
        pop = np.array(new_pop)

    U, m, n, idx, _ = best_sol
    return U, m, n, idx, history


def run_tlbo(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  TLBO  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_tlbo_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("TLBO", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("TLBO", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "TLBO", s['name'], 'plum', 'purple', n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# 6.  DIFFERENTIAL EVOLUTION  (DE/rand/1/bin)
# ============================================================
F_DE = 0.8

def run_de_once(s, bounds, print_gen=False):
    M   = s['M']
    dim = M                         # <-- dim is now M only
    pop = np.array([repair_solution(random_solution(s), s) for _ in range(POP_SIZE)])
    fitnesses = np.array([evaluate(c, s, bounds)[0] for c in pop])

    best_idx_local = np.argmin(fitnesses)
    best_fit = fitnesses[best_idx_local]
    fit, U, L, eta, C, idx, n = evaluate(pop[best_idx_local], s, bounds)
    best_sol = (U, len(idx), n, [int(i) for i in idx], fit)
    history  = [best_fit]

    for gen in range(1, MAX_GEN + 1):
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d}: best U = {best_fit:.6f}")

        for i in range(POP_SIZE):
            cands = [x for x in range(POP_SIZE) if x != i]
            a, b, c = random.sample(cands, 3)
            mutant = np.clip(pop[a] + F_DE * (pop[b] - pop[c]), 0, 1)

            j_rand = random.randint(0, dim - 1)
            trial  = np.where(
                (np.random.rand(dim) < CR) | (np.arange(dim) == j_rand),
                mutant, pop[i])
            trial  = repair_solution(trial, s)

            fit_t, U, L, eta, C, idx, n = evaluate(trial, s, bounds)
            if fit_t <= fitnesses[i]:
                pop[i]       = trial
                fitnesses[i] = fit_t
                if fit_t < best_fit:
                    best_fit = fit_t
                    best_sol = (U, len(idx), n, [int(k) for k in idx], fit_t)

        history.append(best_fit)

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

    U, m, n, idx, _ = best_sol
    return U, m, n, idx, history


def run_de(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  DE  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_de_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("DE", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("DE", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "DE", s['name'], 'sandybrown', 'saddlebrown', n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# 7.  ARTIFICIAL BEE COLONY  (ABC)
# ============================================================
NUM_EMPLOYED = POP_SIZE // 2
NUM_ONLOOKER = POP_SIZE // 2
SCOUT_LIMIT  = 20

def abc_neighbour(source, all_sources, src_idx, dim):
    k   = random.choice([x for x in range(len(all_sources)) if x != src_idx])
    j   = random.randint(0, dim - 1)
    phi = random.uniform(-1, 1)
    nb  = source.copy()
    nb[j] = np.clip(source[j] + phi * (source[j] - all_sources[k][j]), 0, 1)
    return nb


def run_abc_once(s, bounds, print_gen=False):
    M   = s['M']
    dim = M                         # <-- dim is now M only

    sources   = np.array([repair_solution(random_solution(s), s) for _ in range(NUM_EMPLOYED)])
    fitnesses = np.array([evaluate(c, s, bounds)[0] for c in sources])
    trials    = np.zeros(NUM_EMPLOYED)

    best_fit = np.min(fitnesses)
    best_src = np.argmin(fitnesses)
    fit, U, L, eta, C, idx, n = evaluate(sources[best_src], s, bounds)
    best_sol = (U, len(idx), n, [int(i) for i in idx], fit)
    history  = []

    for gen in range(1, MAX_GEN + 1):

        for i in range(NUM_EMPLOYED):
            nb   = abc_neighbour(sources[i], sources, i, dim)
            nb   = repair_solution(nb, s)
            fit_n, U, L, eta, C, idx, n = evaluate(nb, s, bounds)
            if fit_n < fitnesses[i]:
                sources[i]   = nb
                fitnesses[i] = fit_n
                trials[i]    = 0
                if fit_n < best_fit:
                    best_fit = fit_n
                    best_sol = (U, len(idx), n, [int(k) for k in idx], fit_n)
            else:
                trials[i] += 1

        inv_fit = 1.0 / (fitnesses - np.min(fitnesses) + 1e-10)
        probs   = inv_fit / np.sum(inv_fit)

        for _ in range(NUM_ONLOOKER):
            i    = np.random.choice(NUM_EMPLOYED, p=probs)
            nb   = abc_neighbour(sources[i], sources, i, dim)
            nb   = repair_solution(nb, s)
            fit_n, U, L, eta, C, idx, n = evaluate(nb, s, bounds)
            if fit_n < fitnesses[i]:
                sources[i]   = nb
                fitnesses[i] = fit_n
                trials[i]    = 0
                if fit_n < best_fit:
                    best_fit = fit_n
                    best_sol = (U, len(idx), n, [int(k) for k in idx], fit_n)
            else:
                trials[i] += 1

        for i in range(NUM_EMPLOYED):
            if trials[i] > SCOUT_LIMIT:
                sources[i]   = repair_solution(random_solution(s), s)
                fitnesses[i] = evaluate(sources[i], s, bounds)[0]
                trials[i]    = 0

        history.append(best_fit)
        if print_gen and gen % 50 == 0:
            print(f"    Gen {gen:3d}: best U = {best_fit:.6f}")

        if convergence_check(history):
            if print_gen:
                print(f"    Converged at gen {gen}")
            break

    U, m, n, idx, _ = best_sol
    return U, m, n, idx, history


def run_abc(s, bounds, n_val=None):
    print(f"\n{'='*55}")
    print(f"  ABC  |  {s['name']}  |  n={N_FIXED}")
    print(f"{'='*55}")
    utilities, times, stopped_gens, all_histories = [], [], [], []

    for r in range(RUNS):
        t0 = time.time()
        U, m, n, idx, hist = run_abc_once(s, bounds, print_gen=(r == 0))
        elapsed = time.time() - t0

        if r == 0:
            print_run1_result("ABC", s['name'], U, m, n, idx, s['xi'])

        utilities.append(U)
        times.append(elapsed)
        stopped_gens.append(len(hist))
        all_histories.append(hist)

    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]

    print_stats("ABC", s['name'], utilities, times, stopped_gens, n=N_FIXED)
    plot_convergence(padded, "ABC", s['name'], 'gold', 'goldenrod', n=N_FIXED)
    return utilities, times, stopped_gens


# ============================================================
# COMPARATIVE SUMMARY TABLE
# ============================================================

def print_comparison_table(results, setting_name, n=None):
    n_str = f"  |  n={n}" if n is not None else ""
    print(f"\n{'='*75}")
    print(f"  COMPARISON TABLE  |  {setting_name}{n_str}")
    print(f"{'='*75}")
    header = f"{'Algorithm':<12} {'Best U':>10} {'Avg U':>10} {'Std U':>10} {'Avg Time(ms)':>13} {'Avg Gen':>8}"
    print(header)
    print('-'*75)
    for algo, (utils, times, gens) in results.items():
        print(f"{algo:<12} {np.min(utils):>10.6f} {np.mean(utils):>10.6f} "
              f"{np.std(utils):>10.6f} {np.mean(times)*1000:>13.2f} {np.mean(gens):>8.1f}")
    print('='*75)


def print_transaction_summary_table(tx_results, setting_name, algo_name):
    print(f"\n{'='*65}")
    print(f"  TRANSACTION EXPERIMENT  |  {algo_name}  |  {setting_name}")
    print(f"{'='*65}")
    header = f"{'n (txns)':>10} {'Best U':>10} {'Avg U':>10} {'Std U':>10} {'Avg Gen':>8}"
    print(header)
    print('-'*65)
    for n_val, (utils, times, gens) in sorted(tx_results.items()):
        print(f"{n_val:>10} {np.min(utils):>10.6f} {np.mean(utils):>10.6f} "
              f"{np.std(utils):>10.6f} {np.mean(gens):>8.1f}")
    print('='*65)


def plot_transaction_experiment(tx_results, setting_name, algo_name):
    """
    Bar/line plot of Best Utility vs n for one algo & setting.
    """
    n_vals  = sorted(tx_results.keys())
    best_us = [np.min(tx_results[n][0]) for n in n_vals]
    avg_us  = [np.mean(tx_results[n][0]) for n in n_vals]

    plt.figure(figsize=(7, 4))
    plt.plot(n_vals, best_us, 'o-', label='Best U',  color='navy',    linewidth=2)
    plt.plot(n_vals, avg_us,  's--', label='Avg U', color='steelblue', linewidth=1.5)
    plt.xlabel("Number of Transactions (n)")
    plt.ylabel("Utility U (minimize)")
    plt.title(f"Effect of n on Utility – {algo_name} | {setting_name}")
    plt.xticks(n_vals)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    algos = [
        ("BCGA",   run_bcga),
        ("RCGA",   run_rcga),
        ("Hybrid", run_hybrid),
        ("PSO",    run_pso),
        ("TLBO",   run_tlbo),
        ("DE",     run_de),
        ("ABC",    run_abc),
    ]

    # ===========================================================
    # PART A: Main run with fixed n = 50
    # ===========================================================
    print("\n" + "#"*60)
    print("  PART A: Main Experiment  (n = 50 fixed for all settings)")
    print("#"*60)

    N_FIXED = 50   # fixed transactions for all settings

    for s in SETTINGS:
        print(f"\n\n{'#'*60}")
        print(f"  SETTING: {s['name']}  (M={s['M']} validators, n={N_FIXED})")
        print(f"{'#'*60}")

        bounds  = compute_min_max(s)
        results = {}

        for algo_name, algo_fn in algos:
            utils, times, gens = algo_fn(s, bounds)
            results[algo_name] = (utils, times, gens)

        print_comparison_table(results, s['name'], n=N_FIXED)

    # ===========================================================
    # PART B: Transaction experiment n in [20, 40, 60, 80, 100]
    # ===========================================================
    print("\n\n" + "#"*60)
    print("  PART B: Transaction Experiment")
    print(f"  n values tested: {TRANSACTION_VALUES}")
    print("#"*60)

    for s in SETTINGS:
        print(f"\n\n{'='*60}")
        print(f"  SETTING: {s['name']}  (M={s['M']} validators)")
        print(f"{'='*60}")

        for algo_name, algo_fn in algos:
            tx_results = {}   # { n_val: (utils, times, gens) }

            for n_val in TRANSACTION_VALUES:
                # Update global N_FIXED before running
                N_FIXED = n_val

                # Override t = n_val so feasibility check (t <= n <= X) passes
                s_temp = dict(s)
                s_temp['t'] = n_val

                # Recompute bounds using s_temp so Lmin/Cmax use correct t
                bounds_temp = compute_min_max(s_temp)

                print(f"\n  >> {algo_name} | {s['name']} | n={n_val}")
                utils, times, gens = algo_fn(s_temp, bounds_temp)
                tx_results[n_val] = (utils, times, gens)

            print_transaction_summary_table(tx_results, s['name'], algo_name)
            plot_transaction_experiment(tx_results, s['name'], algo_name)

    # Reset N_FIXED back to default
    N_FIXED = 50