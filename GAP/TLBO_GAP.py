# TLBO 
import numpy as np, random, time
from gap_base import *


class GAP_TLBO:
    def __init__(self, m, n, cost, resource, capacity, method="repair"):
        self.m = m; self.n = n; self.cost = cost
        self.resource = resource; self.capacity = capacity
        self.method  = method
        self.PENALTY = compute_penalty(m, n, cost)

    def fit_sel(self, sol):
        if self.method == "penalty":
            return fitness_guided_penalty(sol, self.m, self.n,
                self.cost, self.resource, self.capacity, self.PENALTY)
        return fitness_raw(sol, self.m, self.n, self.cost)

    def fit_true(self, sol):
        return fitness_true(sol, self.m, self.n,
                            self.cost, self.resource, self.capacity)

    def fix(self, sol):
        if self.method in ("penalty", "repair"):
            return repair_individual(sol[:], self.m, self.n,
                                     self.cost, self.resource, self.capacity)
        if not is_feasible_bool(sol, self.m, self.n,
                                self.resource, self.capacity):
            return create_feasible_individual(self.m, self.n,
                self.cost, self.resource, self.capacity)
        return sol

    def run(self):
        pop    = [create_feasible_individual(self.m, self.n,
                      self.cost, self.resource, self.capacity)
                  for _ in range(POP_SIZE)]
        true0  = [self.fit_true(p) for p in pop]
        bi     = int(np.argmax(true0))
        bf     = true0[bi]; bs = pop[bi][:]
        history = []

        for _ in range(ITERATIONS):
            gfits   = [self.fit_sel(p) for p in pop]
            teacher = pop[int(np.argmax(gfits))]

            # Teacher phase
            for i in range(POP_SIZE):
                new = pop[i][:]
                for j in range(self.n):
                    if self.cost[teacher[j]][j] > self.cost[new[j]][j]:
                        new[j] = teacher[j]
                new = self.fix(new)
                if self.fit_sel(new) > gfits[i]:
                    pop[i] = new
                    gfits[i] = self.fit_sel(new)

            # Learner phase
            for i in range(POP_SIZE):
                partner = random.choice([k for k in range(POP_SIZE) if k != i])
                new = pop[i][:]
                for j in range(self.n):
                    if self.cost[pop[partner][j]][j] > self.cost[new[j]][j]:
                        new[j] = pop[partner][j]
                for j in range(self.n):
                    if random.random() < MUTATION_RATE:
                        new[j] = random.randint(0, self.m - 1)
                new = self.fix(new)
                if self.fit_sel(new) > self.fit_sel(pop[i]):
                    pop[i] = new

            # Update best — true fitness only (penalty=0)
            for p in pop:
                ft = self.fit_true(p)
                if ft > bf: bf = ft; bs = p[:]
            history.append(bf)

        return bs, bf, history


def run_experiment(method, filepath="gap12.txt", color="#7F77DD"):
    problems = read_gap_file(filepath)
    m, n, cost, resource, capacity = problems[0]
    algo = GAP_TLBO(m, n, cost, resource, capacity, method=method)

    extra = (f"Penalty (auto)   : {algo.PENALTY}  (> max profit, guides teacher/learner selection)"
             if method == "penalty" else "")
    print_header("TLBO", method, filepath, problems, m, n, extra)

    all_hist=[]; bvals=[]; rtimes=[]; best_ov=-float('inf'); best_sol=None
    ts = time.time()
    for run in range(NUM_RUNS):
        random.seed(SEED+run); np.random.seed(SEED+run)
        t0=time.time(); sol,val,hist=algo.run(); rt=time.time()-t0
        all_hist.append(hist); bvals.append(val); rtimes.append(rt)
        print(f"  Run {run+1:>2}  |  Best Fitness : {val:>8}  |  Time : {rt:.2f}s")
        if val>best_ov: best_ov=val; best_sol=sol
    total=time.time()-ts

    print_summary(best_ov, bvals, rtimes, total)
    print_feasibility(best_sol, m, n, resource, capacity)
    print_assignment(best_sol)
    plot_results(np.mean(all_hist,axis=0), rtimes,
                 "TLBO", method, filepath, color)


if __name__ == "__main__":
    METHOD = "penalty"
    COLORS = {"penalty": "#378ADD", "repair": "#7F77DD", "replacement": "#D4537E"}
    run_experiment(METHOD, color=COLORS[METHOD])