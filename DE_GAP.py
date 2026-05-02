# DE

import numpy as np, random, time
from gap_base import *


class GAP_DE:
    def __init__(self, m, n, cost, resource, capacity, method="repair"):
        self.m = m; self.n = n; self.cost = cost
        self.resource = resource; self.capacity = capacity
        self.method  = method
        self.CR      = CROSSOVER_RATE
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

    def mutate(self, pop, idx, best):
        idxs = [k for k in range(POP_SIZE) if k != idx]
        a, b = random.sample(idxs, 2)
        mutant = []
        for j in range(self.n):
            r = random.random()
            if r < 0.4:   mutant.append(best[j])
            elif r < 0.7: mutant.append(pop[a][j] if pop[a][j] != pop[b][j]
                                        else pop[b][j])
            else:         mutant.append(random.randint(0, self.m - 1))
        return mutant

    def crossover(self, target, mutant):
        return [mutant[j] if random.random() < self.CR else target[j]
                for j in range(self.n)]

    def run(self):
        pop    = [create_feasible_individual(self.m, self.n,
                      self.cost, self.resource, self.capacity)
                  for _ in range(POP_SIZE)]
        true0  = [self.fit_true(p) for p in pop]
        bi     = int(np.argmax(true0))
        bf     = true0[bi]; bs = pop[bi][:]
        history = []

        for _ in range(ITERATIONS):
            new_pop = []
            for i in range(POP_SIZE):
                mutant = self.mutate(pop, i, bs)
                trial  = self.crossover(pop[i], mutant)
                for j in range(self.n):
                    if random.random() < MUTATION_RATE:
                        trial[j] = random.randint(0, self.m - 1)
                trial   = self.fix(trial)
                # Selection: trial replaces target only if better
                new_sol = trial if self.fit_sel(trial) > self.fit_sel(pop[i]) \
                          else pop[i]
                new_pop.append(new_sol)
            pop = new_pop

            for p in pop:
                ft = self.fit_true(p)
                if ft > bf: bf = ft; bs = p[:]
            history.append(bf)

        return bs, bf, history


def run_experiment(method, filepath="gap12.txt", color="#D4537E"):
    problems = read_gap_file(filepath)
    m, n, cost, resource, capacity = problems[0]
    algo = GAP_DE(m, n, cost, resource, capacity, method=method)

    extra = (f"Penalty (auto)   : {algo.PENALTY}  (> max profit, guides trial-vs-target selection)"
             if method == "penalty" else "")
    print_header("DE", method, filepath, problems, m, n, extra)

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
                 "DE", method, filepath, color)


if __name__ == "__main__":
    METHOD = "replacement"
    COLORS = {"penalty": "#378ADD", "repair": "#D4537E", "replacement": "#BA7517"}
    run_experiment(METHOD, color=COLORS[METHOD])