# ABC — Artificial Bee Colony

import numpy as np, random, time
from gap_base import *


class GAP_ABC:
    def __init__(self, m, n, cost, resource, capacity, method="repair"):
        self.m = m; self.n = n; self.cost = cost
        self.resource = resource; self.capacity = capacity
        self.method   = method
        self.LIMIT    = POP_SIZE
        self.PENALTY  = compute_penalty(m, n, cost)

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

    def neighbor(self, sol, best):
        new = sol[:]
        for j in range(self.n):
            r = random.random()
            if r < 0.4:   new[j] = best[j]
            elif r < 0.7: new[j] = random.randint(0, self.m - 1)
        return new

    def run(self):
        pop    = [create_feasible_individual(self.m, self.n,
                      self.cost, self.resource, self.capacity)
                  for _ in range(POP_SIZE)]
        gfits  = [self.fit_sel(s) for s in pop]
        trials = [0] * POP_SIZE

        true0  = [self.fit_true(s) for s in pop]
        bi     = int(np.argmax(true0))
        bf     = true0[bi]; bs = pop[bi][:]
        history = []

        for _ in range(ITERATIONS):
            # Employed bees
            for i in range(POP_SIZE):
                nb = self.fix(self.neighbor(pop[i], bs))
                ng = self.fit_sel(nb)
                if ng > gfits[i]:
                    pop[i] = nb; gfits[i] = ng; trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bees
            shifted = np.array(gfits) - np.min(gfits) + 1
            probs   = shifted / shifted.sum()
            for _ in range(POP_SIZE):
                i  = np.random.choice(range(POP_SIZE), p=probs)
                nb = self.fix(self.neighbor(pop[i], bs))
                ng = self.fit_sel(nb)
                if ng > gfits[i]:
                    pop[i] = nb; gfits[i] = ng; trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bees
            for i in range(POP_SIZE):
                if trials[i] > self.LIMIT:
                    pop[i]    = create_feasible_individual(self.m, self.n,
                                    self.cost, self.resource, self.capacity)
                    gfits[i]  = self.fit_sel(pop[i])
                    trials[i] = 0

            # Update best — true fitness only (penalty=0)
            for s in pop:
                ft = self.fit_true(s)
                if ft > bf: bf = ft; bs = s[:]
            history.append(bf)

        return bs, bf, history


def run_experiment(method, filepath="gap12.txt", color="#BA7517"):
    problems = read_gap_file(filepath)
    m, n, cost, resource, capacity = problems[0]
    algo = GAP_ABC(m, n, cost, resource, capacity, method=method)

    extra = (f"Penalty (auto)   : {algo.PENALTY}  (> max profit, guides bee selection)\n"
             f"  Scout limit     : {algo.LIMIT}"
             if method == "penalty"
             else f"Scout limit     : {algo.LIMIT}")
    print_header("ABC", method, filepath, problems, m, n, extra)

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
                 "ABC", method, filepath, color)


if __name__ == "__main__":
    METHOD = "replacement"
    COLORS = {"penalty": "#378ADD", "repair": "#BA7517", "replacement": "#7F77DD"}
    run_experiment(METHOD, color=COLORS[METHOD])