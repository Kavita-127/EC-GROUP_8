# RCGA - PENALTY method
import numpy as np, random, time
from gap_base import *


class GAP_RCGA:
    def __init__(self, m, n, cost, resource, capacity):
        self.m = m; self.n = n; self.cost = cost
        self.resource = resource; self.capacity = capacity
        self.LOWER = 0.0; self.UPPER = float(m)
        self.PENALTY = compute_penalty(m, n, cost)

    def decode(self, ind):
        return [min(int(x), self.m - 1) for x in ind]

    def encode(self, assign):
        return [float(assign[j]) + 0.5 for j in range(self.n)]

    def create_individual(self):
        assign = create_feasible_individual(
            self.m, self.n, self.cost, self.resource, self.capacity)
        return self.encode(assign)

    def repair(self, ind):
        dec = self.decode(ind)
        dec = repair_individual(dec, self.m, self.n,
                                self.cost, self.resource, self.capacity)
        return self.encode(dec)

    def fit_guided(self, ind):
        return fitness_guided_penalty(
            self.decode(ind), self.m, self.n,
            self.cost, self.resource, self.capacity, self.PENALTY)

    def fit_true(self, ind):
        return fitness_true(self.decode(ind), self.m, self.n,
                            self.cost, self.resource, self.capacity)

    def sbx(self, p1, p2):
        if random.random() > CROSSOVER_RATE: return p1[:], p2[:]
        c1, c2 = [], []
        for i in range(self.n):
            x1, x2 = p1[i], p2[i]
            if abs(x1-x2) < 1e-10: c1.append(x1); c2.append(x2); continue
            if x1 > x2: x1, x2 = x2, x1
            u = random.random()
            b = 1.0 + 2.0*(x1-self.LOWER)/(x2-x1)
            a = 2.0 - b**(-(ETA_C+1.0))
            bq = (u*a)**(1.0/(ETA_C+1.0)) if u<=1.0/a \
                 else (1.0/(2.0-u*a))**(1.0/(ETA_C+1.0))
            y1 = 0.5*((x1+x2)-bq*(x2-x1))
            b = 1.0 + 2.0*(self.UPPER-x2)/(x2-x1)
            a = 2.0 - b**(-(ETA_C+1.0))
            bq = (u*a)**(1.0/(ETA_C+1.0)) if u<=1.0/a \
                 else (1.0/(2.0-u*a))**(1.0/(ETA_C+1.0))
            y2 = 0.5*((x1+x2)+bq*(x2-x1))
            y1 = max(self.LOWER, min(self.UPPER-1e-9, y1))
            y2 = max(self.LOWER, min(self.UPPER-1e-9, y2))
            if random.random() < 0.5: c1.append(y1); c2.append(y2)
            else:                     c1.append(y2); c2.append(y1)
        return c1, c2

    def pmut(self, ind):
        ind = ind[:]
        for j in range(self.n):
            if random.random() < MUTATION_RATE:
                x = ind[j]; u = random.random()
                d = (2*u)**(1/(ETA_M+1))-1 if u < 0.5 \
                    else 1-(2*(1-u))**(1/(ETA_M+1))
                ind[j] = max(self.LOWER,
                             min(self.UPPER-1e-9,
                                 x + d*(self.UPPER-self.LOWER)))
        return ind

    def tournament(self, pop):
        sel = random.sample(pop, 3)
        return max(sel, key=self.fit_guided)

    def run(self):
        pop   = [self.create_individual() for _ in range(POP_SIZE)]
        true0 = [self.fit_true(p) for p in pop]
        bi    = int(np.argmax(true0))
        bf    = true0[bi]; bs = self.decode(pop[bi]); bi_ind = pop[bi][:]
        hist  = []
        for _ in range(ITERATIONS):
            pop = [self.repair(p) for p in pop]
            new = [bi_ind[:]]
            while len(new) < POP_SIZE:
                c1, c2 = self.sbx(self.tournament(pop), self.tournament(pop))
                c1 = self.repair(self.pmut(c1))
                c2 = self.repair(self.pmut(c2))
                new.append(c1)
                if len(new) < POP_SIZE: new.append(c2)
            pop = new[:POP_SIZE]
            for p in pop:
                ft = self.fit_true(p)
                if ft > bf: bf = ft; bs = self.decode(p); bi_ind = p[:]
            hist.append(bf)
        return bs, bf, hist


if __name__ == "__main__":
    filepath = "gap12.txt"
    problems = read_gap_file(filepath)
    m, n, cost, resource, capacity = problems[0]
    algo = GAP_RCGA(m, n, cost, resource, capacity)

    print_header("RCGA", "penalty", filepath, problems, m, n,
                 f"Penalty (auto)   : {algo.PENALTY}  (> max profit, guarantees feasible best)\n"
                 f"  SBX eta_c={ETA_C}  |  Poly-mutation eta_m={ETA_M}")

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
                 "RCGA","penalty", filepath, "#378ADD")