# PSO
import numpy as np, random, time
from gap_base import *


class GAP_PSO:
    def __init__(self, m, n, cost, resource, capacity, method="repair"):
        self.m = m; self.n = n; self.cost = cost
        self.resource = resource; self.capacity = capacity
        self.method  = method
        self.w  = 0.4; self.c1 = 0.3; self.c2 = 0.3
        self.PENALTY = compute_penalty(m, n, cost)

    # ---- selection fitness ----
    def fit_sel(self, pos):
        if self.method == "penalty":
            return fitness_guided_penalty(pos, self.m, self.n,
                self.cost, self.resource, self.capacity, self.PENALTY)
        return fitness_raw(pos, self.m, self.n, self.cost)

    def fit_true(self, pos):
        return fitness_true(pos, self.m, self.n,
                            self.cost, self.resource, self.capacity)

    def fix(self, pos):
        """
        Constraint handling 

        PENALTY     : repair is still applied so the swarm always has
                      feasible solutions for best tracking.
                      Penalty only affects pbest SELECTION pressure —
                      this is what distinguishes penalty from repair.
        REPAIR      : greedy repair after every position update.
        REPLACEMENT : infeasible particles are replaced by a new
                      feasible individual from scratch.
        """
        if self.method in ("penalty", "repair"):
            return repair_individual(pos[:], self.m, self.n,
                                     self.cost, self.resource, self.capacity)
        # replacement
        if not is_feasible_bool(pos, self.m, self.n,
                                self.resource, self.capacity):
            return create_feasible_individual(self.m, self.n,
                self.cost, self.resource, self.capacity)
        return pos

    def run(self):
        # All initial particles are feasible
        positions  = [create_feasible_individual(self.m, self.n,
                          self.cost, self.resource, self.capacity)
                      for _ in range(POP_SIZE)]
        pbest_pos  = [p[:] for p in positions]
        pbest_fit  = [self.fit_sel(p) for p in positions]

        # Seed gbest from gen-0 true fitness — always a real profit, never -inf
        true0      = [self.fit_true(p) for p in positions]
        bi         = int(np.argmax(true0))
        gbest_pos  = positions[bi][:]
        gbest_fit  = true0[bi]
        history    = []

        for _ in range(ITERATIONS):
            for s in range(POP_SIZE):
                positions[s] = self.fix(positions[s])

                # pbest uses selection fitness (penalty-aware in penalty mode)
                fg = self.fit_sel(positions[s])
                if fg > pbest_fit[s]:
                    pbest_fit[s] = fg
                    pbest_pos[s] = positions[s][:]

                # gbest uses TRUE fitness (penalty=0, feasible solutions only)
                ft = self.fit_true(positions[s])
                if ft > gbest_fit:
                    gbest_fit = ft
                    gbest_pos = positions[s][:]

            history.append(gbest_fit)

            # Velocity / position update
            for s in range(POP_SIZE):
                for j in range(self.n):
                    r = random.random()
                    if r < self.c1:
                        positions[s][j] = pbest_pos[s][j]
                    elif r < self.c1 + self.c2:
                        positions[s][j] = gbest_pos[j]
                    elif random.random() < self.w:
                        positions[s][j] = random.randint(0, self.m - 1)

        return gbest_pos, gbest_fit, history


def run_experiment(method, filepath="gap12.txt", color="#D85A30"):
    problems = read_gap_file(filepath)
    m, n, cost, resource, capacity = problems[0]
    algo = GAP_PSO(m, n, cost, resource, capacity, method=method)

    extra = (f"Penalty (auto)   : {algo.PENALTY}  (> max profit, guides pbest selection)"
             if method == "penalty" else "")
    print_header("PSO", method, filepath, problems, m, n, extra)

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
                 "PSO", method, filepath, color)


# Choose method 
if __name__ == "__main__":
    # Change to "repair" or "replacement" as needed
    METHOD = "penalty"
    COLORS = {"penalty": "#378ADD", "repair": "#639922", "replacement": "#D85A30"}
    run_experiment(METHOD, color=COLORS[METHOD])