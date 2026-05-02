import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANT PARAMETERS
# ==========================================================
POPULATION_SIZE = 100
ARCHIVE_CAPACITY = 100
NUM_GENERATIONS  = 200
NUM_VARIABLES    = 30
MIN_BOUND  = 0
MAX_BOUND  = 1
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY  = 0.1
SBX_ETA = 15
PM_ETA = 20


# SPEA2 paper: k = sqrt(|P_t| + |A_t|)
NEIGHBORHOOD_SIZE = int(np.sqrt(POPULATION_SIZE + ARCHIVE_CAPACITY))


# ==========================================================
# PROBLEM FUNCTIONS (ZDT3)
# ==========================================================
def evaluate_objectives(chromosome):
    """Return (f1, f2) for ZDT3."""
    chromosome = np.asarray(chromosome)
    objective_1 = chromosome[0]
    aux_g  = 1 + 9 * np.sum(chromosome[1:]) / (len(chromosome) - 1)
    aux_h  = 1 - np.sqrt(objective_1 / aux_g) - (objective_1 / aux_g) * np.sin(10 * np.pi * objective_1)
    objective_2 = aux_g * aux_h
    return objective_1, objective_2


# ==========================================================
# WHETHER a DOMINATES b
# ==========================================================
def dominates(solution_a, solution_b):
    """Minimisation: solution_a dominates solution_b. solution_a, solution_b are 1-D numpy arrays."""
    return bool(np.all(solution_a <= solution_b) and np.any(solution_a < solution_b))


# ==========================================================
# FINE‑GRAINED FITNESS ASSIGNMENT (SPEA2)
# ==========================================================
def compute_fitness(individuals, neighborhood_k):
    """
    individuals      : numpy array (N, DIM)
    neighborhood_k   : neighbourhood size for density

    Returns objective_matrix (N,2), strength_vals, raw_fitness_vals, density_vals, total_fitness_vals
    """
    num_individuals  = len(individuals)
    objective_matrix = np.array([evaluate_objectives(ind) for ind in individuals])

    # ---- Strength S(i) ----
    strength_vals = np.zeros(num_individuals, dtype=int)
    for idx_i in range(num_individuals):
        for idx_j in range(num_individuals):
            if idx_i != idx_j and dominates(objective_matrix[idx_i], objective_matrix[idx_j]):
                strength_vals[idx_i] += 1

    # ---- Raw fitness R(i) ----
    raw_fitness_vals = np.zeros(num_individuals, dtype=float)
    for idx_i in range(num_individuals):
        for idx_j in range(num_individuals):
            if idx_i != idx_j and dominates(objective_matrix[idx_j], objective_matrix[idx_i]):
                raw_fitness_vals[idx_i] += strength_vals[idx_j]

    # ---- Density via k-th nearest neighbour ----
    obj_min     = objective_matrix.min(axis=0)
    obj_max     = objective_matrix.max(axis=0)
    range_denom = np.where(obj_max - obj_min == 0, 1e-9, obj_max - obj_min)
    normalized_obj = (objective_matrix - obj_min) / range_denom

    density_vals = np.zeros(num_individuals)
    for idx_i in range(num_individuals):
        dist_vector        = np.linalg.norm(normalized_obj - normalized_obj[idx_i], axis=1)
        dist_vector[idx_i] = np.inf
        sorted_distances   = np.sort(dist_vector)
        kth_distance       = sorted_distances[min(neighborhood_k - 1, num_individuals - 2)]
        density_vals[idx_i] = 1.0 / (kth_distance + 2.0)

    total_fitness_vals = raw_fitness_vals + density_vals
    return objective_matrix, strength_vals, raw_fitness_vals, density_vals, total_fitness_vals


# ==========================================================
# ARCHIVE UPDATE — returns INDICES
# ==========================================================
def build_archive(individuals, objective_matrix, fitness_vals, target_size, neighborhood_k):
    """
    Returns the indices (into `individuals`) of the target_size selected individuals.
    """
    nondominated_mask    = fitness_vals < 1.0
    nondominated_indices = np.where(nondominated_mask)[0]
    num_nondominated     = len(nondominated_indices)

    if num_nondominated == target_size:
        return nondominated_indices.copy()

    elif num_nondominated < target_size:
        chosen            = list(nondominated_indices)
        dominated_indices = np.where(~nondominated_mask)[0]
        sorted_dominated  = dominated_indices[np.argsort(fitness_vals[dominated_indices])]
        slots_needed      = target_size - len(chosen)
        chosen.extend(sorted_dominated[:slots_needed].tolist())
        return np.array(chosen)

    else:   # num_nondominated > target_size — truncation with k-th nearest neighbour
        candidate_obj_list = list(objective_matrix[nondominated_indices])
        candidate_idx_list = list(nondominated_indices)

        while len(candidate_idx_list) > target_size:
            num_candidates    = len(candidate_idx_list)
            candidate_obj_arr = np.array(candidate_obj_list)
            cand_obj_min      = candidate_obj_arr.min(axis=0)
            cand_obj_max      = candidate_obj_arr.max(axis=0)
            cand_range        = np.where(cand_obj_max - cand_obj_min == 0, 1e-9, cand_obj_max - cand_obj_min)
            cand_norm_obj     = (candidate_obj_arr - cand_obj_min) / cand_range

            kth_dist_arr = np.zeros(num_candidates)
            for idx_c in range(num_candidates):
                dist_c         = np.linalg.norm(cand_norm_obj - cand_norm_obj[idx_c], axis=1)
                dist_c[idx_c]  = np.inf
                sorted_dist_c  = np.sort(dist_c)
                kth_dist_arr[idx_c] = sorted_dist_c[min(neighborhood_k - 1, num_candidates - 2)]

            worst_idx = int(np.argmin(kth_dist_arr))
            candidate_idx_list.pop(worst_idx)
            candidate_obj_list.pop(worst_idx)

        return np.array(candidate_idx_list)


# ==========================================================
# BINARY TOURNAMENT SELECTION
# ==========================================================
def tournament_selection(archive_pop, archive_fit):
    """
    Selects POPULATION_SIZE parents via binary tournament from the archive.
    """
    archive_len = len(archive_pop)
    mating_pool = []
    for _ in range(POPULATION_SIZE):
        contestant_a, contestant_b = np.random.choice(archive_len, 2, replace=False)
        if archive_fit[contestant_a] < archive_fit[contestant_b]:
            winner = archive_pop[contestant_a]
        elif archive_fit[contestant_b] < archive_fit[contestant_a]:
            winner = archive_pop[contestant_b]
        else:
            winner = archive_pop[np.random.choice([contestant_a, contestant_b])]
        mating_pool.append(winner)
    return np.array(mating_pool)


# ==========================================================
# SIMULATED BINARY CROSSOVER (SBX)
# ==========================================================
def sbx_crossover(parent_1, parent_2):
    if np.random.rand() >= CROSSOVER_PROBABILITY:
        return parent_1.copy(), parent_2.copy()
    child_1, child_2 = [], []
    for gene_1, gene_2 in zip(parent_1, parent_2):
        rand_u = np.random.rand()
        if rand_u <= 0.5:
            spread_factor = (2 * rand_u) ** (1.0 / (SBX_ETA + 1))
        else:
            spread_factor = (1.0 / (2 * (1 - rand_u))) ** (1.0 / (SBX_ETA + 1))
        child_1.append(0.5 * ((1 + spread_factor) * gene_1 + (1 - spread_factor) * gene_2))
        child_2.append(0.5 * ((1 - spread_factor) * gene_1 + (1 + spread_factor) * gene_2))
    return np.clip(child_1, MIN_BOUND, MAX_BOUND), np.clip(child_2, MIN_BOUND, MAX_BOUND)


# ==========================================================
# POLYNOMIAL MUTATION
# ==========================================================
def polynomial_mutation(individual):
    if np.random.rand() >= MUTATION_PROBABILITY:
        return individual.copy()
    mutated = individual.copy()
    for gene_idx in range(len(mutated)):
        rand_r = np.random.rand()
        if rand_r < 0.5:
            perturbation = (2 * rand_r) ** (1.0 / (PM_ETA + 1)) - 1
        else:
            perturbation = 1 - (2 * (1 - rand_r)) ** (1.0 / (PM_ETA + 1))
        mutated[gene_idx] += perturbation * (MAX_BOUND - MIN_BOUND)
    return np.clip(mutated, MIN_BOUND, MAX_BOUND)


# ==========================================================
# VARIATION (CROSSOVER + MUTATION)
# ==========================================================
def variation(selected_pool):
    """
    Produces exactly POPULATION_SIZE offspring via SBX + polynomial mutation.
    """
    offspring_list = []
    for pair_idx in range(0, len(selected_pool) - 1, 2):
        parent_a, parent_b       = selected_pool[pair_idx], selected_pool[pair_idx + 1]
        offspring_a, offspring_b = sbx_crossover(parent_a, parent_b)
        offspring_list.append(polynomial_mutation(offspring_a))
        offspring_list.append(polynomial_mutation(offspring_b))
    return np.array(offspring_list[:POPULATION_SIZE])


# ==========================================================
# MAIN SPEA2 LOOP
# ==========================================================
def strength_pareto_evolutionary_algorithm():
    # A(0) = ∅  (empty archive, as specified in the SPEA2 paper).
    # The first archive A(1) is built from P(0) ∪ A(0) = P(0) below.
    initial_population = np.random.uniform(MIN_BOUND, MAX_BOUND, (POPULATION_SIZE, NUM_VARIABLES))
    initial_objectives = np.array([evaluate_objectives(ind) for ind in initial_population])

    # Pre-loop: fitness of P(0), build first archive A(1)
    current_obj, _, _, _, current_fitness = compute_fitness(initial_population, NEIGHBORHOOD_SIZE)
    archive_indices         = build_archive(initial_population, current_obj, current_fitness, ARCHIVE_CAPACITY, NEIGHBORHOOD_SIZE)
    current_archive         = initial_population[archive_indices]
    current_archive_fitness = current_fitness[archive_indices]

    for generation in range(NUM_GENERATIONS):

        selected_parents = tournament_selection(current_archive, current_archive_fitness)
        new_offspring    = variation(selected_parents)

        # Combine offspring P(t+1) with current archive A(t)
        merged_population = np.vstack([new_offspring, current_archive])
        merged_obj, _, _, _, merged_fitness = compute_fitness(merged_population, NEIGHBORHOOD_SIZE)

        # Build A(t+1) and extract its fitness in one step
        updated_indices         = build_archive(merged_population, merged_obj, merged_fitness, ARCHIVE_CAPACITY, NEIGHBORHOOD_SIZE)
        current_archive         = merged_population[updated_indices]
        current_archive_fitness = merged_fitness[updated_indices]

    # ---- Final non-dominated solutions ----
    final_objectives, _, _, _, final_fitness = compute_fitness(current_archive, NEIGHBORHOOD_SIZE)
    pareto_mask      = final_fitness < 1.0
    pareto_front_obj = final_objectives[pareto_mask]

    return initial_objectives, pareto_front_obj


# ==========================================================
# PLOTTING
# ==========================================================
def plot_results(initial_obj, pareto_obj):
    # ZDT3 — filter to only the non-dominated portion of the curve
    true_front_f1 = np.linspace(0, 1, 1000)
    true_front_f2 = 1 - np.sqrt(true_front_f1) - true_front_f1 * np.sin(10 * np.pi * true_front_f1)

    # Keep only non-dominated points from the curve
    dominance_mask = np.ones(len(true_front_f1), dtype=bool)
    for curve_i in range(len(true_front_f1)):
        for curve_j in range(len(true_front_f1)):
            if curve_i != curve_j and true_front_f1[curve_j] <= true_front_f1[curve_i] and true_front_f2[curve_j] <= true_front_f2[curve_i]:
                if true_front_f1[curve_j] < true_front_f1[curve_i] or true_front_f2[curve_j] < true_front_f2[curve_i]:
                    dominance_mask[curve_i] = False
                    break

    true_front_f1 = true_front_f1[dominance_mask]
    true_front_f2 = true_front_f2[dominance_mask]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(initial_obj[:, 0], initial_obj[:, 1], s=15, alpha=0.6,
                label='Initial population')
    plt.scatter(true_front_f1, true_front_f2, s=3, c='red', label='True Pareto front')
    plt.xlabel('f1'); plt.ylabel('f2')
    plt.title('Initial Population')
    plt.grid(alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(pareto_obj[:, 0], pareto_obj[:, 1], s=20, c='green', alpha=0.7,
                label='SPEA2 Archive (non‑dominated)')
    plt.scatter(true_front_f1, true_front_f2, s=3, c='red', label='True Pareto front')
    plt.xlabel('f1'); plt.ylabel('f2')
    plt.title('Final Non‑dominated Solutions')
    plt.grid(alpha=0.3); plt.legend()

    plt.suptitle('SPEA2 Performance on ZDT3', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==========================================================
# RUN EXPERIMENT
# ==========================================================
if __name__ == "__main__":
    total_runs = 1
    for run_number in range(total_runs):
        run_start = time.perf_counter()
        initial_obj_data, final_pareto_data = strength_pareto_evolutionary_algorithm()
        run_elapsed = time.perf_counter() - run_start
        plot_results(initial_obj_data, final_pareto_data)
        print(f"Run {run_number + 1}: Time = {run_elapsed:.4f} sec")