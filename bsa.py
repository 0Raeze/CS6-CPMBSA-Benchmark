import numpy as np

class StandardBSA:
    def __init__(self, func, bounds, dim, pop_size=30, max_evals=9000):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.max_iters = max_evals // pop_size
        self.cross_prob = 0.8
        self.F_range = (0.2, 2.0)

    def optimize(self):
        # --- Initialization ---
        low, high = self.bounds
        pop = np.random.uniform(low, high, (self.pop_size, self.dim))
        fit = np.array([self.func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fit)
        best_pos = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

        # Initialize convergence curve
        convergence_curve = []

        # --- Main Loop ---
        for _ in range(self.max_iters):
            # Create historical population (BSA memory)
            old_pop = np.copy(pop)
            np.random.shuffle(old_pop)

            # Scaling factor
            F = np.random.uniform(self.F_range[0], self.F_range[1])

            # --- Mutation (Standard BSA) ---
            mutant = pop + F * (old_pop - pop)
            mutant = np.clip(mutant, low, high)

            # --- Crossover ---
            cross_mask = np.random.rand(self.pop_size, self.dim) < self.cross_prob
            trial = np.where(cross_mask, mutant, pop)

            # --- Selection ---
            trial_fit = np.array([self.func(ind) for ind in trial])
            improved = trial_fit < fit
            if np.any(improved):
                pop[improved] = trial[improved]
                fit[improved] = trial_fit[improved]

            # --- Update best solution ---
            cur_best_idx = np.argmin(fit)
            if fit[cur_best_idx] < best_fit:
                best_fit = float(fit[cur_best_idx])
                best_pos = pop[cur_best_idx].copy()
        
            # Append to convergence curve
            convergence_curve.append(best_fit)
                
        return best_fit, best_pos, convergence_curve

