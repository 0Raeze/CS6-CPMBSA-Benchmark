import numpy as np

class CPMBSA:
    def __init__(self, func, bounds, dim, pop_size=30, max_evals=9000, p=1.5):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.max_iters = max_evals // pop_size
        self.p = p  # Power-mutation control parameter
        self.F_range = (0.2, 1.8)
        self.cross_prob = 0.8

    def optimize(self):
        # --- Initialization ---
        low, high = self.bounds
        pop = np.random.uniform(low, high, (self.pop_size, self.dim))
        fit = np.array([self.func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fit)
        best_pos = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

        # Initialize Convergence Curve
        convergence_curve = []

        # --- Main Loop ---
        for _ in range(self.max_iters):
            # Shuffle population (BSA historical memory)
            old_pop = np.copy(pop)
            np.random.shuffle(old_pop)

            # Scaling factor F (adaptive)
            F = np.random.uniform(self.F_range[0], self.F_range[1])

            # --- Power Mutation ---
            # Random strength sampled from power distribution
            r = np.random.rand(self.pop_size, self.dim)
            delta = np.sign(np.random.randn(self.pop_size, self.dim)) * (r ** (1.0 / self.p))
            mutant = pop + F * (old_pop - pop) * delta

            # Keep within bounds
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

            # --- Update best ---
            cur_best_idx = np.argmin(fit)
            if fit[cur_best_idx] < best_fit:
                best_fit = float(fit[cur_best_idx])
                best_pos = pop[cur_best_idx].copy()

            # Append to convergence curve
            convergence_curve.append(best_fit)

        return best_fit, best_pos, convergence_curve
