import numpy as np

class FA:
    def __init__(self, func, bounds, dim, pop_size=30, max_evals=9000):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_gens = max_evals // pop_size # Convert evals to generations
        
        # FA Specific Parameters
        self.alpha = 0.5      # Randomization parameter (random step size)
        self.beta0 = 1.0      # Maximum attractiveness (when distance is 0)
        self.gamma = 1.0      # Light absorption coefficient (how fast light fades)
        self.alpha_damp = 0.99 # Dampening factor to reduce randomness over time

    def optimize(self):
        # 1. Initialize population randomly within bounds
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        
        # Calculate initial fitness (brightness: lower is better/brighter)
        fitness = np.array([self.func(ind) for ind in pop])
        
        # Track global best
        gbest_pos = pop[np.argmin(fitness)].copy()
        gbest_fit = np.min(fitness)
        
        # Main Generation Loop
        for gen in range(self.max_gens):
            # The O(N^2) double loop: Compare every firefly to every other firefly
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    
                    # If firefly j is brighter (has lower fitness) than firefly i
                    if fitness[j] < fitness[i]:
                        
                        # Calculate Euclidean distance between i and j
                        r = np.linalg.norm(pop[i] - pop[j])
                        
                        # Calculate attractiveness (beta) based on distance
                        beta = self.beta0 * np.exp(-self.gamma * (r ** 2))
                        
                        # Calculate random walk vector
                        rand_walk = self.alpha * (np.random.rand(self.dim) - 0.5)
                        
                        # Move firefly i towards firefly j
                        pop[i] = pop[i] + beta * (pop[j] - pop[i]) + rand_walk
                        
                        # Boundary Control: keep them inside the search space
                        pop[i] = np.clip(pop[i], self.bounds[0], self.bounds[1])
            
            # Gradually reduce the randomness (alpha) to improve exploitation over time
            self.alpha *= self.alpha_damp
            
            # Evaluate new fitness for the entire population
            fitness = np.array([self.func(ind) for ind in pop])
            
            # Update global best
            current_best_fit = np.min(fitness)
            if current_best_fit < gbest_fit:
                gbest_fit = current_best_fit
                gbest_pos = pop[np.argmin(fitness)].copy()
                
        return gbest_fit, gbest_pos