import numpy as np

class RCGA:
    def __init__(self, func, bounds, dim, pop_size=30, max_evals=9000):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_gens = max_evals // pop_size # Convert evaluations to generations
        
        # RCGA Specific Parameters
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.tournament_size = 3
        
    def _tournament_selection(self, population, fitness):
        """Selects the best parent from a small random sample of the population."""
        selected_indices = np.random.choice(self.pop_size, self.tournament_size, replace=False)
        best_idx = selected_indices[np.argmin(fitness[selected_indices])]
        return population[best_idx]

    def optimize(self):
        # 1. Initialize population randomly within bounds
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([self.func(ind) for ind in pop])
        
        # Track the absolute best global solution
        gbest_pos = pop[np.argmin(fitness)].copy()
        gbest_fit = np.min(fitness)
        
        # Main Generation Loop
        for gen in range(self.max_gens):
            new_pop = np.zeros((self.pop_size, self.dim))
            
            # Generate new population
            for i in range(0, self.pop_size, 2):
                # Selection
                parent1 = self._tournament_selection(pop, fitness)
                parent2 = self._tournament_selection(pop, fitness)
                
                # Crossover (Arithmetic Blending)
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.rand(self.dim) # Random blend factor for each dimension
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation (Gaussian Noise)
                for child in (child1, child2):
                    if np.random.rand() < self.mutation_rate:
                        # Add random Gaussian noise. The scale shrinks slightly over time for fine-tuning.
                        sigma = (self.bounds[1] - self.bounds[0]) * 0.1
                        noise = np.random.normal(0, sigma, self.dim)
                        child += noise
                
                # Assign to new population (Ensure we don't go out of bounds if pop_size is odd)
                new_pop[i] = child1
                if i + 1 < self.pop_size:
                    new_pop[i + 1] = child2
                    
            # Boundary Control
            new_pop = np.clip(new_pop, self.bounds[0], self.bounds[1])
            
            # Evaluate new population
            new_fitness = np.array([self.func(ind) for ind in new_pop])
            
            # Elitism: Carry over the best individual from the previous generation
            best_old_idx = np.argmin(fitness)
            worst_new_idx = np.argmax(new_fitness)
            new_pop[worst_new_idx] = pop[best_old_idx]
            new_fitness[worst_new_idx] = fitness[best_old_idx]
            
            # Update population and global best
            pop = new_pop
            fitness = new_fitness
            
            current_best_fit = np.min(fitness)
            if current_best_fit < gbest_fit:
                gbest_fit = current_best_fit
                gbest_pos = pop[np.argmin(fitness)].copy()
                
        return gbest_fit, gbest_pos