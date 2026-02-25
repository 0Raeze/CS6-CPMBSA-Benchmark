import numpy as np

class PSO:
    def __init__(self, func, bounds, dim, pop_size=30, max_evals=9000):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = max_evals // pop_size # Convert evals to iterations
        
        # PSO Specific Parameters
        self.w = 0.7   # Inertia (keeps them moving)
        self.c1 = 1.5  # Cognitive (memory of their own best spot)
        self.c2 = 1.5  # Social (pull towards the swarm's best spot)

    def optimize(self):
        # 1. Initialize positions and velocities
        pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        vel = np.zeros((self.pop_size, self.dim))
        
        # Track personal bests and global best
        pbest_pos = pos.copy()
        pbest_fit = np.array([self.func(p) for p in pos])
        gbest_pos = pbest_pos[np.argmin(pbest_fit)]
        gbest_fit = np.min(pbest_fit)
        
        # Main Loop
        for _ in range(self.max_iters):
            r1, r2 = np.random.rand(2)
            
            # Update Velocity and Position
            vel = (self.w * vel) + (self.c1 * r1 * (pbest_pos - pos)) + (self.c2 * r2 * (gbest_pos - pos))
            pos = pos + vel
            
            # Keep within bounds
            pos = np.clip(pos, self.bounds[0], self.bounds[1])
            
            # Update bests (Fitness evaluation)
            current_fit = np.array([self.func(p) for p in pos])
            
            # Update personal bests
            better_mask = current_fit < pbest_fit
            pbest_pos[better_mask] = pos[better_mask]
            pbest_fit[better_mask] = current_fit[better_mask]
            
            # Update global best
            if np.min(pbest_fit) < gbest_fit:
                gbest_fit = np.min(pbest_fit)
                gbest_pos = pbest_pos[np.argmin(pbest_fit)]
                
        return gbest_fit, gbest_pos