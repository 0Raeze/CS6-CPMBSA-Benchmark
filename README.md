# CS6-CPMBSA-Benchmark
CS6 - Data Structures and Algorithms Final Project

This repository contains the modules to benchmark the Combined Power Mutation-Based Backtracking Search Algorithm (CPMBSA) against other standard metaheuristic algorithms.

Project Structure
- `functions.py`: Contains the 5 benchmark functions following a 2-2-1 ratio (2 Unimodal, 2 Multimodal, 1 Fixed-Dimension).
- `pso.py`: Particle Swarm Optimization (Swarm-based)
- `rcga.py`: Real-Coded Genetic Algorithm (Evolutionary-based)
- `fa.py`: Firefly Algorithm (Bio/Physics-based)

Instructions (Integration Guide)
To hook these up to your `main.py` menu and the `CPMBSA` class:

1. **Functions:** Import the `FUNCTION_SUITE` dictionary from `functions.py`. It contains the function, bounds, and required dimensions (e.g., dim=2 for the Camel function, dim=40 for the rest).
2. **Algorithms:** Every algorithm class I built takes the exact same arguments upon initialization:
   `Algorithm(func, bounds, dim, pop_size=30, max_evals=9000)`
3. **Execution:** To run the algorithm, simply call the `.optimize()` method. It will always return exactly two values:
   `best_fitness, best_position = algo.optimize()`

Make sure your `StandardBSA` and `CPMBSA` classes also have an `.optimize()` method that returns the exact same two variables so our `main.py` menu can switch between them easily.
