import numpy as np

class BenchmarkFunctions:

    # SET-I: UNIMODAL FUNCTIONS (Exploitation)
    @staticmethod
    def sphere(x):
        """F1: The classic smooth bowl. Pure exploitation test."""
        return np.sum(x**2)

    @staticmethod
    def rosenbrock(x):
        """F2: The 'Banana Valley'. Hard to find the exact flat bottom."""
        # Using numpy slicing for efficient vectorized calculation
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

    # SET-II: MULTIMODAL FUNCTIONS (Exploration)
    @staticmethod
    def rastrigin(x):
        """F24 (Example): The 'Egg Carton' with many fake minimums."""
        D = len(x)
        return 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    @staticmethod
    def ackley(x):
        """F26 (Example): Flat planes with a deep, sudden hole in the center."""
        D = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / D))
        term2 = -np.exp(sum_cos / D)
        
        return term1 + term2 + 20 + np.e

    # SET-III: FIXED-DIMENSION MULTIMODAL (Stability/Balance)
    @staticmethod
    def six_hump_camel(x):
        """F52 (Example): Highly complex 2D landscape with 6 local minimums."""
        # This function strictly requires a 2D input (x[0] and x[1])
        x1, x2 = x[0], x[1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return term1 + term2 + term3


# SUITE CONFIGURATION FOR THE MENU (Used in main.py)
FUNCTION_SUITE = {
    # 2 Unimodal Functions
    "F1_Sphere": {
        "func": BenchmarkFunctions.sphere,
        "bounds": [-100, 100],
        "dim": 40,  # Standard paper dimension
        "type": "Set-I: Unimodal (Exploitation)"
    },
    "F2_Rosenbrock": {
        "func": BenchmarkFunctions.rosenbrock,
        "bounds": [-30, 30],
        "dim": 40,
        "type": "Set-I: Unimodal (Exploitation)"
    },
    
    # 2 Multimodal Functions
    "F24_Rastrigin": {
        "func": BenchmarkFunctions.rastrigin,
        "bounds": [-5.12, 5.12],
        "dim": 40,
        "type": "Set-II: Multimodal (Exploration)"
    },
    "F26_Ackley": {
        "func": BenchmarkFunctions.ackley,
        "bounds": [-32, 32],
        "dim": 40,
        "type": "Set-II: Multimodal (Exploration)"
    },
    
    # 1 Fixed-Dimension Function
    "F52_SixHumpCamel": {
        "func": BenchmarkFunctions.six_hump_camel,
        "bounds": [-5, 5],
        "dim": 2,  # Strict dimension limit for Fixed-Dimension Set
        "type": "Set-III: Fixed-Dimension (Stability)"
    }
}