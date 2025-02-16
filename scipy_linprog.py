import numpy as np
from scipy.optimize import linprog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_linear_program(c, A, b):
    """
    Solve linear programming problem using scipy.optimize.linprog

    Args:
        c: Objective function coefficients (maximize)
        A: Constraint coefficients matrix (<=)
        b: Right-hand side values

    Returns:
        Tuple containing optimal solution array and optimal value
    """
    try:
        # Convert inputs to numpy arrays
        c_arr = np.array(c, dtype=np.float64)
        A_arr = np.array(A, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)

        # Since linprog minimizes by default, but we want to maximize,
        # we multiply objective coefficients by -1
        result = linprog(
            c=-c_arr,  # Negative for maximization
            A_ub=A_arr,
            b_ub=b_arr,
            method='highs',  # Using the more modern HiGHS solver
            options={'disp': False}
        )

        if result.success:
            return result.x, -result.fun  # Negative fun for maximization
        else:
            raise ValueError(f"Optimization failed: {result.message}")

    except Exception as e:
        logger.error(f"Error solving linear program: {str(e)}")
        raise

def main():
    # Problem definition
    c = [12, 7, 10]  # Objective coefficients
    A = [
        [1, 1, 1],    # Labor constraint
        [30, 10, 0],  # Capital constraint
        [2, 1, 3]     # Resource constraint
    ]
    b = [300, 3000, 500]  # Right-hand side values

    try:
        solution, optimal_value = solve_linear_program(c, A, b)

        logger.info("最优解:")
        logger.info(f"农业={solution[0]:.1f}")
        logger.info(f"工业={solution[1]:.1f}")
        logger.info(f"服务业={solution[2]:.1f}")
        logger.info(f"最优值: {optimal_value:.1f}")

    except Exception as e:
        logger.error(f"求解失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
