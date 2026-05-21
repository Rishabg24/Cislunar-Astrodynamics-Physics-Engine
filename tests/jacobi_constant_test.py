import numpy as np
import matplotlib.pyplot as plt
from math_engine.cr3bp import cr3bp_eom, solveSystem, MU
from scipy.integrate import solve_ivp 

# --- Initial Conditions (a simple orbit near L1) ---
# These are approximate non-dimensional coordinates

state0 = [0.8369, 0.0, 0.0,   # x, y, z
          0.0,    0.1, 0.0]   # vx, vy, vz

t_span = (0, 3.0) # Time span for integration (non-dimensional)
t_eval = np.linspace(*t_span, 10000)


solve = solveSystem(state0, t_span, t_eval)


def jacobi_constant(state, mu):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2     + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    v_sq = vx**2 + vy**2 + vz**2
    omega = 0.5*(x**2 + y**2) + (1 - mu)/r1 + mu/r2

    return 2*omega - v_sq

# Check conservation over the trajectory
if __name__ == "__main__":
    C_values = [jacobi_constant(solve.y[:, i], MU) for i in range(len(solve.t))]
    print(f"C initial: {C_values[0]:.12f}")
    print(f"C final:   {C_values[-1]:.12f}")
    print(f"Drift:     {abs(C_values[-1] - C_values[0]):.2e}")
