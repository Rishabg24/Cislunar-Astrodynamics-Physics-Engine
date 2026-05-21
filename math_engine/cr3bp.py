import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MU = 0.01215 # Earth-Moon system mass ratio

def returnMU():
    return MU


def cr3bp_eom(t, state, mu):
    """
    Equations of Motion for the Circular Restricted Three-Body Problem.
    State vector: [x, y, z, vx, vy, vz]
    All units are non-dimensional.
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)

    ax = x + 2*vy - ((1-mu) * (x+mu) / (r1**3)) - (mu * (x-1+mu) / (r2**3))
    ay = y - 2*vx - ((1-mu)*y / (r1**3)) - (mu * y / (r2**3))
    az = -1 * ((1-mu)*z)/(r1**3) - (mu * z)/ (r2**3)

    return [vx, vy,vz, ax,ay,az]


# --- Initial Conditions (a simple orbit near L1) ---
# These are approximate non-dimensional coordinates

state0 = [0.8369, 0.0, 0.0,   # x, y, z
          0.0, 0.1, 0.0]   # vx, vy, vz

t_span = (0, 3.0) # Time span for integration (non-dimensional)
t_eval = np.linspace(*t_span, 10000)

def solveSystem(state0, t_span, t_eval):
    return solve_ivp(cr3bp_eom, t_span, state0, method='DOP853',args=(MU,), t_eval=t_eval, rtol=1e-12, atol = 1e-12)

if __name__ == "__main__":

    solve = solveSystem(state0, t_span, t_eval)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(solve.y[0], solve.y[1], lw=0.8, color='steelblue', label='Spacecraft')
    ax.plot(-MU,      0, 'bo', markersize=10, label='Earth')   # Earth position
    ax.plot(1 - MU,   0, 'go', markersize=6,  label='Moon')    # Moon position
    ax.set_xlabel('x (non-dim)')
    ax.set_ylabel('y (non-dim)')
    ax.set_title('CR3BP Trajectory — Rotating Frame')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


