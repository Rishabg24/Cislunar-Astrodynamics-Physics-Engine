import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from astropy.constants import M_sun, M_earth, M_jup 


def returnMU(primary: str = 'earth', secondary: str = 'moon') -> float:
    """Return the reduced mass parameter mu = m2/(m1+m2).

    primary and secondary are lowercase text names like 'earth', 'moon', 'sun',
    'mars', 'venus', 'mercury', 'jupiter', 'saturn', 'uranus', 'neptune'.
    """

    # derived body masses (approximate where needed) using astropy constants
    M_moon = M_earth * 0.0123
    M_mars = M_earth * 0.107
    M_venus = M_earth * 0.815
    M_mercury = M_earth * 0.055
    M_jupiter = M_jup
    M_saturn = M_jup * 0.3
    M_uranus = M_jup * 0.0457
    M_neptune = M_jup * 0.0543

    def name_to_mass(name: str):
        match name.lower():
            case 'sun':
                return M_sun
            case 'earth':
                return M_earth
            case 'moon':
                return M_moon
            case 'mars':
                return M_mars
            case 'venus':
                return M_venus
            case 'mercury':
                return M_mercury
            case 'jupiter' | 'jup':
                return M_jupiter
            case 'saturn':
                return M_saturn
            case 'uranus':
                return M_uranus
            case 'neptune':
                return M_neptune
            case _:
                raise ValueError(f"Unknown body name: {name}")

    m1 = name_to_mass(primary)
    m2 = name_to_mass(secondary)

    return m2 / (m1 + m2)

MU = returnMU()

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

    return [vx, vy, vz, ax,ay,az]


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


