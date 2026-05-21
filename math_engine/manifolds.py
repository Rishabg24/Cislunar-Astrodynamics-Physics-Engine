import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from math_engine.cr3bp import returnMU, solveSystem
from math_engine.lagrange_points import (
    lagrangePoints,
    returnR1R2,
    partial_effective_potential,
)
from tests.jacobi_constant_test import jacobi_constant

MU = returnMU()
X_l1 = lagrangePoints(partial_effective_potential)[0]  # L1 point
state = [X_l1, 0, 0, 0, 0, 0]  # State vector at L1 (x, y, z, vx, vy, vz)
r1, r2 = returnR1R2(X_l1)


partial_omega_xx = (
    1
    - (1 - MU) * (1 / r1**3 - 3 * (r1**2) / r1**5)
    - MU * (1 / r2**3 - 3 * (r2**2) / r2**5)
)
partial_omega_yy = (
    1
    - (1 - MU) * (1 / r1**3 - 3 * (0**2) / r1**5)
    - MU * (1 / r2**3 - 3 * (0**2) / r2**5)
)
partial_omega_xy = 0 - (1 - MU) * (-3 * (r1 * 0) / r1**5) - MU * (-3 * (r2 * 0) / r2**5)
partial_omega_yx = partial_omega_xy

A = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [partial_omega_xx, partial_omega_xy, 0, 2],
        [partial_omega_yx, partial_omega_yy, -2, 0],
    ]
)


def compute_manifold_eigenvalues(monodromy_matrix):
    """
    Compute stable eigenvalues and eigenvectors from monodromy matrix.
    
    Parameters
    ----------
    monodromy_matrix : np.ndarray
        4x4 monodromy matrix
    
    Returns
    -------
    tuple
        (dominant_stable_eigenvalue, stable_eigenvector)
    """
    eigenvalues, eigenvectors = np.linalg.eig(monodromy_matrix)
    # strip small imaginary parts due to numerical errors and only keep largest negative real part

    real_eigenvalues_indicies = [i for i, ev in enumerate(eigenvalues) if np.abs(ev.imag) < 1e-10]

    if not real_eigenvalues_indicies:
        return None, None

    min_real_eigenvalue_index = min(real_eigenvalues_indicies, key=lambda i: eigenvalues[i].real)
   
    stable_eigenvalues = eigenvalues[min_real_eigenvalue_index]
    stable_eigenvectors = eigenvectors[:, min_real_eigenvalue_index]
    
    return stable_eigenvalues, stable_eigenvectors


def compute_stable_manifolds(lagrange_point_index=0, perturbation_magnitude=1e-6, num_periods=15, num_points=10000):
    """
    Compute stable manifold trajectories for a given Lagrange point.
    
    Parameters
    ----------
    lagrange_point_index : int
        Index of Lagrange point (0=L1, 1=L2, 2=L3)
    perturbation_magnitude : float
        Magnitude of perturbation along eigenvector
    num_periods : int
        Number of periods to integrate backward in time
    num_points : int
        Number of evaluation points
    
    Returns
    -------
    dict
        Contains: x_lp, state, dominant_stable_ev, positive_trajectory, negative_trajectory
    """
    lagrange_points = lagrangePoints(partial_effective_potential)
    X_lp = lagrange_points[lagrange_point_index]
    state = [X_lp, 0, 0, 0, 0, 0]
    r1, r2 = returnR1R2(X_lp)
    
    # Compute monodromy matrix for this Lagrange point
    partial_omega_xx = (
        1
        - (1 - MU) * (1 / r1**3 - 3 * (X_lp + MU)**2 / r1**5)
        - MU * (1 / r2**3 - 3 * (X_lp - 1 + MU)**2 / r2**5)
    )
    partial_omega_yy = (
        1
        - (1 - MU) * (1 / r1**3)
        - MU * (1 / r2**3)
    )
    partial_omega_xy = 0
    partial_omega_yx = 0
    
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [partial_omega_xx, partial_omega_xy, 0, 2],
            [partial_omega_yx, partial_omega_yy, -2, 0],
        ]
    )
    
    dominant_stable_ev, stable_eigenvectors = compute_manifold_eigenvalues(A)
    
    if dominant_stable_ev is None or stable_eigenvectors is None:
        return None
    
    # Extend eigenvector to 6D
    eigvec_4d = stable_eigenvectors.real
    eigvec_6d = np.array([eigvec_4d[0], eigvec_4d[1], 0, eigvec_4d[2], eigvec_4d[3], 0])
    
    perturbed_positive_state = state + perturbation_magnitude * eigvec_6d
    perturbed_negative_state = state - perturbation_magnitude * eigvec_6d
    
    t_span = (0, -num_periods * np.pi / np.abs(dominant_stable_ev))
    t_eval = np.linspace(*t_span, num_points)
    
    positive_trajectory = solveSystem(perturbed_positive_state, t_span, t_eval).y
    negative_trajectory = solveSystem(perturbed_negative_state, t_span, t_eval).y
    
    return {
        'x_lp': X_lp,
        'state': state,
        'dominant_stable_ev': dominant_stable_ev,
        'positive_trajectory': positive_trajectory,
        'negative_trajectory': negative_trajectory,
    }


# --- For backward compatibility, compute manifolds for L1 by default ---
dominant_stable_ev, stable_eigenvectors = compute_manifold_eigenvalues(A)

if dominant_stable_ev is not None and stable_eigenvectors is not None:
    eigvec_4d = stable_eigenvectors.real
    eigvec_6d = np.array([eigvec_4d[0], eigvec_4d[1], 0, eigvec_4d[2], eigvec_4d[3], 0])

    perturbation_magnitude = 1e-6

    perturbed_positive_state = (
        state + perturbation_magnitude * eigvec_6d
    )
    perturbed_negative_state = (
        state - perturbation_magnitude * eigvec_6d
    )

    t_span = (
        0,
        -15 * np.pi / np.abs(dominant_stable_ev),
    )
    t_eval = np.linspace(*t_span, 10000)
    positive_trajectory = solveSystem(perturbed_positive_state, t_span, t_eval).y
    negative_trajectory = solveSystem(perturbed_negative_state, t_span, t_eval).y


print(f"Dominant Stable Eigenvalue: {dominant_stable_ev}")
print(f"Stable Eigenvectors:\n{stable_eigenvectors}")


if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    if dominant_stable_ev is not None and stable_eigenvectors.all():
        plt.plot(
            positive_trajectory[0],
            positive_trajectory[1],
            lw=0.8,
            color="steelblue",
            label="Stable Manifold (Positive Perturbation)",
        )
        plt.plot(
            negative_trajectory[0],
            negative_trajectory[1],
            lw=0.8,
            color="coral",
            label="Stable Manifold (Negative Perturbation)",
        )
    plt.plot(-MU, 0, "bo", markersize=10, label="Earth")  # Earth position
    plt.plot(1 - MU, 0, "go", markersize=6, label="Moon")  # Moon position
    plt.plot(X_l1, 0, "ro", markersize=8, label="L1 Point")
    plt.xlabel("x (non-dim)")
    plt.ylabel("y (non-dim)")
    plt.title("Stable Manifolds of L1 in the CR3BP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
