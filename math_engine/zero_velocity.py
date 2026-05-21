import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from tests.jacobi_constant_test import jacobi_constant
from math_engine.cr3bp import returnMU
from math_engine.lagrange_points import lagrangePoints, partial_effective_potential


MU = returnMU()

def compute_zero_velocity_curves(initial_conditions):
    C = jacobi_constant(initial_conditions, MU)  # Using Earth-Moon mass ratio

    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)

    r1 = np.sqrt((X + MU) ** 2 + Y**2)
    r2 = np.sqrt((X - 1 + MU) ** 2 + Y**2)

    Z = 2 *  (0.5*(X**2 + Y**2) + (1-MU)/r1 + MU/r2) - C

    return X, Y, Z, C


lagrange_points = lagrangePoints(partial_effective_potential)

if __name__ == "__main__":
    initial_conditions = [0.84, 0.0, 0.0, 0.0, 0.4, 0.0]  # Example state vector

    X, Y, Z, C = compute_zero_velocity_curves(initial_conditions)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels = [0], cmap="viridis")
    if Z.min() < 0:
     plt.contourf(X, Y, Z, levels=[Z.min(), 0], cmap="viridis", alpha=0.5)
    plt.plot(-MU, 0, "bo", markersize=10, label="Earth")  # Earth position
    plt.plot(1 - MU, 0, "go", markersize=6, label="Moon")  # Moon position
    plt.plot(lagrange_points[0], 0, "ro", markersize=8, label="L1")
    plt.plot(lagrange_points[1], 0, "mo", markersize=8, label="L2")
    plt.plot(lagrange_points[2], 0, "co", markersize=8, label="L3")
    plt.plot(
        lagrange_points[3][0], lagrange_points[3][1], "yo", markersize=8, label="L4"
    )
    plt.plot(
        lagrange_points[4][0], lagrange_points[4][1], "ko", markersize=8, label="L5"
    )
    plt.xlabel("x (non-dim)")
    plt.ylabel("y (non-dim)")
    plt.title("Zero Velocity Curves in the CR3BP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
