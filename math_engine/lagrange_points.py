import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.optimize import newton
from math_engine.cr3bp import returnMU
from tests.jacobi_constant_test import jacobi_constant
import numpy as np
import matplotlib.pyplot as plt

MU = returnMU()
r1 = lambda x: np.abs(x+MU)
r2 = lambda x: np.abs(x-1+MU)

def returnR1R2(x):
    return r1(x), r2(x)

effective_potential = lambda x, y: 0.5*(x**2 + y**2) + (1-MU)/r1(x) + MU/r2(x)
partial_effective_potential = lambda x: x - (1-MU)*(x+MU)/r1(x)**3 - MU*(x-1+MU)/r2(x)**3

def lagrangePoints(partial_func):
    collinear_guess = [0.5, 1.5, -1.0] # Initial guesses for L1, L2, L3
    equilateral_points = [(0.5 - MU, np.sqrt(3)/2), (0.5 - MU, -np.sqrt(3)/2)] # L4 and L5
    lagrange_points = []
    for guess in collinear_guess:
        point = newton(partial_func, guess)
        lagrange_points.append(point)
    lagrange_points.extend(equilateral_points)
    return lagrange_points


if __name__ == "__main__":
    lagrange_points = lagrangePoints(partial_effective_potential)
    print("Lagrange Points (x, y):")
    for i, point in enumerate(lagrange_points):
        print(f"L{i+1}: {point}")

    jacobi_values = jacobi_constant([lagrange_points[0], 0, 0, 0, 0, 0], MU)
    print(f"Jacobi Constant at L1: {jacobi_values:.12f}")   
    jacobi_values = jacobi_constant([lagrange_points[1], 0, 0, 0, 0, 0], MU)
    print(f"Jacobi Constant at L2: {jacobi_values:.12f}")
    jacobi_values = jacobi_constant([lagrange_points[2], 0, 0, 0, 0, 0], MU)
    print(f"Jacobi  Constant at L3: {jacobi_values:.12f}")
    jacobi_values = jacobi_constant([lagrange_points[3][0], lagrange_points[3][1], 0, 0, 0, 0], MU)
    print(f"Jacobi Constant at L4: {jacobi_values:.12f}")
    jacobi_values = jacobi_constant([lagrange_points[4][0], lagrange_points[4][1], 0, 0, 0, 0], MU)
    print(f"Jacobi Constant at L5: {jacobi_values:.12f}")
    

        
