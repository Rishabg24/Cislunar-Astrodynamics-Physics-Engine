"""
Main entry point for Cislunar Mission Design Engine.

This script demonstrates how to use the visualization functions
from the visuals module to plot various orbital mechanics phenomena
in the Circular Restricted Three-Body Problem (CR3BP).
"""

import numpy as np
from visuals.plots import (
    plot_cr3bp_trajectory,
    plot_3d_cr3bp_trajectory,
    plot_lagrange_points,
    plot_zero_velocity_curves,
    plot_stable_manifolds,
    plot_all_visualizations,
)


def main():
    """
    Run example visualizations of CR3BP dynamics.
    """
    # Example initial conditions
    state0 = [0.8369, 0.0, 0.0, 0.0, 0.1, 0.0]  # [x, y, z, vx, vy, vz]
    t_span = (0, 3.0)
    t_eval = np.linspace(*t_span, 10000)
    
    # Uncomment to generate individual visualizations:
    
    # # 1. Plot CR3BP trajectory (2D)
    # print("Plotting CR3BP trajectory (2D)...")
    # plot_cr3bp_trajectory(state0, t_span, t_eval)
    
    # # 2. Plot CR3BP trajectory (3D)
    # print("Plotting CR3BP trajectory (3D)...")
    # plot_3d_cr3bp_trajectory(state0, t_span, t_eval)
    
    # # 3. Plot Lagrange points
    # print("Plotting Lagrange points...")
    # plot_lagrange_points()
    
    # # 4. Plot zero velocity curves
    # print("Plotting zero velocity curves...")
    # plot_zero_velocity_curves(state0)
    
    # # 5. Plot stable manifolds for specific Lagrange point (L1)
    # print("Plotting stable manifolds for L1...")
    # plot_stable_manifolds(lagrange_point_index=0)
    
    # # 6. Plot all visualizations
    print("Generating all visualizations...")
    plot_all_visualizations(state0, t_span, t_eval)


if __name__ == "__main__":
    main()
