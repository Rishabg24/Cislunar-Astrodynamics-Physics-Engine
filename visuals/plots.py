"""
Visualization module for Cislunar Mission Design Engine.

This module provides plotting functions that use the computation functions
from the math engine to visualize CR3BP dynamics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, List, Optional

from math_engine.cr3bp import returnMU, solveSystem
from math_engine.lagrange_points import (
    lagrangePoints,
    partial_effective_potential,
)
from math_engine.zero_velocity import compute_zero_velocity_curves
from math_engine.manifolds import compute_stable_manifolds

MU = returnMU()


# =============================================================================
# CR3BP TRAJECTORY PLOTTING
# =============================================================================

def plot_cr3bp_trajectory(
    state0: List[float],
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    title: str = "CR3BP Trajectory — Rotating Frame",
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
) -> Figure:
    """
    Plot a CR3BP trajectory in the rotating frame.
    
    Parameters
    ----------
    state0 : list
        Initial state vector [x, y, z, vx, vy, vz]
    t_span : tuple
        Time span (t_start, t_end) for integration
    t_eval : np.ndarray
        Evaluation times
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    solve = solveSystem(state0, t_span, t_eval)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(solve.y[0], solve.y[1], lw=0.8, color='steelblue', label='Spacecraft')
    ax.plot(-MU, 0, 'bo', markersize=10, label='Earth')
    ax.plot(1 - MU, 0, 'go', markersize=6, label='Moon')
    ax.set_xlabel('x (non-dim)')
    ax.set_ylabel('y (non-dim)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_3d_cr3bp_trajectory(
    state0: List[float],
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    title: str = "CR3BP Trajectory — 3D View",
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
) -> Figure:
    """
    Plot a CR3BP trajectory in 3D.
    
    Parameters
    ----------
    state0 : list
        Initial state vector [x, y, z, vx, vy, vz]
    t_span : tuple
        Time span (t_start, t_end) for integration
    t_eval : np.ndarray
        Evaluation times
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    solve = solveSystem(state0, t_span, t_eval)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solve.y[0], solve.y[1], solve.y[2], lw=0.8, color='steelblue', label='Spacecraft')
    ax.scatter([-MU], [0], [0], c='blue', s=100, label='Earth')
    ax.scatter([1 - MU], [0], [0], c='green', s=50, label='Moon')
    ax.set_xlabel('x (non-dim)')
    ax.set_ylabel('y (non-dim)')
    ax.set_zlabel('z (non-dim)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# LAGRANGE POINTS PLOTTING
# =============================================================================

def plot_lagrange_points(
    title: str = "Lagrange Points in the CR3BP",
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
) -> Figure:
    """
    Plot all Lagrange points and primary bodies.
    
    Parameters
    ----------
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    lagrange_points = lagrangePoints(partial_effective_potential)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot primary bodies
    ax.plot(-MU, 0, 'bo', markersize=12, label='Earth', zorder=5)
    ax.plot(1 - MU, 0, 'go', markersize=8, label='Moon', zorder=5)
    
    # Plot collinear Lagrange points
    colors = ['red', 'magenta', 'cyan']
    for i in range(3):
        ax.plot(lagrange_points[i], 0, 'o', color=colors[i], markersize=10, 
                label=f'L{i+1}', zorder=5)
    
    # Plot equilateral Lagrange points
    ax.plot(lagrange_points[3][0], lagrange_points[3][1], 'yo', markersize=10, 
            label='L4', zorder=5)
    ax.plot(lagrange_points[4][0], lagrange_points[4][1], 'ko', markersize=10, 
            label='L5', zorder=5)
    
    ax.set_xlabel('x (non-dim)')
    ax.set_ylabel('y (non-dim)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# ZERO VELOCITY CURVES PLOTTING
# =============================================================================

def plot_zero_velocity_curves(
    initial_conditions: List[float],
    title: str = "Zero Velocity Curves in the CR3BP",
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
    xlim: Tuple[float, float] = (-1.5, 1.5),
    ylim: Tuple[float, float] = (-1.5, 1.5),
) -> Figure:
    """
    Plot zero velocity curves for a given Jacobi constant.
    
    Parameters
    ----------
    initial_conditions : list
        State vector [x, y, z, vx, vy, vz] to compute Jacobi constant
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    lagrange_points = lagrangePoints(partial_effective_potential)
    X, Y, Z, C = compute_zero_velocity_curves(initial_conditions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot zero velocity curve
    ax.contour(X, Y, Z, levels=[0], cmap="viridis", linewidths=2)
    
    # Fill accessible region if it exists
    if Z.min() < 0:
        ax.contourf(X, Y, Z, levels=[Z.min(), 0], cmap="viridis", alpha=0.3)
    
    # Plot primary bodies
    ax.plot(-MU, 0, 'bo', markersize=12, label='Earth', zorder=5)
    ax.plot(1 - MU, 0, 'go', markersize=8, label='Moon', zorder=5)
    
    # Plot Lagrange points
    ax.plot(lagrange_points[0], 0, 'ro', markersize=10, label='L1', zorder=5)
    ax.plot(lagrange_points[1], 0, 'mo', markersize=10, label='L2', zorder=5)
    ax.plot(lagrange_points[2], 0, 'co', markersize=10, label='L3', zorder=5)
    ax.plot(lagrange_points[3][0], lagrange_points[3][1], 'yo', markersize=10, 
            label='L4', zorder=5)
    ax.plot(lagrange_points[4][0], lagrange_points[4][1], 'ko', markersize=10, 
            label='L5', zorder=5)
    
    ax.set_xlabel('x (non-dim)')
    ax.set_ylabel('y (non-dim)')
    ax.set_title(f"{title}\n(C = {C:.6f})")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# STABLE MANIFOLDS PLOTTING
# =============================================================================

def plot_stable_manifolds(
    lagrange_point_index: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
    xlim: Tuple[float, float] = (-1.5, 1.5),
    ylim: Tuple[float, float] = (-1.5, 1.5),
) -> Figure:
    """
    Plot stable manifolds for a specified Lagrange point.
    
    Parameters
    ----------
    lagrange_point_index : int, optional
        Index of Lagrange point (0=L1, 1=L2, 2=L3)
    title : str, optional
        Plot title. If None, auto-generated based on Lagrange point.
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    manifold_data = compute_stable_manifolds(lagrange_point_index)
    
    if manifold_data is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No real eigenvalues found", ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        manifold_data['positive_trajectory'][0],
        manifold_data['positive_trajectory'][1],
        lw=0.8,
        color="steelblue",
        label="Stable Manifold (Positive Perturbation)",
    )
    ax.plot(
        manifold_data['negative_trajectory'][0],
        manifold_data['negative_trajectory'][1],
        lw=0.8,
        color="coral",
        label="Stable Manifold (Negative Perturbation)",
    )
    
    ax.plot(-MU, 0, 'bo', markersize=10, label='Earth', zorder=5)
    ax.plot(1 - MU, 0, 'go', markersize=6, label='Moon', zorder=5)
    ax.plot(manifold_data['x_lp'], 0, 'ro', markersize=8, 
            label=f'L{lagrange_point_index + 1}', zorder=5)
    
    if title is None:
        title = f"Stable Manifolds of L{lagrange_point_index + 1} in the CR3BP"
    
    ax.set_xlabel('x (non-dim)')
    ax.set_ylabel('y (non-dim)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# COMPOSITE PLOTTING
# =============================================================================

def plot_all_visualizations(
    state0: Optional[List[float]] = None,
    t_span: Optional[Tuple[float, float]] = None,
    t_eval: Optional[np.ndarray] = None,
) -> None:
    """
    Generate and display all available visualizations.
    
    Parameters
    ----------
    state0 : list, optional
        Initial state for CR3BP trajectory. Default: [0.8369, 0, 0, 0, 0.1, 0]
    t_span : tuple, optional
        Time span for integration. Default: (0, 3.0)
    t_eval : np.ndarray, optional
        Evaluation times. Default: 10000 points
    """
    if state0 is None:
        state0 = [0.8369, 0.0, 0.0, 0.0, 0.1, 0.0]
    if t_span is None:
        t_span = (0, 3.0)
    if t_eval is None:
        t_eval = np.linspace(*t_span, 10000)
    
    print("Generating CR3BP trajectory visualization...")
    plot_cr3bp_trajectory(state0, t_span, t_eval)
    
    print("Generating Lagrange points visualization...")
    plot_lagrange_points()
    
    print("Generating zero velocity curves visualization...")
    plot_zero_velocity_curves(state0)
    
    print("Generating stable manifolds visualizations...")
    for i in range(3):  # L1, L2, L3
        try:
            plot_stable_manifolds(lagrange_point_index=i)
        except Exception as e:
            print(f"Error plotting manifold for L{i+1}: {e}")


if __name__ == "__main__":
    plot_all_visualizations()
