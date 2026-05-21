# Cislunar Mission Design Engine

A Python toolkit for exploring and visualizing dynamics in the Earth–Moon Circular Restricted Three-Body Problem (CR3BP). This project computes trajectories, Lagrange points, zero velocity curves, and stable manifolds for mission design and astrodynamics study.

## Features

- CR3BP equations of motion integration
- Visualization of 2D and 3D spacecraft trajectories in the rotating frame
- Computation and plotting of Earth–Moon Lagrange points (L1–L5)
- Zero velocity curve generation using Jacobi constant contours
- Stable manifold computation and plotting for collinear Lagrange points
- Example scripts demonstrating visualization workflows

## Repository Structure

- `main.py` - Primary entry point for generating example visualizations
- `math_engine/` - Core dynamics and mathematical utilities
  - `cr3bp.py` - CR3BP equations of motion and numerical integration
  - `lagrange_points.py` - Lagrange point computation via root finding
  - `manifolds.py` - Stable manifold computation for Lagrange points
  - `zero_velocity.py` - Zero velocity curve computation from Jacobi constant
- `visuals/` - Plotting and visualization utilities
  - `plots.py` - Plot functions for trajectories, Lagrange points, zero velocity curves, and manifolds
- `tests/` - Project tests
  - `jacobi_constant_test.py` - Jacobi constant calculation and verification

## Installation

Recommended: use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, install dependencies manually:

```bash
pip install numpy scipy matplotlib
```

## Usage

Run the main visualization script:

```bash
python main.py
```

By default, `main.py` calls `plot_all_visualizations()` and generates:

- CR3BP 2D trajectory
- Lagrange point plot
- Zero velocity curve plot
- Stable manifold plots for L1, L2, and L3

### Custom visualizations

Edit `main.py` to enable or modify the example plot calls, or use the plotting API directly:

```python
from visuals.plots import plot_cr3bp_trajectory, plot_lagrange_points, plot_zero_velocity_curves, plot_stable_manifolds
import numpy as np

state0 = [0.8369, 0.0, 0.0, 0.0, 0.1, 0.0]
t_span = (0, 3.0)
t_eval = np.linspace(*t_span, 10000)

plot_cr3bp_trajectory(state0, t_span, t_eval)
plot_lagrange_points()
plot_zero_velocity_curves(state0)
plot_stable_manifolds(lagrange_point_index=0)
```

## Testing

Run tests with `pytest` or the built-in Python module:

```bash
pytest
```

or

```bash
python -m pytest
```

## Notes

- The project uses non-dimensional normalized units for the CR3BP model.
- The Earth–Moon mass ratio is set in `math_engine/cr3bp.py` as `MU = 0.01215`.
- Stable manifold computation is currently based on linearized dynamics around the collinear Lagrange points.
