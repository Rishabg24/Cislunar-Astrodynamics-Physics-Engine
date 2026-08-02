import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

from math_engine.cr3bp import returnMU, solveSystem, cr3bp_eom
from math_engine.lagrange_points import (
    lagrangePoints,
    returnR1R2,
    partial_effective_potential,
)

from math_engine.richardson import richardson_halo_seed

from tests.jacobi_constant_test import jacobi_constant

from scipy.integrate import solve_ivp
from scipy.linalg import eig

MU = returnMU()
X_l_point = lagrangePoints(partial_effective_potential)[0]  # [L1, L2, L3, L4, L5]
Xl_isTuple = False

print("================> ", X_l_point)

if isinstance(X_l_point, tuple):
    Xl_isTuple = True

    x_cord = X_l_point[0]
    y_cord = X_l_point[1]
    X_l_point = x_cord


r1 = lambda x, y, z, mu: ((x + mu) ** 2 + y**2 + z**2) ** 0.5
r2 = lambda x, y, z, mu: ((x - 1 + mu) ** 2 + y**2 + z**2) ** 0.5


partial_omega_xx = lambda x, y, z, mu: (
    1
    - (1 - mu) / r1(x, y, z, mu) ** 3
    + 3 * (1 - mu) * (x + mu) ** 2 / r1(x, y, z, mu) ** 5
    - mu / r2(x, y, z) ** 3
    + 3 * mu * (x - 1 + mu) ** 2 / r2(x, y, z, mu) ** 5
)

partial_omega_yy = lambda x, y, z, mu: (
    1
    - (1 - mu) / r1(x, y, z, mu) ** 3
    + 3 * (1 - mu) * y**2 / r1(x, y, z, mu) ** 5
    - mu / r2(x, y, z, mu) ** 3
    + 3 * mu * y**2 / r2(x, y, z, mu) ** 5
)

partial_omega_zz = (
    lambda x, y, z, mu: (-1 * (1 - mu) / r1(x, y, z, mu) ** 3)
    + 3 * (1 - mu) * z**2 / r1(x, y, z, mu) ** 5
    - (mu / r2(x, y, z, mu) ** 3)
    + 3 * mu * z**2 / r2(x, y, z, mu) ** 5
)


partial_omega_xy = (
    lambda x, y, z, mu: (3 * (1 - mu) * (x + mu) * y) / r1(x, y, z, mu) ** 5
    + 3 * mu * (x - 1 + mu) * y / r2(x, y, z, mu) ** 5
)
partial_omega_yx = partial_omega_xy

partial_omega_xz = (
    lambda x, y, z, mu: (3 * (1 - mu) * (x + mu) * z) / r1(x, y, z, mu) ** 5
    + 3 * mu * (x - 1 + mu) * z / r2(x, y, z, mu) ** 5
)
partial_omega_zx = partial_omega_xz

partial_omega_yz = (
    lambda x, y, z, mu: (3 * (1 - mu) * y * z) / r1(x, y, z, mu) ** 5
    + 3 * mu * y * z / r2(x, y, z, mu) ** 5
)
partial_omega_zy = partial_omega_yz


def compute_A_matrix(state, mu):
    """Return the 6x6 Jacobian A = df/dX evaluated at state."""

    x = state[0]
    y = state[1]
    z = state[2]

    # Compute the Jacobian matrix of the system at the given state
    partial_omega_xx_val = partial_omega_xx(x, y, z, mu)
    partial_omega_yy_val = partial_omega_yy(x, y, z, mu)
    partial_omega_zz_val = partial_omega_zz(x, y, z, mu)
    partial_omega_xy_val = partial_omega_xy(x, y, z, mu)
    partial_omega_xz_val = partial_omega_xz(x, y, z, mu)
    partial_omega_yz_val = partial_omega_yz(x, y, z, mu)
    partial_omega_yx_val = partial_omega_xy_val
    partial_omega_zx_val = partial_omega_xz_val
    partial_omega_zy_val = partial_omega_yz_val

    jacobian_matrix = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [
                partial_omega_xx_val,
                partial_omega_xy_val,
                partial_omega_xz_val,
                0,
                2,
                0,
            ],
            [
                partial_omega_yx_val,
                partial_omega_yy_val,
                partial_omega_yz_val,
                -2,
                0,
                0,
            ],
            [
                partial_omega_zx_val,
                partial_omega_zy_val,
                partial_omega_zz_val,
                0,
                0,
                0,
            ],
        ]
    )

    return jacobian_matrix


def ode_function(t, state42, mu):
    state6 = state42[:6]
    state_dot = cr3bp_eom(t, state6, mu)

    state36 = np.reshape(state42[6:], (6, 6))

    A = compute_A_matrix(state6, mu)

    phi_dot = A @ state36

    return np.concatenate([state_dot, phi_dot.flatten()])


def integrator(
    timespan, state6, events, mu, dense_output=False, rtol=1e-12, atol=1e-12
):
    full_state = np.concatenate([state6, np.eye(6).flatten()])
    solution = solve_ivp(
        ode_function,
        timespan,
        full_state,
        method="DOP853",
        dense_output=dense_output,
        events=events,
        args=(mu,),
        rtol=rtol,
        atol=atol,
    )
    return solution


def differential_corrector(
    T,
    x0_guess,
    z0_guess,
    vy0_guess, 
    mu,
    method="3D",
    max_newton=50,
    max_expand=6,
    tol=1e-12,
):
    """
    Newtonian Differential Corrector with separated crossing-search
    and Newton-correction loops.
    """
    if method != "2D" and method != "3D":
        raise ValueError("Differential Corrector Method Should Either be 2D or 3D")

    def event_function(t, y, mu):
        return y[1]

    event_function.terminal = False
    event_function.direction = 0

    t_span_bound = 1.5 * T  # starting estimate, from linear period

    alpha = 0.3

    for newton_iter in range(max_newton):
        state6 = [x0_guess, 0, z0_guess, 0, vy0_guess, 0]

        # --- INNER LOOP: find a valid half-period crossing ---
        t_events = np.array([])
        for expand_iter in range(max_expand):
            t_span = (0, t_span_bound)
            solution = integrator(
                t_span,
                state6,
                events=event_function,
                mu=mu,
                dense_output=True,
                rtol=tol,
                atol=tol,
            )
            t_events = solution.t_events[0]
            t_events = t_events[t_events > 1e-6]

            if len(t_events) > 0:
                break  # found it, drop out of expansion loop

            t_span_bound *= 1.5  # no touching vy0_guess here
        else:
            # inner loop exhausted max_expand without ever breaking
            raise RuntimeError(
                f"No y=0 crossing found after expanding t_span to "
                f"{t_span_bound:.4f} — check direction convention or "
                f"initial eigenvector guess"
            )

        # --- past this point, t_events is guaranteed non-empty ---
        t_half = t_events[0]
        sol = solution.sol(t_half)

        vx_f = sol[3]
        vy_f = sol[4]
        vz_f = sol[5]
        phi = sol[6:].reshape((6, 6))

        G = np.transpose(np.array([vx_f, vz_f]))
        P = np.array([x0_guess, vy0_guess])

        if method == "2D":
            if np.abs(vx_f) < tol:
                T_converged = t_half * 2
                print("2D Correction Converged. Integration Finished")
                return state6, T_converged

        elif method == "3D":
            if np.linalg.norm(G) < tol:
                T_converged = t_half * 2
                print("3D differential Correction Converged. Integration Finished")
                return state6, T_converged
        else:
            raise ValueError("Method Error")

        if method == "2D":

            # --- Newton correction step 2D ---
            ax_f = cr3bp_eom(t_half, sol[:6], mu)[3]

            correction_derivative = phi[3, 4] - (ax_f / vy_f) * phi[1, 4]

            vy0_guess -= alpha * vx_f / correction_derivative

            print(
                f"[newton {newton_iter}] vy0={vy0_guess:.10f}, "
                f"vx_final={vx_f:.2e}, t_half={t_half:.6f}"
            )

        elif method == "3D":
            # --- Newton correction step 3D---
            ax_f, az_f = (
                cr3bp_eom(t_half, sol[:6], mu)[3],
                cr3bp_eom(t_half, sol[:6], mu)[5],
            )

            phi_34 = phi[3, 4]
            phi_30 = phi[3, 0]
            phi_14 = phi[1, 4]
            phi_10 = phi[1, 0]
            phi_54 = phi[5, 4]
            phi_50 = phi[5, 0]

            partial_G = np.array([[phi_30, phi_34], [phi_50, phi_54]])
            dth_dp = np.array([phi_10, phi_14])
            accel_matrix = np.array([ax_f, az_f])

            D = partial_G - (1 / vy_f) * np.outer(accel_matrix, dth_dp)

            delta_p = np.linalg.solve(D, G)
            x0_guess -= alpha * delta_p[0]
            vy0_guess -= alpha * delta_p[1]

        else:
            raise ValueError("Method Error")
    else:
        if method == "2D":
            raise RuntimeError(
                f"[Error] Newton correction did not converge after {max_newton} iterations "
                f"(vx_final={vx_f:.2e} at last attempt)"
            )
        elif method == "3D":
            raise RuntimeError(
                f"[Error] Newton correction did not converge after {max_newton} iterations "
                f"([v_x, v_z]={vx_f:.2e}, {vx_f:.2e} at last attempt)"
            )


def calc_init_conditions(mu, method="3D"):

    y = y_cord if Xl_isTuple else 0

    state = [X_l_point, y, 0, 0, 0, 0]

    STM = compute_A_matrix(state, mu)

    if method == "2D":
        eigenvalues, eigenvectors = eig(STM)

        omega = None  # <-- will hold the scalar frequency from the eigenvalue

        for i, val in enumerate(eigenvalues):
            if np.iscomplex(val) and np.imag(val) > 0:
                complex_vector = eigenvectors[:, i]
                if abs(complex_vector[0]) > 1e-8 or abs(complex_vector[4]) > 1e-8:
                    real_part = np.real(complex_vector)
                    omega = np.imag(val)  # <-- scalar, from the eigenVALUE
                    print("Real Part", real_part)

        if omega is None:
            raise RuntimeError(
                "No valid oscillatory (center) mode found at this libration point"
            )

        Ax = 0.01
        scale = Ax / real_part[0]
        x0 = X_l_point - Ax
        vy0_guess = scale * real_part[4]

        T_linear = (2 * np.pi) / np.abs(omega)  # now a real scalar

        return x0, vy0_guess, T_linear
    elif method == "3D":
        lpoint = 1  # 1, 2, or 3
        Az_km = 8000.0  # desired vertical amplitude
        primary_dist_km = 384400.0  # Earth-Moon separation
        (
            x0,
            y0,
            z0,
            vx0,
            vy0,
            vz0,
            T_linear,
        ) = richardson_halo_seed(
            mu=mu,
            lpoint=lpoint,
            Az_km=Az_km,
            primary_dist_km=primary_dist_km,
            northern=True,
            phase=0.0,
        )

        # phase=0 gives y0 = vx0 = vz0 = 0, which is exactly the
        # symmetry-plane structure expected by differential_corrector().
        return x0, z0, vy0, T_linear


def export_trajectory(state6, T, N=1000, filepath="trajectory.npz"):
    solution = integrator((0, T), state6, events=None, dense_output=True)
    t_array = np.linspace(0, T, N)
    # sol(t) returns the full 42-vector; slice to the 6 physical states
    state_array = np.array([solution.sol(t)[:6] for t in t_array])
    np.savez(filepath, t=t_array, state=state_array, mu=MU, T=T)
    return t_array, state_array


MU = returnMU()
x0, z0, vy, T_guess = calc_init_conditions(MU)

converged_state, T = differential_corrector(T_guess, x0=x0, z0_guess=z0, vy0_guess=vy, mu=MU, max_newton=1000, tol=1e-12)
solution = integrator((0, T), converged_state, None, MU,)

# Extract Monodromy and Floquet
final_sol_vector = solution.y[:, -1]
STM_flat = final_sol_vector[6:]
monodromy = STM_flat.reshape((6, 6))

print("Extracting Floquet Multipliers . . .")
floquet_multipliers, eigenvectors = eig(monodromy)


t, state = export_trajectory(converged_state, T ,)

print("trajectory: ", state)

plt.figure(figsize=(8, 6))
plt.plot(state[:, 0], state[:, 1], lw=0.8, color="steelblue", label="Spacecraft")
plt.plot(-MU, 0, "bo", markersize=10, label="Earth")  # Earth position
plt.plot(1 - MU, 0, "go", markersize=6, label="Moon")  # Moon position
plt.xlabel("x (non-dim)")
plt.ylabel("y (non-dim)")
plt.title("CR3BP Trajectory — Rotating Frame")
plt.legend()
plt.gca().set_aspect("equal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# print(f"Converged at: {solution.t[-1]:2f} time | Success: {solution.success}\n")

# print("--- Floquet Multipliers ---")
# for i, fm in enumerate(floquet_multipliers):
#     # Formats real and imaginary parts to 4 decimal places
#     print(f"λ_{i+1}: {fm.real:>9.4f} + {fm.imag:>9.4f}j")

# print("\n--- Eigenvectors ---")
# for i, row in enumerate(eigenvectors):
#     formatted_row = [f"{val.real:>8.4f}+{val.imag:>8.4f}j" for val in row]
#     print(f"Row {i+1}: [{', '.join(formatted_row)}]")
