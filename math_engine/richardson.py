"""Richardson halo seeds and CR3BP single-shooting differential correction.

The analytic portion implements Richardson (1980), Eqs. (8), (17)-(22) and
Appendix I.  Richardson's coefficients create a third-order *initial guess*;
they are not themselves a differential corrector.  The numerical portion of
this module integrates the 6x6 state-transition matrix (STM) and applies a
Newton correction to obtain a periodic CR3BP orbit.

State convention
----------------
``[x, y, z, xdot, ydot, zdot]`` in barycentric, rotating, nondimensional
CR3BP coordinates.  The large primary is at ``(-mu, 0, 0)`` and the small
primary at ``(1-mu, 0, 0)``.

Richardson's expansion applies only to L1/L2/L3.  L4/L5 use their planar
linear normal modes and a separate full-period corrector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


Array = np.ndarray
MU_ROUTH = 0.5 * (1.0 - np.sqrt(23.0 / 27.0))


def _check_mu(mu: float) -> None:
    if not (0.0 < mu <= 0.5):
        raise ValueError("mu must satisfy 0 < mu <= 0.5")


# ---------------------------------------------------------------------------
# CR3BP dynamics and variational equations
# ---------------------------------------------------------------------------

def cr3bp_rhs(_t: float, state: Array, mu: float) -> Array:
    """Return the six-dimensional CR3BP vector field."""
    x, y, z, vx, vy, vz = np.asarray(state, dtype=float)
    dx1, dx2 = x + mu, x - 1.0 + mu
    r1_sq = dx1 * dx1 + y * y + z * z
    r2_sq = dx2 * dx2 + y * y + z * z
    if r1_sq == 0.0 or r2_sq == 0.0:
        raise FloatingPointError("state lies at a CR3BP primary")
    r1_3, r2_3 = r1_sq ** 1.5, r2_sq ** 1.5
    ax = x + 2.0 * vy - (1.0 - mu) * dx1 / r1_3 - mu * dx2 / r2_3
    ay = y - 2.0 * vx - (1.0 - mu) * y / r1_3 - mu * y / r2_3
    az = -(1.0 - mu) * z / r1_3 - mu * z / r2_3
    return np.array([vx, vy, vz, ax, ay, az])


def cr3bp_jacobian(state: Array, mu: float) -> Array:
    """Return ``df/dstate`` for the rotating-frame CR3BP equations."""
    x, y, z = np.asarray(state, dtype=float)[:3]
    dx1, dx2 = x + mu, x - 1.0 + mu
    r1_sq = dx1 * dx1 + y * y + z * z
    r2_sq = dx2 * dx2 + y * y + z * z
    r1_3, r2_3 = r1_sq ** 1.5, r2_sq ** 1.5
    r1_5, r2_5 = r1_sq ** 2.5, r2_sq ** 2.5
    common = (1.0 - mu) / r1_3 + mu / r2_3

    uxx = 1.0 - common + 3.0 * (
        (1.0 - mu) * dx1 * dx1 / r1_5 + mu * dx2 * dx2 / r2_5
    )
    uyy = 1.0 - common + 3.0 * (
        (1.0 - mu) * y * y / r1_5 + mu * y * y / r2_5
    )
    uzz = -common + 3.0 * (
        (1.0 - mu) * z * z / r1_5 + mu * z * z / r2_5
    )
    uxy = 3.0 * (
        (1.0 - mu) * dx1 * y / r1_5 + mu * dx2 * y / r2_5
    )
    uxz = 3.0 * (
        (1.0 - mu) * dx1 * z / r1_5 + mu * dx2 * z / r2_5
    )
    uyz = 3.0 * (
        (1.0 - mu) * y * z / r1_5 + mu * y * z / r2_5
    )

    a = np.zeros((6, 6))
    a[:3, 3:] = np.eye(3)
    a[3:, :3] = np.array(
        [[uxx, uxy, uxz], [uxy, uyy, uyz], [uxz, uyz, uzz]]
    )
    a[3, 4] = 2.0
    a[4, 3] = -2.0
    return a


def _augmented_rhs(t: float, augmented: Array, mu: float) -> Array:
    state = augmented[:6]
    phi = augmented[6:].reshape(6, 6)
    return np.concatenate((cr3bp_rhs(t, state, mu), (cr3bp_jacobian(state, mu) @ phi).ravel()))


def jacobi_constant(state: Array, mu: float) -> float:
    """Return the CR3BP Jacobi integral."""
    x, y, z, vx, vy, vz = np.asarray(state, dtype=float)
    r1 = np.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y * y + z * z)
    omega = 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2
    return 2.0 * omega - (vx * vx + vy * vy + vz * vz)


# ---------------------------------------------------------------------------
# Libration points and Richardson coefficients
# ---------------------------------------------------------------------------

def _collinear_equilibrium(x: float, mu: float) -> float:
    dx1, dx2 = x + mu, x - 1.0 + mu
    return x - (1.0 - mu) * dx1 / abs(dx1) ** 3 - mu * dx2 / abs(dx2) ** 3


def collinear_libration_point(mu: float, lpoint: Literal[1, 2, 3]) -> float:
    """Compute L1, L2, or L3 directly from the equilibrium equation."""
    _check_mu(mu)
    eps = 1.0e-12
    if lpoint == 1:
        bracket = (-mu + eps, 1.0 - mu - eps)
    elif lpoint == 2:
        bracket = (1.0 - mu + eps, 3.0)
    elif lpoint == 3:
        bracket = (-3.0, -mu - eps)
    else:
        raise ValueError("lpoint must be 1, 2, or 3")
    return float(brentq(_collinear_equilibrium, *bracket, args=(mu,), xtol=1e-14))


def _gamma(mu: float, lpoint: int) -> tuple[float, float]:
    x_l = collinear_libration_point(mu, lpoint)
    reference_primary = 1.0 - mu if lpoint in (1, 2) else -mu
    return abs(x_l - reference_primary), x_l


def _cn(n: int, mu: float, gamma: float, lpoint: int) -> float:
    """Richardson (1980), Eq. (8), adapted to the standard CR3BP mu."""
    if lpoint == 1:
        return (
            mu + (-1.0) ** n * (1.0 - mu) * gamma ** (n + 1) / (1.0 - gamma) ** (n + 1)
        ) / gamma**3
    if lpoint == 2:
        return (-1.0) ** n * (
            mu + (1.0 - mu) * gamma ** (n + 1) / (1.0 + gamma) ** (n + 1)
        ) / gamma**3
    if lpoint == 3:
        return (
            1.0 - mu + mu * gamma ** (n + 1) / (1.0 + gamma) ** (n + 1)
        ) / gamma**3
    raise ValueError("lpoint must be 1, 2, or 3")


@dataclass(frozen=True)
class RichardsonCoefficients:
    """All factors used by Richardson's complete third-order solution."""

    gamma: float
    x_libration: float
    c2: float
    c3: float
    c4: float
    lam: float
    k: float
    delta: float
    d1: float
    d2: float
    a21: float
    a22: float
    a23: float
    a24: float
    a31: float
    a32: float
    b21: float
    b22: float
    b31: float
    b32: float
    d21: float
    d31: float
    d32: float
    s1: float
    s2: float
    l1: float
    l2: float


def richardson_coefficients(mu: float, lpoint: Literal[1, 2, 3]) -> RichardsonCoefficients:
    """Evaluate Richardson's Appendix-I factors without omitted coefficients."""
    gamma, x_l = _gamma(mu, lpoint)
    c2, c3, c4 = (_cn(n, mu, gamma, lpoint) for n in (2, 3, 4))

    # Positive root of q^2 + (c2 - 2)q - (c2 - 1)(1 + 2c2) = 0, q=lambda^2.
    q = 0.5 * (2.0 - c2 + np.sqrt(9.0 * c2 * c2 - 8.0 * c2))
    if q <= 0.0:
        raise RuntimeError(f"non-positive lambda^2 at L{lpoint}: {q}")
    lam = np.sqrt(q)
    k = 2.0 * lam / (lam * lam + 1.0 - c2)
    delta = lam * lam - c2

    d1 = 3.0 * lam * lam / k * (k * (6.0 * lam * lam - 1.0) - 2.0 * lam)
    d2 = 8.0 * lam * lam / k * (k * (11.0 * lam * lam - 1.0) - 2.0 * lam)

    a21 = 3.0 * c3 * (k * k - 2.0) / (4.0 * (1.0 + 2.0 * c2))
    a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2))
    a23 = -3.0 * c3 * lam / (4.0 * k * d1) * (
        3.0 * k**3 * lam - 6.0 * k * (k - lam) + 4.0
    )
    a24 = -3.0 * c3 * lam / (4.0 * k * d1) * (2.0 + 3.0 * k * lam)
    b21 = -3.0 * c3 * lam / (2.0 * d1) * (3.0 * k * lam - 4.0)
    b22 = 3.0 * c3 * lam / d1
    d21 = -c3 / (2.0 * lam * lam)

    a31 = (
        -9.0 * lam / (4.0 * d2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k * k))
        + (9.0 * lam * lam + 1.0 - c2) / (2.0 * d2)
        * (3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * k * k))
    )
    a32 = -1.0 / d2 * (
        9.0 * lam / 4.0 * (4.0 * c3 * (k * a24 - b22) + k * c4)
        + 1.5 * (9.0 * lam * lam + 1.0 - c2)
        * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
    )
    b31 = 3.0 / (8.0 * d2) * (
        8.0 * lam * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * k * k))
        + (9.0 * lam * lam + 1.0 + 2.0 * c2)
        * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k * k))
    )
    b32 = 1.0 / d2 * (
        9.0 * lam * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
        + 3.0 / 8.0 * (9.0 * lam * lam + 1.0 + 2.0 * c2)
        * (4.0 * c3 * (k * a24 - b22) + k * c4)
    )
    d31 = 3.0 / (64.0 * lam * lam) * (4.0 * c3 * a24 + c4)
    d32 = 3.0 / (64.0 * lam * lam) * (
        4.0 * c3 * (a23 - d21) + c4 * (4.0 + k * k)
    )

    frequency_denominator = 2.0 * lam * (lam * (1.0 + k * k) - 2.0 * k)
    s1 = 1.0 / frequency_denominator * (
        1.5 * c3 * (2.0 * a21 * (k * k - 2.0) - a23 * (k * k + 2.0) - 2.0 * k * b21)
        - 3.0 / 8.0 * c4 * (3.0 * k**4 - 8.0 * k * k + 8.0)
    )
    s2 = 1.0 / frequency_denominator * (
        1.5 * c3 * (2.0 * a22 * (k * k - 2.0) + a24 * (k * k + 2.0) + 2.0 * k * b22 + 5.0 * d21)
        + 3.0 / 8.0 * c4 * (12.0 - k * k)
    )
    alpha1 = -1.5 * c3 * (2.0 * a21 + a23 + 5.0 * d21) - 3.0 / 8.0 * c4 * (12.0 - k * k)
    alpha2 = 1.5 * c3 * (a24 - 2.0 * a22) + 9.0 / 8.0 * c4
    l1 = alpha1 + 2.0 * lam * lam * s1
    l2 = alpha2 + 2.0 * lam * lam * s2

    return RichardsonCoefficients(
        gamma, x_l, c2, c3, c4, lam, k, delta, d1, d2,
        a21, a22, a23, a24, a31, a32, b21, b22, b31, b32,
        d21, d31, d32, s1, s2, l1, l2,
    )


def richardson_halo_seed(
    mu: float,
    lpoint: Literal[1, 2, 3],
    Az_km: float,
    primary_dist_km: float,
    northern: bool = True,
    phase: float = 0.0,
    validate_with=None,
) -> tuple[float, float, float, float, float, float, float]:
    """Return Richardson's third-order state and period estimate.

    ``northern`` selects Richardson's Class-I/Class-II sign independently of
    the libration point.  The earlier implementation incorrectly changed this
    sign at L2.  ``phase=0`` gives the symmetry-plane state used by the halo
    corrector: ``y=xdot=zdot=0``.
    """
    if Az_km < 0.0 or primary_dist_km <= 0.0:
        raise ValueError("Az_km must be nonnegative and primary_dist_km positive")
    c = richardson_coefficients(mu, lpoint)
    az = Az_km / (primary_dist_km * c.gamma)
    ax_sq = -(c.delta + c.l2 * az * az) / c.l1
    if ax_sq < -1e-14:
        raise ValueError(
            f"Az={Az_km:g} km is outside this third-order branch (Ax^2={ax_sq:.6e})"
        )
    ax = np.sqrt(max(0.0, ax_sq))
    branch_sign = 1.0 if northern else -1.0  # Richardson delta_n=2-n, n=1 or 3.
    tau = float(phase)

    xbar = (
        c.a21 * ax**2 + c.a22 * az**2 - ax * np.cos(tau)
        + (c.a23 * ax**2 - c.a24 * az**2) * np.cos(2.0 * tau)
        + (c.a31 * ax**3 - c.a32 * ax * az**2) * np.cos(3.0 * tau)
    )
    ybar = (
        c.k * ax * np.sin(tau)
        + (c.b21 * ax**2 - c.b22 * az**2) * np.sin(2.0 * tau)
        + (c.b31 * ax**3 - c.b32 * ax * az**2) * np.sin(3.0 * tau)
    )
    zbar = branch_sign * (
        az * np.cos(tau) + c.d21 * ax * az * (np.cos(2.0 * tau) - 3.0)
        + (c.d32 * az * ax**2 - c.d31 * az**3) * np.cos(3.0 * tau)
    )
    dx_dtau = (
        ax * np.sin(tau)
        - 2.0 * (c.a23 * ax**2 - c.a24 * az**2) * np.sin(2.0 * tau)
        - 3.0 * (c.a31 * ax**3 - c.a32 * ax * az**2) * np.sin(3.0 * tau)
    )
    dy_dtau = (
        c.k * ax * np.cos(tau)
        + 2.0 * (c.b21 * ax**2 - c.b22 * az**2) * np.cos(2.0 * tau)
        + 3.0 * (c.b31 * ax**3 - c.b32 * ax * az**2) * np.cos(3.0 * tau)
    )
    dz_dtau = branch_sign * (
        -az * np.sin(tau) - 2.0 * c.d21 * ax * az * np.sin(2.0 * tau)
        - 3.0 * (c.d32 * az * ax**2 - c.d31 * az**3) * np.sin(3.0 * tau)
    )

    nu = 1.0 + c.s1 * ax * ax + c.s2 * az * az
    angular_rate = c.lam * nu
    scale = c.gamma
    state = np.array(
        [
            c.x_libration + scale * xbar, # x
            scale * ybar, # y
            scale * zbar,# z
            scale * angular_rate * dx_dtau, # vx
            scale * angular_rate * dy_dtau, # vy
            scale * angular_rate * dz_dtau, # vz
        ]
    )
    period = 2.0 * np.pi / angular_rate
    if validate_with is not None:
        validate_with(state.copy(), period, mu)
    return (*state, period)


# ---------------------------------------------------------------------------
# Symmetry-constrained 3D halo differential correction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrectionResult:
    state: Array
    period: float
    converged: bool
    iterations: int
    residual_norm: float


def _half_period_crossing(state0: Array, mu: float, max_time: float, rtol: float, atol: float):
    direction = -np.sign(state0[4])
    if direction == 0.0:
        raise ValueError("initial ydot must be nonzero for symmetry-plane shooting")

    def y_crossing(_t: float, augmented: Array, _mu: float) -> float:
        return float(augmented[1])

    y_crossing.terminal = True
    y_crossing.direction = direction
    augmented0 = np.concatenate((state0, np.eye(6).ravel()))
    sol = solve_ivp(
        _augmented_rhs, (0.0, max_time), augmented0, args=(mu,),
        events=y_crossing, rtol=rtol, atol=atol, method="DOP853",
    )
    if not sol.t_events[0].size:
        raise RuntimeError("no opposite-direction y=0 crossing found within max_time")
    final = sol.y_events[0][0]
    return float(sol.t_events[0][0]), final[:6], final[6:].reshape(6, 6)


# def differential_correct_halo(
#     initial_state: Array,
#     mu: float,
#     period_guess: float | None = None,
#     tol: float = 1e-11,
#     max_iterations: int = 20,
#     rtol: float = 2e-12,
#     atol: float = 2e-14,
# ) -> CorrectionResult:
#     """Correct an x-z symmetric 3D halo orbit by single shooting.

#     The family parameter ``z0`` is held fixed.  Newton updates ``x0`` and
#     ``ydot0`` so that ``xdot(tf/2)=zdot(tf/2)=0`` at the next ``y=0``
#     crossing.  Event-time sensitivity is included explicitly; omitting it is
#     a common source of false convergence in 3D correctors.
#     """
#     _check_mu(mu)
#     state = np.array(initial_state, dtype=float, copy=True)
#     if state.shape != (6,):
#         raise ValueError("initial_state must have shape (6,)")
#     state[[1, 3, 5]] = 0.0
#     max_time = 0.75 * period_guess if period_guess is not None else 10.0
#     residual_norm = np.inf

#     for iteration in range(max_iterations + 1):
#         half_period, final, phi = _half_period_crossing(state, mu, max_time, rtol, atol)
#         residual = final[[3, 5]]
#         residual_norm = float(np.linalg.norm(residual, ord=np.inf))
#         if residual_norm <= tol:
#             return CorrectionResult(state.copy(), 2.0 * half_period, True, iteration, residual_norm)
#         if iteration == max_iterations:
#             break

#         f_final = cr3bp_rhs(half_period, final, mu)
#         if abs(f_final[1]) < 1e-14:
#             raise RuntimeError("grazing y=0 crossing makes the correction Jacobian singular")
#         rows = np.array([3, 5])
#         cols = np.array([0, 4])
#         correction_matrix = phi[np.ix_(rows, cols)] - np.outer(
#             f_final[rows] / f_final[1], phi[1, cols]
#         )
#         step = np.linalg.solve(correction_matrix, -residual)

#         # Backtracking protects against a large Newton jump from a rough seed.
#         step_scale = 1.0
#         accepted = False
#         for _ in range(8):
#             trial = state.copy()
#             trial[cols] += step_scale * step
#             try:
#                 _, trial_final, _ = _half_period_crossing(trial, mu, max_time, rtol, atol)
#                 trial_norm = np.linalg.norm(trial_final[rows], ord=np.inf)
#             except RuntimeError:
#                 trial_norm = np.inf
#             if trial_norm < residual_norm:
#                 state = trial
#                 accepted = True
#                 break
#             step_scale *= 0.5
#         if not accepted:
#             state[cols] += 0.125 * step

#     return CorrectionResult(state, 2.0 * half_period, False, max_iterations, residual_norm)


# ---------------------------------------------------------------------------
# L4/L5 planar modes and full-period correction
# ---------------------------------------------------------------------------

def l4l5_seed(
    mu: float,
    compute_A_matrix_fn=None,
    which: Literal["L4", "L5"] = "L4",
    amplitude: float = 0.01,
    mode: Literal["short", "long"] = "short",
) -> tuple[float, float, float, float, float, float, float]:
    """Return a planar linear-mode seed about L4 or L5.

    The old routine sorted all imaginary eigenvalues and accidentally selected
    the vertical unit-frequency mode as the "short" planar mode.  This version
    rejects vertical eigenvectors, phase-normalizes the planar eigenvector, and
    exposes both planar families.
    """
    _check_mu(mu)
    if mu >= MU_ROUTH:
        raise ValueError(
            f"L4/L5 planar centers require mu < Routh's value {MU_ROUTH:.12f}; got {mu}"
        )
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    if which not in ("L4", "L5"):
        raise ValueError("which must be 'L4' or 'L5'")
    if mode not in ("short", "long"):
        raise ValueError("mode must be 'short' or 'long'")

    x_l = 0.5 - mu
    y_l = (1.0 if which == "L4" else -1.0) * np.sqrt(3.0) / 2.0
    equilibrium = np.array([x_l, y_l, 0.0, 0.0, 0.0, 0.0])
    # Retain the old dependency-injection argument for callers that already
    # pass their own A-matrix function, while supplying a correct internal one.
    a_matrix = (
        cr3bp_jacobian(equilibrium, mu)
        if compute_A_matrix_fn is None
        else np.asarray(compute_A_matrix_fn(equilibrium, mu), dtype=float)
    )
    if a_matrix.shape != (6, 6):
        raise ValueError("compute_A_matrix_fn must return a 6x6 array")
    values, vectors = np.linalg.eig(a_matrix)
    planar_modes: list[tuple[float, Array]] = []
    for index, value in enumerate(values):
        vector = vectors[:, index]
        planar_power = np.linalg.norm(vector[[0, 1, 3, 4]])
        vertical_power = np.linalg.norm(vector[[2, 5]])
        if value.imag > 1e-9 and abs(value.real) < 1e-8 and planar_power > 100.0 * vertical_power:
            planar_modes.append((float(value.imag), vector))
    if len(planar_modes) != 2:
        raise RuntimeError(f"expected two planar center modes at {which}; found {len(planar_modes)}")
    planar_modes.sort(key=lambda item: item[0])
    omega, vector = planar_modes[-1 if mode == "short" else 0]

    # Rotate the arbitrary complex eigenvector phase so delta-x is real/positive.
    vector = vector * np.exp(-1j * np.angle(vector[0]))
    if abs(vector[0].real) < 1e-12:
        raise RuntimeError("could not phase-normalize the selected planar eigenvector")
    perturbation = amplitude / vector[0].real * vector.real
    state = equilibrium + perturbation
    state[[2, 5]] = 0.0
    return (*state, 2.0 * np.pi / omega)


def differential_correct_l4l5(
    initial_state: Array,
    period_guess: float,
    mu: float,
    tol: float = 1e-11,
    max_iterations: int = 20,
    rtol: float = 2e-12,
    atol: float = 2e-14,
) -> CorrectionResult:
    """Correct a planar L4/L5 periodic orbit with full-period shooting.

    ``x0`` is the continuation/family parameter.  Newton solves for
    ``[y0, xdot0, ydot0, T]`` from planar state closure.  This is intentionally
    separate from the collinear halo corrector because the local center-center
    structure and appropriate constraints are different.
    """
    _check_mu(mu)
    state = np.array(initial_state, dtype=float, copy=True)
    if state.shape != (6,):
        raise ValueError("initial_state must have shape (6,)")
    state[[2, 5]] = 0.0
    period = float(period_guess)
    planar = np.array([0, 1, 3, 4])
    variables = np.array([1, 3, 4])
    residual_norm = np.inf

    for iteration in range(max_iterations + 1):
        augmented0 = np.concatenate((state, np.eye(6).ravel()))
        sol = solve_ivp(
            _augmented_rhs, (0.0, period), augmented0, args=(mu,),
            rtol=rtol, atol=atol, method="DOP853",
        )
        final = sol.y[:6, -1]
        phi = sol.y[6:, -1].reshape(6, 6)
        residual = final[planar] - state[planar]
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm <= tol:
            return CorrectionResult(state.copy(), period, True, iteration, residual_norm)
        if iteration == max_iterations:
            break

        jac = np.empty((4, 4))
        # d(X_f-X_0)/dq = Phi(:,q)-I(:,q) for q=[y0,xdot0,ydot0].
        jac[:, :3] = (
            phi[np.ix_(planar, variables)]
            - np.eye(6)[np.ix_(planar, variables)]
        )
        jac[:, 3] = cr3bp_rhs(period, final, mu)[planar]
        step, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        state[variables] += step[:3]
        period += step[3]
        if period <= 0.0:
            raise RuntimeError("L4/L5 Newton iteration produced a non-positive period")

    return CorrectionResult(state, period, False, max_iterations, residual_norm)


__all__ = [
    "CorrectionResult",
    "MU_ROUTH",
    "RichardsonCoefficients",
    "collinear_libration_point",
    "cr3bp_jacobian",
    "cr3bp_rhs",
    "differential_correct_halo",
    "differential_correct_l4l5",
    "jacobi_constant",
    "l4l5_seed",
    "richardson_coefficients",
    "richardson_halo_seed",
]