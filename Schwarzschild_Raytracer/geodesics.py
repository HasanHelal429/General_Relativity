"""
Planar photon geodesic integrator for Schwarzschild spacetime.

Because Schwarzschild is spherically symmetric, a photon's full 3-vector
angular momentum is conserved, so its trajectory is confined to a single
plane through the center. Writing u = 1/r, the two photon conservation
laws (energy E and angular momentum L, with impact parameter b = L/E)
combine with the null condition into

    (du/dphi)^2 = 1/b**2 - u**2 * (1 - 2*M*u)

Differentiating once with respect to phi gives the standard second-order
form used here (valid through any radial turning point, no sign-flipping
needed once integration has started):

    d^2u/dphi^2 = 3*M*u**2 - u
"""

import numpy as np
from scipy.integrate import solve_ivp

import metric


def _rhs(_phi, y, M):
    u, up = y
    return [up, 3.0 * M * u**2 - u]


def _make_event(u_target, terminal_direction):
    def event(_phi, y, _M):
        return y[0] - u_target
    event.terminal = True
    event.direction = terminal_direction
    return event


def integrate_ray(r0, phi0, b, sign0, M=metric.M_DEFAULT, phi_max=50 * np.pi,
                   r_escape=1.0e6, max_step=1e-2):
    """
    Integrate a photon trajectory starting at (r0, phi0) with impact
    parameter b, where sign0 = +1 means the photon starts moving outward
    (dr/dphi > 0, i.e. du/dphi < 0) and sign0 = -1 means it starts moving
    inward.

    Returns a dict with:
        outcome   : "horizon", "escaped", or "trapped" (phi_max exhausted
                    without resolving -- only expected extremely close to
                    b_crit)
        phi, u    : full trajectory arrays (phi angle, u = 1/r)
        delta_phi : total angle swept (phi_end - phi0)
    """
    u0 = 1.0 / r0
    radicand = 1.0 / b**2 - u0**2 * (1.0 - 2.0 * M * u0)
    up0 = -np.sign(sign0) * np.sqrt(max(radicand, 0.0))

    u_horizon = 1.0 / metric.r_horizon(M)
    u_far = 1.0 / r_escape

    event_horizon = _make_event(u_horizon, terminal_direction=+1)
    event_escape = _make_event(u_far, terminal_direction=-1)

    sol = solve_ivp(
        _rhs, (phi0, phi0 + phi_max), [u0, up0], args=(M,),
        events=[event_horizon, event_escape],
        max_step=max_step, rtol=1e-9, atol=1e-12, dense_output=False,
    )

    if sol.t_events[0].size > 0:
        outcome = "horizon"
    elif sol.t_events[1].size > 0:
        outcome = "escaped"
    else:
        outcome = "trapped"

    return {
        "outcome": outcome,
        "phi": sol.t,
        "u": sol.y[0],
        "up": sol.y[1],
        "delta_phi": sol.t[-1] - phi0,
    }


def deflection_angle(b, M=metric.M_DEFAULT, r0=1.0e6, phi_max=50 * np.pi):
    """
    Net bending angle for a scattering trajectory (b > b_crit) that comes
    in from r0 (a stand-in for r=infinity) and escapes back out to r0.

    The baseline (zero-deflection) angle swept between two points at finite
    radius r0 on either side of a straight-line closest approach b is
    2*arccos(b/r0), not pi -- that only reduces to pi as r0 -> infinity.
    Subtracting the exact finite-r0 baseline instead of pi keeps this
    accurate without needing r0 >> b.

    Returns None if the ray does not escape (b <= b_crit, within r0's
    approximation of infinity).
    """
    result = integrate_ray(r0, 0.0, b, sign0=-1, M=M, phi_max=phi_max)
    if result["outcome"] != "escaped":
        return None
    baseline = 2.0 * np.arccos(b / r0)
    return result["delta_phi"] - baseline
