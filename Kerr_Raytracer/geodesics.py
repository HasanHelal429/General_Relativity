"""
Kerr photon geodesic integrator, via the Carter constant in Mino time.

See Kerr_Raytracer_Plan.md's "The physics" section for the full
derivation. Summary: with P(r) = E(r^2+a^2) - a*L_z and
K = (L_z-aE)^2 + Q,

    R(r)      = P(r)^2 - Delta(r)*K
    Theta(th) = Q + cos^2(th)*(a^2*E^2 - L_z^2/sin^2(th))

and, in Mino time tau (d(tau) = d(lambda)/Sigma), r and theta obey fully
decoupled first-order equations dr/dtau = +-sqrt(R), dtheta/dtau =
+-sqrt(Theta). Differentiating once more removes both the turning-point
singularity and the need to track the +- sign by hand (exactly the same
trick Schwarzschild_Raytracer.geodesics uses for d^2u/dphi^2):

    d^2r/dtau^2     = (1/2) dR/dr
    d^2theta/dtau^2 = (1/2) dTheta/dtheta
    dphi/dtau       = -(a*E - L_z/sin^2(th)) + a*P(r)/Delta(r)

Validated by hand (see the Plan) to reduce exactly to
Schwarzschild_Raytracer.geodesics's own (du/dphi)^2 = 1/b^2-u^2*(1-2Mu)
at a=0, equatorial (Q=0); re-checked numerically in
"Geodesic Validation.ipynb" by comparing full trajectories, not just the
formulas.
"""

import numpy as np
from scipy.integrate import solve_ivp

import metric

SIN_THETA_FLOOR = 1e-8  # keeps 1/sin(theta) finite exactly at the poles (measure zero, not physically special)


def _safe_sin(theta):
    s = np.sin(theta)
    return np.where(np.abs(s) < SIN_THETA_FLOOR, np.sign(s) * SIN_THETA_FLOOR + (s == 0) * SIN_THETA_FLOOR, s)


def _P(r, E, Lz, a):
    return E * (r**2 + a**2) - a * Lz


def _K(E, Lz, Q, a):
    return (Lz - a * E) ** 2 + Q


def R_func(r, E, Lz, Q, a, M=metric.M_DEFAULT):
    return _P(r, E, Lz, a) ** 2 - metric.Delta(r, a, M) * _K(E, Lz, Q, a)


def Theta_func(theta, E, Lz, Q, a):
    s = _safe_sin(theta)
    return Q + np.cos(theta) ** 2 * (a**2 * E**2 - Lz**2 / s**2)


def _dR_dr(r, E, Lz, Q, a, M=metric.M_DEFAULT):
    K = _K(E, Lz, Q, a)
    return 4.0 * E * r * _P(r, E, Lz, a) - 2.0 * (r - M) * K


def _dTheta_dtheta(theta, E, Lz, Q, a):
    s = _safe_sin(theta)
    return 2.0 * np.cos(theta) * (Lz**2 / s**3 - a**2 * E**2 * s)


def _dphi_dtau(r, theta, E, Lz, a, M=metric.M_DEFAULT):
    s = _safe_sin(theta)
    return -(a * E - Lz / s**2) + a * _P(r, E, Lz, a) / metric.Delta(r, a, M)


def _rhs(_tau, y, E, Lz, Q, a, M):
    r, dr, theta, dtheta, _phi = y
    return [
        dr,
        0.5 * _dR_dr(r, E, Lz, Q, a, M),
        dtheta,
        0.5 * _dTheta_dtheta(theta, E, Lz, Q, a),
        _dphi_dtau(r, theta, E, Lz, a, M),
    ]


def _make_event(r_target, terminal_direction):
    def event(_tau, y, *_args):
        return y[0] - r_target
    event.terminal = True
    event.direction = terminal_direction
    return event


def integrate_ray(r0, theta0, phi0, dr0, dtheta0, E, Lz, Q, a, M=metric.M_DEFAULT,
                   tau_max=2000.0, r_escape=1.0e6, max_step=1e-2, horizon_buffer=1.001):
    """
    Integrate a Kerr photon trajectory in Mino time from (r0, theta0,
    phi0) with initial radial/polar velocities (dr0, dtheta0) (signed --
    see camera.py's ray_setup for how these come from a local ray
    direction) and conserved quantities (E, L_z, Q).

    Returns a dict with outcome ("horizon"/"escaped"/"trapped"), the full
    trajectory (r, theta, phi vs. tau), matching Schwarzschild_Raytracer's
    integrate_ray return shape.

    horizon_buffer: the capture event triggers at r = horizon_buffer *
    r_horizon, not exactly at the horizon. Boyer-Lindquist coordinates
    have a genuine coordinate singularity there (dphi/dtau ~ 1/Delta -> oo
    as Delta -> 0) -- unlike Schwarzschild_Raytracer's geodesics (which
    sidesteps this entirely by using phi itself, not an affine-like
    parameter, as the independent variable), this project's Mino-time
    parameterization needs r/theta *and* phi integrated together, and
    phi's blow-up forces the adaptive step size toward zero right at the
    horizon -- found via a real stall (a "trapped" outcome sitting at
    r essentially equal to r_horizon, never firing the capture event
    within a generous tau_max). A small buffer avoids ever integrating
    through that stiff region; physically harmless, since a photon this
    close to the horizon is captured for every practical (rendering)
    purpose regardless of the exact triggering radius.
    """
    r_hz = metric.r_horizon(a, M) * horizon_buffer
    event_horizon = _make_event(r_hz, terminal_direction=-1)
    event_escape = _make_event(r_escape, terminal_direction=+1)

    sol = solve_ivp(
        _rhs, (0.0, tau_max), [r0, dr0, theta0, dtheta0, phi0], args=(E, Lz, Q, a, M),
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
        "tau": sol.t,
        "r": sol.y[0],
        "dr": sol.y[1],
        "theta": sol.y[2],
        "dtheta": sol.y[3],
        "phi": sol.y[4],
    }


def equatorial_ray(r0, phi0, b, sign0, a, M=metric.M_DEFAULT, prograde=True, **kwargs):
    """Convenience wrapper for an equatorial (theta=pi/2, Q=0) photon with
    impact parameter b = L_z/E, matching Schwarzschild_Raytracer's
    parameterization -- used for the a=0 cross-check and the equatorial
    critical-impact-parameter validation. sign0 = +1 outward, -1 inward
    (same convention as Schwarzschild_Raytracer.geodesics.integrate_ray).
    prograde=False flips the sign of L_z (retrograde orbit) while keeping
    b's magnitude meaning intact.
    """
    E = 1.0
    Lz = b if prograde else -b
    theta0 = np.pi / 2
    Q = 0.0
    dr0 = sign0 * np.sqrt(max(R_func(r0, E, Lz, Q, a, M), 0.0))
    dtheta0 = 0.0
    return integrate_ray(r0, theta0, phi0, dr0, dtheta0, E, Lz, Q, a, M, **kwargs)
