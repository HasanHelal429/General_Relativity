"""
Kerr metric core quantities, in Boyer-Lindquist coordinates.

Geometrized units (G = c = 1), mass fixed at M = 1 throughout this
project (same convention as Schwarzschild_Raytracer), spin parameter
a = J/M in [0, 1) -- a=0 reduces to Schwarzschild exactly, the primary
validation target for every formula below (see Kerr_Raytracer_Plan.md's
"The physics" section for the full derivation this module implements).

    Sigma(r, theta) = r^2 + a^2*cos^2(theta)
    Delta(r)        = r^2 - 2*M*r + a^2
    A(r, theta)     = (r^2+a^2)^2 - a^2*Delta*sin^2(theta)   [g_phiphi = A*sin^2(theta)/Sigma]

    alpha(r, theta) = sqrt(Sigma*Delta/A)     [ZAMO lapse]
    omega(r, theta) = 2*M*a*r/A                [frame-dragging angular velocity]
"""

import numpy as np

M_DEFAULT = 1.0


def Sigma(r, theta, a):
    return r**2 + a**2 * np.cos(theta) ** 2


def Delta(r, a, M=M_DEFAULT):
    return r**2 - 2.0 * M * r + a**2


def A_func(r, theta, a, M=M_DEFAULT):
    return (r**2 + a**2) ** 2 - a**2 * Delta(r, a, M) * np.sin(theta) ** 2


def lapse(r, theta, a, M=M_DEFAULT):
    """ZAMO lapse alpha = sqrt(Sigma*Delta/A)."""
    return np.sqrt(Sigma(r, theta, a) * Delta(r, a, M) / A_func(r, theta, a, M))


def frame_dragging_omega(r, theta, a, M=M_DEFAULT):
    """Frame-dragging angular velocity omega = 2*M*a*r/A -- the angular
    velocity a ZAMO observer is forced to co-rotate at, purely from being
    at rest in the local orthonormal sense (zero angular momentum) rather
    than from any force. omega -> 0 as a -> 0 (no frame dragging without
    spin) and omega -> 2*M*a*r/(r^2+a^2)^2 far from the hole (~1/r^3,
    matching the familiar weak-field Lense-Thirring falloff)."""
    return 2.0 * M * a * r / A_func(r, theta, a, M)


def r_horizon(a, M=M_DEFAULT):
    """Outer event horizon radius, M + sqrt(M^2-a^2) (root of Delta=0).
    Reduces to 2M at a=0."""
    return M + np.sqrt(M**2 - a**2)


def r_ergo(theta, a, M=M_DEFAULT):
    """Ergosphere (static-limit surface) radius at polar angle theta,
    M + sqrt(M^2 - a^2*cos^2(theta)) (root of g_tt=0). Touches the horizon
    at the poles (theta=0,pi) and reduces to 2M everywhere at a=0 (no
    ergosphere without spin)."""
    return M + np.sqrt(M**2 - a**2 * np.cos(theta) ** 2)


def r_photon_sphere(a, M=M_DEFAULT, prograde=True):
    """Equatorial circular photon-orbit radius (Bardeen 1973 closed form):
    2M*[1 + cos((2/3)*arccos(mp a/M))], - for prograde (co-rotating, the
    smaller/tighter orbit) and + for retrograde. Both reduce to 3M at
    a=0, where there's no preferred rotation direction."""
    sign = -1.0 if prograde else 1.0
    return 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(sign * a / M)))


def r_isco(a, M=M_DEFAULT, prograde=True):
    """Innermost stable circular orbit radius (Bardeen-Press-Teukolsky
    1972 closed form). Reduces to 6M at a=0 regardless of prograde/
    retrograde (again, no preferred direction without spin)."""
    z1 = 1.0 + (1.0 - a**2 / M**2) ** (1.0 / 3.0) * (
        (1.0 + a / M) ** (1.0 / 3.0) + (1.0 - a / M) ** (1.0 / 3.0)
    )
    z2 = np.sqrt(3.0 * a**2 / M**2 + z1**2)
    sign = -1.0 if prograde else 1.0
    return M * (3.0 + z2 + sign * np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))
