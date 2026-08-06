"""
Schwarzschild metric core quantities.

Geometrized units (G = c = 1). Mass M is a length scale; the module default
M=1 throughout the project, giving:
    horizon r_s        = 2M
    photon sphere       = 3M
    shadow (b_crit)     = 3*sqrt(3)*M

Photon effective potential
---------------------------
For a null geodesic in the equatorial plane of a photon's own orbital plane,
the two Killing-vector conservation laws (energy E from ∂t, angular momentum
L from ∂phi) combine with the null condition ds^2=0 into

    (dr/dlambda)^2 = E^2 - L^2 * f(r) / r^2 = E^2 - L^2 * V_eff(r)

so V_eff(r) = f(r)/r**2 = (1 - 2M/r)/r**2 plays the role of an effective
potential for the radial motion, with b = L/E the impact parameter setting
the "energy level" 1/b^2 the photon has to clear.

V_eff has a single maximum at r = 3M (set dV_eff/dr = 0 and solve), which is
why r = 3M is the *unstable* circular photon orbit (photon sphere): a photon
sitting exactly on that maximum orbits forever, but any perturbation grows.
The value of V_eff at that maximum is 1/(27*M**2), so the critical impact
parameter separating capture from escape is

    b_crit = 1/sqrt(V_eff(3M)) = 3*sqrt(3)*M
"""

import numpy as np

M_DEFAULT = 1.0


def f(r, M=M_DEFAULT):
    """Schwarzschild lapse-squared function 1 - 2M/r."""
    return 1.0 - 2.0 * M / r


def r_horizon(M=M_DEFAULT):
    """Event horizon radius (Schwarzschild radius), 2M."""
    return 2.0 * M


def r_photon_sphere(M=M_DEFAULT):
    """Unstable circular photon orbit radius, 3M."""
    return 3.0 * M


def b_critical(M=M_DEFAULT):
    """Critical impact parameter (black-hole shadow radius), 3*sqrt(3)*M."""
    return 3.0 * np.sqrt(3.0) * M


def v_eff(r, M=M_DEFAULT):
    """Photon effective potential f(r)/r**2."""
    return f(r, M) / r**2
