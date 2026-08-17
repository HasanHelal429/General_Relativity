"""
Leading-order (quadrupole-formula) gravitational-wave strain from a
simulated binary trajectory, plus geometrized-solar-mass <-> SI unit
conversions.

For a face-on observer (along the orbital-plane normal), h+ = (G/(c^4
D))(I_xx_ddot - I_yy_ddot), h_x = (2G/(c^4 D)) I_xy_ddot, with the
*reduced* mass quadrupole I_ij = mu*(x_i x_j - delta_ij r^2/3) for the
relative separation vector (equivalent, in the two-body reduction, to
summing each body's individual quadrupole about the center of mass).
Differentiated analytically rather than by finite-differencing the
trajectory (see GW_Inspiral_Plan.md's derivation) -- the delta_ij trace
term cancels identically in h+ and never appears in h_x, so only the
untraced x_i*x_j second derivative is needed:

    I_xx_ddot = 2*(vx^2 + x*ax)
    I_yy_ddot = 2*(vy^2 + y*ay)
    I_xy_ddot = 2*vx*vy + x*ay + y*ax

using the exact (pos, vel, acc) the integrator already produced --
`acc` is `pn_dynamics.relative_accel` evaluated at each retained step,
not a separate finite-difference estimate.
"""

import numpy as np

# Geometrized solar-mass units (G=c=1, mass in M_sun) <-> SI, per
# GW_Inspiral_Plan.md's convention.
G_SI = 6.674e-11        # m^3/(kg s^2)
C_SI = 2.998e8           # m/s
M_SUN_KG = 1.989e30      # kg

LENGTH_UNIT_M = G_SI * M_SUN_KG / C_SI**2   # meters per geometrized mass-unit of length
TIME_UNIT_S = G_SI * M_SUN_KG / C_SI**3      # seconds per geometrized mass-unit of time

MPC_IN_M = 3.0857e22  # meters per megaparsec


def strain_plus_cross(pos, vel, acc, mu, D):
    """h+, h_x for a face-on observer at (geometrized) distance D, given
    the relative-motion trajectory (mu = reduced mass = m1*m2/(m1+m2)).
    `pos`/`vel`/`acc` may be (2,) vectors or (n,2) arrays; returns
    matching-shape (scalar or (n,)) h+, h_x."""
    pos, vel, acc = np.asarray(pos), np.asarray(vel), np.asarray(acc)
    x, y = pos[..., 0], pos[..., 1]
    vx, vy = vel[..., 0], vel[..., 1]
    ax, ay = acc[..., 0], acc[..., 1]

    Ixx_ddot = 2.0 * (vx**2 + x * ax)
    Iyy_ddot = 2.0 * (vy**2 + y * ay)
    Ixy_ddot = 2.0 * vx * vy + x * ay + y * ax

    prefactor = mu / D  # G=c=1 here; SI G/c^4 factor applied in strain_plus_cross_SI
    h_plus = prefactor * (Ixx_ddot - Iyy_ddot)
    h_cross = 2.0 * prefactor * Ixy_ddot
    return h_plus, h_cross


def geometrized_to_seconds(t_geo):
    return t_geo * TIME_UNIT_S


def geometrized_freq_to_hz(f_geo):
    """Angular frequency in geometrized units (1/mass-unit-of-time) to
    ordinary frequency in Hz: f_Hz = (f_geo / (2*pi)) / TIME_UNIT_S."""
    return (f_geo / (2.0 * np.pi)) / TIME_UNIT_S


def mpc_to_geometrized(D_mpc):
    return D_mpc * MPC_IN_M / LENGTH_UNIT_M


def strain_plus_cross_SI(pos, vel, acc, mu, D_mpc):
    """Same as strain_plus_cross, but with the reduced mass mu in M_sun
    and the observer distance D given in Mpc -- returns dimensionless
    strain directly comparable to a real detector's h(t) (still using
    the restricted, face-on, quadrupole-only approximation)."""
    D_geo = mpc_to_geometrized(D_mpc)
    return strain_plus_cross(pos, vel, acc, mu, D_geo)
