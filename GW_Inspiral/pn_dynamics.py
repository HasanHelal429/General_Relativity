"""
Two-body relative-motion equations of motion: Newtonian gravity plus
the leading-order (2.5PN) gravitational-wave radiation-reaction force.
Geometrized solar-mass units throughout (G=c=1, mass in M_sun) -- see
GW_Inspiral_Plan.md for the derivation and unit conventions, and
`waveform.py` for the SI conversions.

Unlike the raytracer projects' geodesic ODEs, `a_2.5PN` is
velocity-dependent and non-conservative by design (it's what drains
orbital energy and makes this an inspiral rather than a fixed ellipse),
which is why this integrates with an adaptive `solve_ivp` rather than
reusing N_Body_Gravity's kick-drift-kick leapfrog (leapfrog's symplectic
accuracy assumes velocity-independent forces).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_ISCO_CUTOFF_M = 6.0  # stop integrating once r drops below this many M -- PN itself
                        # is untrustworthy this close in; see Plan's Known limitations


def relative_accel(pos, vel, m1, m2):
    """Newtonian + 2.5PN radiation-reaction acceleration of the relative
    separation vector r = r1 - r2. `pos`, `vel` are (2,) arrays."""
    M = m1 + m2
    eta = m1 * m2 / M**2
    r = np.linalg.norm(pos)
    n = pos / r
    v2 = np.dot(vel, vel)
    rdot = np.dot(n, vel)

    a_newtonian = -(M / r**2) * n
    a_25pn = (8.0 / 5.0) * eta * (M**2 / r**3) * (
        (3.0 * v2 + (17.0 / 3.0) * M / r) * rdot * n - (v2 + 3.0 * M / r) * vel
    )
    return a_newtonian + a_25pn


def _rhs(t, y, m1, m2):
    pos, vel = y[0:2], y[2:4]
    acc = relative_accel(pos, vel, m1, m2)
    return [vel[0], vel[1], acc[0], acc[1]]


def _merger_event(r_cutoff):
    def event(t, y, m1, m2):
        return np.hypot(y[0], y[1]) - r_cutoff
    event.terminal = True
    event.direction = -1
    return event


def integrate_inspiral(m1, m2, a0, e0=0.0, r_cutoff_m=R_ISCO_CUTOFF_M,
                        t_max=None, max_step_frac=1.0 / 100.0, rtol=1e-9, atol=1e-11):
    """
    Integrate a binary from an initial osculating semi-major axis `a0`
    and eccentricity `e0` (started at apoapsis) down to a near-merger
    cutoff at `r_cutoff_m * (m1+m2)`, or `t_max` if given.

    Returns a dict with time and trajectory arrays (positions,
    velocities, and accelerations -- the same acceleration
    `relative_accel` computed at each retained step, reused directly by
    `waveform.py` rather than recomputed via finite differences).
    """
    M = m1 + m2
    r_cutoff = r_cutoff_m * M
    r0 = a0 * (1.0 + e0)  # apoapsis
    v0 = np.sqrt(M * (1.0 - e0) / (a0 * (1.0 + e0)))  # vis-viva, tangential at apoapsis
    y0 = [r0, 0.0, 0.0, v0]

    period0 = 2.0 * np.pi * np.sqrt(a0**3 / M)
    if t_max is None:
        t_max = 2.0 * (5.0 / 256.0) * a0**4 / (m1 * m2 * M)  # ~2x the circular-orbit Peters estimate

    sol = solve_ivp(
        _rhs, (0.0, t_max), y0, args=(m1, m2),
        events=_merger_event(r_cutoff), rtol=rtol, atol=atol,
        max_step=period0 * max_step_frac, dense_output=False,
    )

    pos = sol.y[0:2].T   # (n, 2)
    vel = sol.y[2:4].T
    acc = np.array([relative_accel(pos[i], vel[i], m1, m2) for i in range(len(sol.t))])

    return {
        "t": sol.t,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "merged": sol.status == 1,
        "m1": m1,
        "m2": m2,
    }


def semi_major_axis(pos, vel, m1, m2):
    """Osculating (instantaneous) semi-major axis from the two-body
    vis-viva relation. `pos`/`vel` may be single (2,) vectors or (n,2)
    arrays."""
    M = m1 + m2
    r = np.linalg.norm(pos, axis=-1)
    v2 = np.sum(np.asarray(vel) ** 2, axis=-1)
    return 1.0 / (2.0 / r - v2 / M)


def orbital_frequency(pos, vel):
    """Instantaneous orbital angular frequency d(phi)/dt = (x*vy -
    y*vx)/r^2. The gravitational-wave frequency is exactly twice this
    (the standard quadrupole-formula relation, since h ~ I_ddot ~
    cos/sin(2*phi))."""
    pos, vel = np.asarray(pos), np.asarray(vel)
    r2 = np.sum(pos**2, axis=-1)
    cross = pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0]
    return cross / r2


def peters_dadt(a, m1, m2, e=0.0):
    """Peters (1964) orbit-averaged semi-major-axis decay rate --
    derived independently (orbit-averaging the GW luminosity formula,
    not the instantaneous force law above), used as the primary
    validation target."""
    M = m1 + m2
    ecc_factor = (1.0 + 73.0 / 24.0 * e**2 + 37.0 / 96.0 * e**4) / (1.0 - e**2) ** 3.5
    return -64.0 / 5.0 * m1 * m2 * M / a**3 * ecc_factor


def peters_coalescence_time(a0, m1, m2, e0=0.0):
    """Peters (1964) time to coalescence from initial semi-major axis
    a0 (exact closed form for e0=0; e0!=0 uses the standard leading
    -order circular-equivalent estimate)."""
    M = m1 + m2
    T_circ = 5.0 / 256.0 * a0**4 / (m1 * m2 * M)
    if e0 == 0.0:
        return T_circ
    return T_circ * (1.0 - e0**2) ** 3.5  # standard leading-order eccentricity rescaling
