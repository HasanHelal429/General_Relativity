"""
Render loop: wires camera.py + geodesics.py + scene.py together into an
image. `render` is a straightforward per-pixel loop -- correct and easy
to reason about, but each pixel pays for its own `solve_ivp` call, which
is too slow for large images. `render_fast` replaces the integration
with a hand-written fixed-step vectorized RK4 that advances every
pixel's (r, dr, theta, dtheta, phi) state together as numpy arrays,
reusing geodesics.py's own (already-vectorized) derivative formulas
directly rather than re-deriving them -- and should be cross-checked
against `render` at a shared preview resolution before it's trusted at
full res (see Kerr_Raytracer.ipynb).

Unlike Schwarzschild_Raytracer's camera, Kerr's ray_setup has no
degenerate "purely radial ray" special case to handle separately: it
always returns well-defined (E, L_z, Q, dr0, dtheta0) directly from the
ZAMO-frame direction cosines, with no orbital-plane reduction step that
could become singular.

`render_fast`'s default step h=1e-3 (much smaller than
Schwarzschild_Raytracer.raytrace's own render_fast, which gets away
with h=1e-2) is load-bearing, not just a precision nicety: this
project's d^2r/dtau^2 = 0.5*dR/dr equation is genuinely stiffer near a
turning point than Schwarzschild's du/dphi equation, and h=1e-2 was
found (empirically, comparing single rays against `render`'s adaptive
solve_ivp) to occasionally manufacture a spurious turning point well
short of the true one for rays passing close to the critical impact
parameter, after which the fixed-step RK4 diverges (r runs away,
eventually overflowing to inf/NaN) instead of correctly continuing on
to capture. h=1e-3 was confirmed stable and accurate (r_min matching
the adaptive solver to ~1e-3) on the same rays; h=5e-4 tightens that
further with no qualitative change. See Kerr_Raytracer_Plan.md's Phase
5 entry for the full writeup.
"""

import numpy as np

import metric
import geodesics
import camera
import scene


def _color_ray(res, disk, r_isco, r_out):
    if disk:
        r_hit = scene.disk_crossing(res["theta"], res["r"], r_isco, r_out)
        if r_hit is not None:
            return scene.disk_color(r_hit, r_isco, r_out)

    if res["outcome"] == "escaped":
        theta_f, phi_f = res["theta"][-1], res["phi"][-1]
        direction = np.array([
            np.sin(theta_f) * np.cos(phi_f),
            np.sin(theta_f) * np.sin(phi_f),
            np.cos(theta_f),
        ])
        return scene.celestial_sphere_color(direction)

    # "horizon" (captured) or "trapped" (tau_max exhausted right at the
    # critical impact parameter, indistinguishable from capture at any
    # practical resolution)
    return scene.HORIZON_COLOR


def render(resolution, r_cam, theta_cam, phi_cam, a, M=metric.M_DEFAULT,
           fov_deg=60.0, disk=True, r_isco=None, r_out=scene.R_OUT_DEFAULT,
           prograde_disk=True, up_hint=(0.0, 0.0, 1.0), tau_max=2000.0,
           r_escape=1.0e6, max_step=1e-2, horizon_buffer=1.001):
    """
    Per-pixel MVP render. `resolution` is (nx, ny). Returns an (ny, nx, 3)
    RGB image in [0, 1]. `r_isco`, if not given explicitly, defaults to
    metric.r_isco(a, M, prograde=prograde_disk) -- i.e. the disk is
    assumed prograde (the common case) unless told otherwise.
    """
    if r_isco is None:
        r_isco = metric.r_isco(a, M, prograde=prograde_disk)

    position = camera.camera_position(r_cam, theta_cam, phi_cam)
    forward, right, up = camera.camera_basis(position, up_hint)
    dirs = camera.pixel_directions(resolution, fov_deg, forward, right, up)
    ny, nx = dirs.shape[:2]

    image = np.zeros((ny, nx, 3))
    for iy in range(ny):
        for ix in range(nx):
            E, Lz, Q, dr0, dtheta0 = camera.ray_setup(
                dirs[iy, ix], r_cam, theta_cam, phi_cam, a, M
            )
            res = geodesics.integrate_ray(
                r_cam, theta_cam, phi_cam, dr0, dtheta0, E, Lz, Q, a, M,
                tau_max=tau_max, r_escape=r_escape, max_step=max_step,
                horizon_buffer=horizon_buffer,
            )
            image[iy, ix] = _color_ray(res, disk, r_isco, r_out)

    return image


def _rk4_step_vec(state, h, E, Lz, Q, a, M):
    r, dr, theta, dtheta, phi = state

    def deriv(r, dr, theta, dtheta, phi):
        d2r = 0.5 * geodesics._dR_dr(r, E, Lz, Q, a, M)
        d2theta = 0.5 * geodesics._dTheta_dtheta(theta, E, Lz, Q, a)
        dphi = geodesics._dphi_dtau(r, theta, E, Lz, a, M)
        return dr, d2r, dtheta, d2theta, dphi

    k1 = deriv(r, dr, theta, dtheta, phi)
    k2 = deriv(*(x + 0.5 * h * k for x, k in zip(state, k1)))
    k3 = deriv(*(x + 0.5 * h * k for x, k in zip(state, k2)))
    k4 = deriv(*(x + h * k for x, k in zip(state, k3)))
    return tuple(
        x + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        for x, a1, a2, a3, a4 in zip(state, k1, k2, k3, k4)
    )


def render_fast(resolution, r_cam, theta_cam, phi_cam, a, M=metric.M_DEFAULT,
                 fov_deg=60.0, disk=True, r_isco=None, r_out=scene.R_OUT_DEFAULT,
                 prograde_disk=True, up_hint=(0.0, 0.0, 1.0), tau_max=500.0,
                 h=1e-3, r_escape=1.0e6, horizon_buffer=1.001):
    """
    Vectorized fixed-step RK4 render: advances every pixel's (r, dr,
    theta, dtheta, phi) state together as (N,) numpy arrays instead of
    one solve_ivp call per pixel. Same physics and same ray_setup as
    `render` (the (E, L_z, Q, dr0, dtheta0) formulas below are camera
    .ray_setup's, batched over pixels since the camera position itself
    is fixed) -- cross-validate against `render` before trusting this at
    full resolution.
    """
    if r_isco is None:
        r_isco = metric.r_isco(a, M, prograde=prograde_disk)

    position = camera.camera_position(r_cam, theta_cam, phi_cam)
    forward, right, up_vec = camera.camera_basis(position, up_hint)
    dirs = camera.pixel_directions(resolution, fov_deg, forward, right, up_vec)
    ny, nx = dirs.shape[:2]
    flat_dirs = dirs.reshape(-1, 3)
    n_rays = flat_dirs.shape[0]

    e_r, e_theta, e_phi = camera._local_spherical_basis(theta_cam, phi_cam)
    n_r = flat_dirs @ e_r
    n_theta = flat_dirs @ e_theta
    n_phi = flat_dirs @ e_phi

    Sigma_cam = metric.Sigma(r_cam, theta_cam, a)
    Delta_cam = metric.Delta(r_cam, a, M)
    A_cam = metric.A_func(r_cam, theta_cam, a, M)
    alpha_cam = metric.lapse(r_cam, theta_cam, a, M)
    omega_cam = metric.frame_dragging_omega(r_cam, theta_cam, a, M)
    sin_theta_cam = np.sin(theta_cam)

    Lz = n_phi * np.sqrt(A_cam / Sigma_cam) * sin_theta_cam
    E = alpha_cam + omega_cam * Lz
    Q = Sigma_cam * n_theta**2 - np.cos(theta_cam) ** 2 * (
        a**2 * E**2 - Lz**2 / sin_theta_cam**2
    )
    dr0 = n_r * np.sqrt(Delta_cam * Sigma_cam)
    dtheta0 = n_theta * np.sqrt(Sigma_cam)

    r = np.full(n_rays, r_cam)
    dr = dr0.copy()
    theta = np.full(n_rays, theta_cam)
    dtheta = dtheta0.copy()
    phi = np.full(n_rays, phi_cam)

    r_hz = metric.r_horizon(a, M) * horizon_buffer

    outcome = np.full(n_rays, "trapped", dtype=object)
    disk_rgb = np.zeros((n_rays, 3))
    active = np.ones(n_rays, dtype=bool)

    n_steps = int(np.ceil(tau_max / h))
    for _ in range(n_steps):
        if not active.any():
            break
        was_active = active.copy()

        if disk:
            r_prev = r
            c_prev = np.cos(theta)

        r_new, dr_new, theta_new, dtheta_new, phi_new = _rk4_step_vec(
            (r, dr, theta, dtheta, phi), h, E, Lz, Q, a, M
        )
        r = np.where(was_active, r_new, r)
        dr = np.where(was_active, dr_new, dr)
        theta = np.where(was_active, theta_new, theta)
        dtheta = np.where(was_active, dtheta_new, dtheta)
        phi = np.where(was_active, phi_new, phi)

        if disk:
            c_new = np.cos(theta)
            crossing = was_active & (c_prev * c_new < 0.0)
            if crossing.any():
                idx = np.where(crossing)[0]
                t = c_prev[idx] / (c_prev[idx] - c_new[idx])
                r_pts = r_prev[idx] + t * (r[idx] - r_prev[idx])
                in_disk = (r_pts >= r_isco) & (r_pts <= r_out)
                hit_idx = idx[in_disk]
                outcome[hit_idx] = "disk"
                disk_rgb[hit_idx] = scene.disk_color(r_pts[in_disk], r_isco, r_out)
                active[hit_idx] = False

        newly_horizon = active & (r <= r_hz)
        outcome[newly_horizon] = "horizon"
        active[newly_horizon] = False

        newly_escaped = active & (r >= r_escape)
        outcome[newly_escaped] = "escaped"
        active[newly_escaped] = False

    # any rays still unresolved exhausted tau_max right at the critical
    # impact parameter; treat as captured (matches `render`'s "trapped"
    # convention -- see _color_ray)
    outcome[active] = "horizon"

    final_dir = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    )

    image = np.zeros((n_rays, 3))
    is_disk = outcome == "disk"
    is_escaped = outcome == "escaped"
    is_horizon = outcome == "horizon"
    image[is_disk] = disk_rgb[is_disk]
    image[is_escaped] = scene.celestial_sphere_color(final_dir[is_escaped])
    image[is_horizon] = scene.HORIZON_COLOR

    return image.reshape(ny, nx, 3)
