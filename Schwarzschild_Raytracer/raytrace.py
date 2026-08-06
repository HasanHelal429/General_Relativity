"""
Render loop: wires camera.py + geodesics.py + scene.py together into an
image. `render` is a straightforward per-pixel loop -- correct and easy to
reason about, but each pixel pays for its own `solve_ivp` call, which is
too slow for large images. `render_fast` replaces the integration with a
hand-written fixed-step vectorized RK4 that advances every pixel's photon
state together as numpy arrays, and should be cross-checked against
`render` at a shared preview resolution before it's trusted at full res.
"""

import numpy as np

import metric
import geodesics
import camera
import scene


def _color_ray(res, e1, e2, disk, r_isco, r_out):
    r_traj = 1.0 / res["u"]
    positions = camera.plane_to_3d(r_traj, res["phi"], e1, e2)

    if disk:
        hit = scene.disk_crossing(positions, r_isco, r_out)
        if hit is not None:
            return scene.disk_color(np.linalg.norm(hit), r_isco, r_out)

    if res["outcome"] == "escaped":
        final_dir = positions[-1] / np.linalg.norm(positions[-1])
        return scene.celestial_sphere_color(final_dir)

    # "horizon" (captured) or "trapped" (phi_max exhausted right at b_crit,
    # indistinguishable from capture at any practical resolution)
    return scene.HORIZON_COLOR


def render(resolution, r_cam, theta_cam=np.pi / 2, phi_cam=0.0, M=metric.M_DEFAULT,
           fov_deg=60.0, disk=True, r_isco=scene.R_ISCO_DEFAULT, r_out=scene.R_OUT_DEFAULT,
           up_hint=(0.0, 0.0, 1.0), phi_max=50 * np.pi, max_step=1e-2):
    """
    Per-pixel MVP render. `resolution` is (nx, ny). Returns an (ny, nx, 3)
    RGB image in [0, 1].
    """
    position = camera.camera_position(r_cam, theta_cam, phi_cam)
    forward, right, up = camera.camera_basis(position, up_hint)
    dirs = camera.pixel_directions(resolution, fov_deg, forward, right, up)
    ny, nx = dirs.shape[:2]

    image = np.zeros((ny, nx, 3))
    for iy in range(ny):
        for ix in range(nx):
            b, sign0, e1, e2 = camera.ray_setup(dirs[iy, ix], position, M)

            if b < 1e-8:
                # Purely radial ray: degenerate orbital plane, but the
                # outcome is unambiguous from sign0 alone.
                image[iy, ix] = scene.HORIZON_COLOR if sign0 < 0 else scene.celestial_sphere_color(dirs[iy, ix])
                continue

            res = geodesics.integrate_ray(r_cam, 0.0, b, sign0, M=M, phi_max=phi_max, max_step=max_step)
            image[iy, ix] = _color_ray(res, e1, e2, disk, r_isco, r_out)

    return image


def _rk4_step_vec(u, up, h, M):
    def deriv(u, up):
        return up, 3.0 * M * u**2 - u

    k1u, k1up = deriv(u, up)
    k2u, k2up = deriv(u + 0.5 * h * k1u, up + 0.5 * h * k1up)
    k3u, k3up = deriv(u + 0.5 * h * k2u, up + 0.5 * h * k2up)
    k4u, k4up = deriv(u + h * k3u, up + h * k3up)
    u_new = u + (h / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    up_new = up + (h / 6.0) * (k1up + 2 * k2up + 2 * k3up + k4up)
    return u_new, up_new


def render_fast(resolution, r_cam, theta_cam=np.pi / 2, phi_cam=0.0, M=metric.M_DEFAULT,
                fov_deg=60.0, disk=True, r_isco=scene.R_ISCO_DEFAULT, r_out=scene.R_OUT_DEFAULT,
                up_hint=(0.0, 0.0, 1.0), phi_max=50 * np.pi, h=1e-2, r_escape=1e6):
    """
    Vectorized fixed-step RK4 render: advances every pixel's (u, du/dphi)
    state together as (N,) numpy arrays instead of one solve_ivp call per
    pixel. Same physics and same (b, sign0, plane) setup as `render`, just
    integrated all at once. Cross-validate against `render` before trusting
    this at full resolution -- see Geodesic/Raytrace validation notebooks.
    """
    position = camera.camera_position(r_cam, theta_cam, phi_cam)
    forward, right, up_vec = camera.camera_basis(position, up_hint)
    dirs = camera.pixel_directions(resolution, fov_deg, forward, right, up_vec)
    ny, nx = dirs.shape[:2]
    flat_dirs = dirs.reshape(-1, 3)
    n_rays = flat_dirs.shape[0]

    e1 = position / np.linalg.norm(position)
    a = flat_dirs @ e1
    tangential_vec = flat_dirs - a[:, None] * e1[None, :]
    tangential_mag = np.linalg.norm(tangential_vec, axis=1)
    b = r_cam * tangential_mag / np.sqrt(metric.f(r_cam, M))
    sign0 = np.where(a >= 0.0, 1.0, -1.0)

    radial_mask = tangential_mag <= 1e-12
    e2 = np.zeros((n_rays, 3))
    e2[~radial_mask] = tangential_vec[~radial_mask] / tangential_mag[~radial_mask, None]
    b_safe = np.where(radial_mask, 1.0, b)  # placeholder only; radial rays never enter the loop

    u = np.full(n_rays, 1.0 / r_cam)
    radicand = np.clip(1.0 / b_safe**2 - u**2 * (1.0 - 2.0 * M * u), 0.0, None)
    up = -sign0 * np.sqrt(radicand)
    phi = np.zeros(n_rays)

    u_horizon = 1.0 / metric.r_horizon(M)
    u_far = 1.0 / r_escape

    outcome = np.full(n_rays, "escaped", dtype=object)
    outcome[radial_mask & (sign0 < 0)] = "horizon"
    disk_rgb = np.zeros((n_rays, 3))
    active = ~radial_mask

    n_steps = int(np.ceil(phi_max / h))
    for _ in range(n_steps):
        if not active.any():
            break
        was_active = active.copy()

        if disk:
            pos_prev = camera.plane_to_3d(1.0 / u, phi, e1, e2)

        u_new, up_new = _rk4_step_vec(u, up, h, M)
        phi = np.where(was_active, phi + h, phi)
        u = np.where(was_active, u_new, u)
        up = np.where(was_active, up_new, up)

        if disk:
            pos_new = camera.plane_to_3d(1.0 / u, phi, e1, e2)
            z_prev, z_new = pos_prev[:, 2], pos_new[:, 2]
            crossing = was_active & (z_prev * z_new < 0.0)
            if crossing.any():
                t = z_prev[crossing] / (z_prev[crossing] - z_new[crossing])
                pts = pos_prev[crossing] + t[:, None] * (pos_new[crossing] - pos_prev[crossing])
                r_pts = np.linalg.norm(pts, axis=1)
                in_disk = (r_pts >= r_isco) & (r_pts <= r_out)
                idx = np.where(crossing)[0][in_disk]
                outcome[idx] = "disk"
                disk_rgb[idx] = scene.disk_color(r_pts[in_disk], r_isco, r_out)
                active[idx] = False

        newly_horizon = active & (u >= u_horizon)
        outcome[newly_horizon] = "horizon"
        active[newly_horizon] = False

        newly_escaped = active & (u <= u_far)
        outcome[newly_escaped] = "escaped"
        active[newly_escaped] = False

    # any rays still unresolved exhausted phi_max right at b_crit; treat as captured
    outcome[active] = "horizon"

    final_pos = camera.plane_to_3d(1.0 / u, phi, e1, e2)
    final_dir = final_pos / np.linalg.norm(final_pos, axis=1, keepdims=True)
    outward_radial = radial_mask & (sign0 >= 0)
    final_dir[outward_radial] = flat_dirs[outward_radial]

    image = np.zeros((n_rays, 3))
    is_disk = outcome == "disk"
    is_escaped = outcome == "escaped"
    is_horizon = outcome == "horizon"
    image[is_disk] = disk_rgb[is_disk]
    image[is_escaped] = scene.celestial_sphere_color(final_dir[is_escaped])
    image[is_horizon] = scene.HORIZON_COLOR

    return image.reshape(ny, nx, 3)
