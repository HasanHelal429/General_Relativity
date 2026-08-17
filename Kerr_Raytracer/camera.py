"""
Pinhole camera and ray -> (E, L_z, Q) mapping via the ZAMO frame.

`camera_position`, `camera_basis`, `pixel_directions` are pure flat-R^3
pinhole-camera geometry, independent of the metric -- copied verbatim
from Schwarzschild_Raytracer.camera (not imported cross-folder, per this
repo's self-contained-project convention).

`ray_setup` replaces Schwarzschild's version: Kerr has no single orbital
plane to reduce a ray to (see Kerr_Raytracer_Plan.md), so instead of
`(b, sign0, e1, e2)` it returns the conserved quantities `(E, L_z, Q)`
and signed initial `(dr, dtheta)` that geodesics.integrate_ray needs
directly. Derived from the ZAMO (zero-angular-momentum-observer) frame,
the natural replacement for Schwarzschild's static-observer frame (no
observer can hold position fixed inside the ergosphere): with lapse
`alpha`, frame-dragging `omega`, and a camera ray's local ZAMO-frame
direction cosines `(n_r, n_theta, n_phi)` (unit vector, obtained the same
way Schwarzschild's camera got its `(radial, tangential)` split -- flat
-embedding spherical unit vectors coincide with the local orthonormal
tetrad once the metric's scale factors are accounted for, exactly as
Schwarzschild_Raytracer.camera's docstring argues),

    L_z = n_phi * sqrt(A/Sigma) * sin(theta)
    E   = alpha + omega*L_z
    Q   = Sigma*n_theta^2 - cos^2(theta)*(a^2*E^2 - L_z^2/sin^2(theta))
    dr/dtau|_0     = n_r * sqrt(Delta*Sigma)
    dtheta/dtau|_0 = n_theta * sqrt(Sigma)

worked through by hand from the Kerr metric's ADM lapse/shift
decomposition before writing this module (see the Plan's "The physics"
section for the full derivation).
"""

import numpy as np

import metric


def camera_position(r_cam, theta_cam=np.pi / 2, phi_cam=0.0):
    """Camera position in the flat R^3 embedding."""
    st, ct = np.sin(theta_cam), np.cos(theta_cam)
    sp, cp = np.sin(phi_cam), np.cos(phi_cam)
    return r_cam * np.array([st * cp, st * sp, ct])


def camera_basis(position, up_hint=(0.0, 0.0, 1.0)):
    """Standard look-at basis for a camera at `position` looking toward
    the origin, with `up_hint` fixing the roll."""
    position = np.asarray(position, dtype=float)
    forward = -position / np.linalg.norm(position)
    up_hint = np.asarray(up_hint, dtype=float)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def pixel_directions(resolution, fov_deg, forward, right, up):
    """Unit ray direction (flat R^3 embedding) for every pixel of a
    pinhole camera. Returns an array of shape (ny, nx, 3)."""
    nx, ny = resolution
    fov_x = np.radians(fov_deg)
    half_w = np.tan(fov_x / 2.0)
    half_h = half_w * ny / nx

    u = np.linspace(-half_w, half_w, nx)
    v = np.linspace(half_h, -half_h, ny)
    uu, vv = np.meshgrid(u, v)

    directions = (
        forward[None, None, :]
        + uu[:, :, None] * right[None, None, :]
        + vv[:, :, None] * up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions


def _local_spherical_basis(theta_cam, phi_cam):
    """Orthonormal (e_r, e_theta, e_phi) at (theta_cam, phi_cam) in the
    flat R^3 embedding -- the local proper tetrad directions, same
    argument as Schwarzschild_Raytracer.camera's docstring."""
    st, ct = np.sin(theta_cam), np.cos(theta_cam)
    sp, cp = np.sin(phi_cam), np.cos(phi_cam)
    e_r = np.array([st * cp, st * sp, ct])
    e_theta = np.array([ct * cp, ct * sp, -st])
    e_phi = np.array([-sp, cp, 0.0])
    return e_r, e_theta, e_phi


def ray_setup(direction, r_cam, theta_cam, phi_cam, a, M=metric.M_DEFAULT):
    """
    For a unit ray direction leaving the camera at (r_cam, theta_cam,
    phi_cam), compute the conserved quantities (E, L_z, Q) and signed
    initial (dr, dtheta) that geodesics.integrate_ray needs -- see this
    module's docstring for the derivation.
    """
    e_r, e_theta, e_phi = _local_spherical_basis(theta_cam, phi_cam)
    n_r = np.dot(direction, e_r)
    n_theta = np.dot(direction, e_theta)
    n_phi = np.dot(direction, e_phi)

    Sigma = metric.Sigma(r_cam, theta_cam, a)
    Delta = metric.Delta(r_cam, a, M)
    A = metric.A_func(r_cam, theta_cam, a, M)
    alpha = metric.lapse(r_cam, theta_cam, a, M)
    omega = metric.frame_dragging_omega(r_cam, theta_cam, a, M)
    sin_theta = np.sin(theta_cam)

    Lz = n_phi * np.sqrt(A / Sigma) * sin_theta
    E = alpha + omega * Lz
    Q = Sigma * n_theta**2 - np.cos(theta_cam) ** 2 * (a**2 * E**2 - Lz**2 / sin_theta**2)

    dr0 = n_r * np.sqrt(Delta * Sigma)
    dtheta0 = n_theta * np.sqrt(Sigma)

    return E, Lz, Q, dr0, dtheta0
