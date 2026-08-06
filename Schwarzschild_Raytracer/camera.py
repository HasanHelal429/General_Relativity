"""
Pinhole camera and ray-to-orbital-plane mapping.

The camera can sit at any Schwarzschild position (r_cam, theta_cam, phi_cam)
-- there is no "WLOG equatorial" gauge-fixing here, because a fixed feature
of the scene (the accretion disk, Phase 4) lives at the *global* equatorial
plane theta=pi/2, so a camera's polar angle relative to that plane is
physically meaningful (e.g. looking down at the disk from above).

Positions/directions are embedded in ordinary flat R^3 via the standard
spherical-to-Cartesian map purely as visualization/bookkeeping coordinates.
The key fact that makes this rigorous rather than approximate: the three
spherical unit vectors (e_r, e_theta, e_phi) at a point are *exactly* the
Schwarzschild static observer's local orthonormal (proper-length) spatial
tetrad there -- e_r has proper length dr/sqrt(f(r)) per coordinate dr, but
as a *unit* vector it's still just the ordinary radial unit vector; the
sqrt(f(r)) only shows up when relating a coordinate increment to a proper
length, which is exactly where it appears below, in the impact-parameter
formula. So decomposing a unit ray direction into "radial component" +
"orthogonal (tangential) remainder" via plain dot/cross products in the
R^3 embedding gives the same numbers as doing it explicitly in the local
proper tetrad.
"""

import numpy as np

import metric


def camera_position(r_cam, theta_cam=np.pi / 2, phi_cam=0.0):
    """Camera position in the flat R^3 embedding."""
    st, ct = np.sin(theta_cam), np.cos(theta_cam)
    sp, cp = np.sin(phi_cam), np.cos(phi_cam)
    return r_cam * np.array([st * cp, st * sp, ct])


def camera_basis(position, up_hint=(0.0, 0.0, 1.0)):
    """
    Standard look-at basis for a camera at `position` looking toward the
    origin (the black hole), with `up_hint` fixing the roll (need not be
    exactly perpendicular to the look direction).
    """
    position = np.asarray(position, dtype=float)
    forward = -position / np.linalg.norm(position)
    up_hint = np.asarray(up_hint, dtype=float)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def pixel_directions(resolution, fov_deg, forward, right, up):
    """
    Unit ray direction (in the flat R^3 embedding) for every pixel of a
    pinhole camera image plane. `fov_deg` is the horizontal field of view;
    the vertical field of view follows from the image aspect ratio.

    Returns an array of shape (ny, nx, 3).
    """
    nx, ny = resolution
    fov_x = np.radians(fov_deg)
    half_w = np.tan(fov_x / 2.0)
    half_h = half_w * ny / nx

    u = np.linspace(-half_w, half_w, nx)
    v = np.linspace(half_h, -half_h, ny)  # image row 0 = top = +v
    uu, vv = np.meshgrid(u, v)  # each shape (ny, nx)

    directions = (
        forward[None, None, :]
        + uu[:, :, None] * right[None, None, :]
        + vv[:, :, None] * up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions


def ray_setup(direction, position, M=metric.M_DEFAULT):
    """
    For a unit ray direction leaving the camera at `position`, compute the
    impact parameter b, the initial radial sign (+1 outward, -1 inward, for
    geodesics.integrate_ray), and an orthonormal (e1, e2) basis for the
    ray's orbital plane (e1 = radial direction at the camera, e2 = in-plane
    tangential direction), used to map the planar (r, phi) trajectory back
    into 3D via plane_to_3d.

    A ray aimed exactly radially (tangential component ~0, b ~ 0) has no
    well-defined orbital plane; e2 is returned as a zero vector in that
    case (the trajectory stays on the e1 axis regardless of e2).
    """
    position = np.asarray(position, dtype=float)
    r_cam = np.linalg.norm(position)
    e1 = position / r_cam

    a = np.dot(direction, e1)
    tangential_vec = direction - a * e1
    tangential_mag = np.linalg.norm(tangential_vec)

    b = r_cam * tangential_mag / np.sqrt(metric.f(r_cam, M))
    sign0 = 1.0 if a >= 0.0 else -1.0
    e2 = tangential_vec / tangential_mag if tangential_mag > 1e-12 else np.zeros(3)

    return b, sign0, e1, e2


def plane_to_3d(r, phi, e1, e2):
    """Map in-plane polar coordinates (r, phi) back to 3D via basis (e1, e2)."""
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return r[..., None] * np.cos(phi)[..., None] * e1 + r[..., None] * np.sin(phi)[..., None] * e2
