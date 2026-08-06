"""
Scene model: celestial sphere background, event horizon, and a thin
equatorial accretion disk.

The accretion disk fixes a preferred global plane (Schwarzschild
theta=pi/2, i.e. Cartesian z=0 in the embedding used throughout this
project) between an inner edge at the innermost stable circular orbit
r_isco = 6M and a chosen outer edge r_out.
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib import colormaps

import metric

R_ISCO_DEFAULT = 6.0 * metric.M_DEFAULT
R_OUT_DEFAULT = 20.0 * metric.M_DEFAULT

HORIZON_COLOR = np.array([0.0, 0.0, 0.0])


def celestial_sphere_color(direction, n_lon=24, n_lat=12):
    """
    Procedural equirectangular background texture sampled by a 3D unit
    direction: a lat/lon checkerboard (brightness) with hue set by
    longitude (a rainbow tint), chosen so that gravitational lensing's
    warping/duplication of the sky is immediately legible in a render --
    a photographic skybox can be swapped in later without touching the
    raytracer.

    `direction` may have any shape (..., 3); returns an array (..., 3) of
    RGB colors in [0, 1].
    """
    direction = np.asarray(direction, dtype=float)
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]

    lon = np.arctan2(y, x)  # [-pi, pi]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))  # [-pi/2, pi/2]

    lon_cell = np.floor((lon + np.pi) / (2 * np.pi) * n_lon)
    lat_cell = np.floor((lat + np.pi / 2) / np.pi * n_lat)
    checker = np.mod(lon_cell + lat_cell, 2)
    brightness = np.where(checker > 0.5, 0.95, 0.25)

    hue = (lon + np.pi) / (2 * np.pi)
    saturation = 0.55 * np.ones_like(hue)
    hsv = np.stack([hue, saturation, brightness], axis=-1)
    return hsv_to_rgb(hsv)


def disk_color(r, r_isco=R_ISCO_DEFAULT, r_out=R_OUT_DEFAULT, cmap_name="inferno"):
    """
    Color for accretion-disk emission at radius r, from a simple
    non-relativistic thin-disk temperature profile T(r) ~ r**(-3/4)
    (Doppler/gravitational redshift coloring is a possible later
    extension, not included here). Hottest (brightest) at r_isco, coolest
    at r_out. `r` may be an array; returns an array (..., 3) of RGB colors.
    """
    r = np.asarray(r, dtype=float)
    scalar_input = r.ndim == 0
    r = np.atleast_1d(r)

    temperature = r ** (-0.75)
    t_in = r_isco ** (-0.75)
    t_out = r_out ** (-0.75)
    t_norm = np.clip((temperature - t_out) / (t_in - t_out), 0.0, 1.0)
    cmap = colormaps[cmap_name]
    rgb = cmap(t_norm)[..., :3]
    return rgb[0] if scalar_input else rgb


def disk_crossing(positions, r_isco=R_ISCO_DEFAULT, r_out=R_OUT_DEFAULT):
    """
    Given a ray's 3D trajectory `positions` (shape (N, 3), in propagation
    order from the camera), find the first point where it crosses the
    global equatorial plane (z=0) with radius in [r_isco, r_out].

    Returns the crossing point (3,) array, or None if there is no such
    crossing.
    """
    z = positions[:, 2]
    for i in range(len(z) - 1):
        z0, z1 = z[i], z[i + 1]
        if z0 == 0.0:
            r0 = np.linalg.norm(positions[i])
            if r_isco <= r0 <= r_out:
                return positions[i]
            continue
        if z0 * z1 < 0.0:
            t = z0 / (z0 - z1)
            point = positions[i] + t * (positions[i + 1] - positions[i])
            r_point = np.linalg.norm(point)
            if r_isco <= r_point <= r_out:
                return point
    return None
