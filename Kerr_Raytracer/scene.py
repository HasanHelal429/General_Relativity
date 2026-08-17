"""
Scene model: celestial sphere background, event horizon, and a thin
equatorial accretion disk.

`celestial_sphere_color` and `disk_color` are copied verbatim from
Schwarzschild_Raytracer.scene (metric-independent -- not imported
cross-folder, per this repo's self-contained-project convention).

`disk_crossing` is reworked: Schwarzschild_Raytracer's geodesic
integrator tracks a single planar (r, phi) trajectory and has to
reconstruct 3D Cartesian positions to find the z=0 disk plane. Kerr's
integrator (geodesics.py) already tracks (r, theta, phi) directly in
Mino time, so a disk crossing is just theta crossing pi/2 (the global
equatorial plane, same convention as Schwarzschild) -- no
embedding/Cartesian step needed, and the disk radius at the crossing is
simply the Boyer-Lindquist r there.

r_isco has no module-level default here (unlike Schwarzschild's fixed
6M): it depends on spin and orbital sense, via metric.r_isco(a, M,
prograde) from Phase 1, and is expected to be passed in explicitly by
the caller (raytrace.py).
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib import colormaps

import metric

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


def disk_color(r, r_isco, r_out=R_OUT_DEFAULT, cmap_name="inferno"):
    """
    Color for accretion-disk emission at Boyer-Lindquist radius r, from a
    simple non-relativistic thin-disk temperature profile T(r) ~
    r**(-3/4) (Doppler/gravitational redshift coloring -- which would
    make prograde vs. retrograde disks visibly asymmetric near a spinning
    hole -- is a possible later extension, not included here). Hottest
    (brightest) at r_isco, coolest at r_out. `r` may be an array; returns
    an array (..., 3) of RGB colors.
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


def disk_crossing(theta, r, r_isco, r_out=R_OUT_DEFAULT):
    """
    Given a ray's theta(tau) and Boyer-Lindquist r(tau) arrays (in
    propagation order from the camera), find the first point where it
    crosses the global equatorial plane theta=pi/2 with r in [r_isco,
    r_out].

    Returns the crossing radius r, or None if there is no such crossing.
    """
    c = np.cos(theta)
    for i in range(len(c) - 1):
        c0, c1 = c[i], c[i + 1]
        if c0 == 0.0:
            r0 = r[i]
            if r_isco <= r0 <= r_out:
                return r0
            continue
        if c0 * c1 < 0.0:
            t = c0 / (c0 - c1)
            r_point = r[i] + t * (r[i + 1] - r[i])
            if r_isco <= r_point <= r_out:
                return r_point
    return None
