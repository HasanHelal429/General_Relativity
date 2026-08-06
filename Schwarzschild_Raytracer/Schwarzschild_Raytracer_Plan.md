# Schwarzschild Black-Hole Raytracer — Design Plan

## Context

The user wants to start a new umbrella project, `General_Relativity/`, for modeling GR effects, and wants the first concrete build to be a raytracer through a Schwarzschild spacetime (a static, non-rotating black hole) — the classic "render a black hole" project: bend light along null geodesics instead of straight lines, and see gravitational lensing, the photon sphere, the black-hole shadow, and (optionally) a lensed accretion disk.

There is no existing GR/relativity code anywhere in the repo (`3_Body_Orbit_Phase_Space.ipynb` is the closest orbital-mechanics analog but is purely Newtonian, `solve_ivp`-based). The closest structural precedent is `Quantum Mechanics/HF_solver/`: a self-contained project folder with its own `*_Plan.md`, several small importable `.py` modules (one per phase of the physics), validation notebooks, and a `media/` folder for saved PNGs/MP4s — used precisely because "solver machinery is substantial and reused across notebooks." That's the right template here too, since the raytracer needs a geodesic integrator, a camera model, and a scene model all reused across a validation notebook and a final render notebook.

House style to match: plain procedural functions (no classes), `numpy`/`scipy`/`matplotlib` header, `scipy.integrate.solve_ivp` for ODE work (already the repo's standard, e.g. orbit integration in `3_Body_Orbit_Phase_Space.ipynb`), `scienceplots` styling where convenient (optional, skip if unavailable rather than block on it), animations via `matplotlib.animation.FuncAnimation` saved with the Pillow/ffmpeg writer. Run/test with `torch.venv`'s Python (already has numpy/scipy/matplotlib). No numba/jax/cupy — not used anywhere else in the repo; if the naive per-pixel Python loop is too slow at full resolution, the fallback is a hand-vectorized fixed-step RK4 over all pixels as numpy arrays (Phase 5), not a new dependency.

Units: geometrized, `G = c = 1`, and mass is fixed at `M = 1` (a length scale) throughout, so the horizon is at `r = 2`, the photon sphere at `r = 3`, and the critical impact parameter (black-hole shadow radius) at `b_crit = 3*sqrt(3) ≈ 5.196`.

**Key physics trick that keeps this tractable**: Schwarzschild is spherically symmetric, so every photon's full 3-vector angular momentum is conserved — its trajectory is confined to a single plane through the center, whatever that plane's orientation. So instead of integrating the full 4D geodesic equations in `(t, r, θ, φ)`, each ray is reduced to a clean 2D polar-orbit ODE in *its own* orbital plane (in terms of `u = 1/r` vs. the in-plane angle `φ`), and the 2D result is rotated back into 3D using that plane's orientation (determined by the ray's starting position and initial direction). This is the standard efficient construction used in real black-hole raytracers and avoids ever touching `θ` directly.

## File layout

```
General_Relativity/Schwarzschild_Raytracer/Schwarzschild_Raytracer_Plan.md   # this design document
General_Relativity/Schwarzschild_Raytracer/metric.py               # Phase 1: Schwarzschild metric quantities, horizon/photon-sphere/b_crit
General_Relativity/Schwarzschild_Raytracer/geodesics.py            # Phase 2: planar photon-orbit ODE + integrator + deflection angle
General_Relativity/Schwarzschild_Raytracer/camera.py               # Phase 3: pinhole camera, local static-observer frame, ray -> (b, plane) mapping
General_Relativity/Schwarzschild_Raytracer/scene.py                # Phase 4: celestial sphere texture, event horizon, accretion disk
General_Relativity/Schwarzschild_Raytracer/raytrace.py             # Phase 5: per-pixel render loop + vectorized RK4 fast path
General_Relativity/Schwarzschild_Raytracer/Geodesic Validation.ipynb   # Phase 2 validation: deflection angle, photon sphere, conservation
General_Relativity/Schwarzschild_Raytracer/Schwarzschild_Raytracer.ipynb  # Phase 6: final renders, lensing/shadow demo, media export
General_Relativity/Schwarzschild_Raytracer/media/                  # saved PNGs/MP4s from every phase
```

Future GR-effects projects (e.g. a Kerr raytracer, gravitational-wave chirp modeling) would live as sibling folders under `General_Relativity/`, each with its own plan — not built now, just why the umbrella folder exists.

## Phase 1 — Metric core (`metric.py`)

- `f(r, M=1)` — the Schwarzschild lapse-squared function `1 - 2*M/r`.
- `r_horizon(M=1)`, `r_photon_sphere(M=1)` (`= 3*M`), `b_critical(M=1)` (`= 3*sqrt(3)*M`) — small closed-form helpers used everywhere downstream for validation and for the "did this ray fall in" cutoff.
- Docstring derivation of the photon effective potential `V_eff(r) = f(r)/r**2` and why circular photon orbits only exist at `r = 3M` (max of `V_eff`, hence unstable) — this is what Phase 2's validation checks against.

## Phase 2 — Planar photon geodesic integrator (`geodesics.py`)

- The governing ODE, derived from the two photon conservation laws (`E`, `L`, with `b = L/E`) and standard for null geodesics in Schwarzschild: `d²u/dφ² = 3*M*u² - u`, where `u = 1/r`. Implemented as a first-order system `y = [u, du/dφ]` for `solve_ivp`.
- `integrate_ray(r0, phi0, b, sign0, M=1, phi_max=50*pi)`: integrates from an initial `(r0, φ0)` and initial radial direction (`sign0 = ±1`, since `du/dφ` at the start is `±sqrt(1/b**2 - u0**2*f(u0))`), using `solve_ivp` with **terminal events**: `u` reaching `1/r_horizon` (ray captured) or `u` dropping to `~0` (ray escapes to infinity). Returns an outcome tag (`"horizon"` / `"escaped"`) plus the full `(u, φ)` trajectory (needed later for accretion-disk crossing checks) and, for escaped rays, the total accumulated `φ`.
- `deflection_angle(b, M=1)`: for `b > b_crit`, integrate a scattering orbit from `r = ∞` and return `Δφ - π` (the bending angle).
- **Validation** (done in `Geodesic Validation.ipynb`, not the module itself): (a) for `b >> b_crit`, `deflection_angle(b)` matches the weak-field formula `4*M/b` to a few %, tightening as `b` grows; (b) a ray launched at `b` just above `b_crit` orbits many times near `r=3M` before escaping (unstable photon sphere behavior), and `b` just below `b_crit` falls into the horizon — bracketing `b_crit` numerically should converge to `metric.b_critical()`; (c) spot-check that `b` computed from a trajectory's own `(u, du/dφ)` stays constant along the numerical path (conservation check on the integrator itself).

## Phase 3 — Camera & ray-to-plane mapping (`camera.py`)

- `local_frame(r_cam, M=1)`: orthonormal tetrad for a static observer hovering at `r_cam` in Schwarzschild coordinates (accounts for the `sqrt(f(r))` redshift factor between coordinate and locally-measured quantities) — this is what makes pixel angles physically correct rather than just coordinate angles.
- `pixel_directions(resolution, fov_deg)`: standard pinhole-camera unit vectors in the *local* frame for each pixel of an image plane.
- `ray_setup(direction_local, r_cam, M=1)`: for a given local ray direction, compute (1) the impact parameter `b = r_cam * sin(psi) / sqrt(f(r_cam))` where `psi` is the angle from the outward radial direction, and (2) the initial in-plane radial sign, and (3) the 3D orbital-plane basis (`e1` = radial unit vector at the camera, `e2` = in-plane perpendicular vector) used to later rotate the 2D `(r, φ)` result from Phase 2 back into a 3D direction/position.
- Ties directly into Phase 2: for each pixel, `ray_setup` produces exactly the `(b, sign0)` that `geodesics.integrate_ray` needs, plus the rotation basis to interpret the result in 3D.

## Phase 4 — Scene model (`scene.py`)

- `celestial_sphere_color(direction)`: procedural equirectangular texture (lat/long checkerboard/grid, optionally a simple procedural starfield) sampled by a final escaped-ray 3D direction — chosen over a photographic texture first because a grid makes lensing distortion immediately legible; a real starfield/skybox image can be swapped in later without touching the raytracer.
- `disk_color(r)`: thin equatorial accretion disk between `r_isco = 6*M` and a chosen `r_out`, colored by a simple non-relativistic temperature profile `T(r) ∝ r**(-3/4)` mapped through a matplotlib colormap (e.g. `inferno`) — flagged as an approximation; true Doppler/gravitational redshift coloring is called out as an optional Phase 6 stretch extension, not required for the core image.
- `disk_crossing(trajectory_3d)`: given a ray's 3D path (reconstructed from Phase 2's in-plane trajectory via Phase 3's basis), find whether/where it crosses the global equatorial plane (`z=0`) within `[r_isco, r_out]`.
- `horizon_color = black`.

## Phase 5 — Render loop (`raytrace.py`)

- `render(resolution, r_cam, M=1, fov_deg=60, disk=True)`: MVP is a straightforward per-pixel Python loop wiring Phases 2-4 together (camera → integrate → classify hit → color), returning an RGB array. Built and correctness-checked first at a small preview resolution (e.g. 240×135).
- **Performance pass** (only after the per-pixel version is verified correct): replace the per-pixel `solve_ivp` calls with a hand-written fixed-step vectorized RK4 that advances *all* pixels' `(u, du/dφ)` states together as `(N, 2)` numpy arrays, with per-ray early-exit masks for horizon/escape — this is what makes a full-resolution (e.g. 960×540+) render tractable in pure numpy without adding a new dependency. Cross-validated against the `solve_ivp` MVP on the preview resolution before trusting it at full res.

## Phase 6 — Validation & demo (`Schwarzschild_Raytracer.ipynb`)

- Renders the core "money shot": lensed grid-textured celestial sphere + black-hole shadow (should visually match the analytic shadow angular size from `b_crit`), then the same scene with the accretion disk added (showing the characteristic lensed/doubled disk image above and below the horizon).
- A quantitative validation plot: numeric `deflection_angle(b)` vs. impact parameter, overlaid with the analytic weak-field asymptote `4M/b`, with the photon-sphere divergence at `b_crit` visible.
- Optional stretch (only if time permits, not required for the project to be "done"): an orbiting-camera flythrough animation exported to `media/`, and/or relativistic Doppler+gravitational redshift coloring of the disk instead of the flat temperature-profile approximation.
- All representative renders/plots saved to `media/`.

## Progress

- [ ] Phase 1 — `metric.py`
- [ ] Phase 2 — `geodesics.py` + `Geodesic Validation.ipynb`
- [ ] Phase 3 — `camera.py`
- [ ] Phase 4 — `scene.py`
- [ ] Phase 5 — `raytrace.py` (correctness pass, then vectorized performance pass)
- [ ] Phase 6 — `Schwarzschild_Raytracer.ipynb` + `media/`

## Verification

1. Run `Geodesic Validation.ipynb`: confirm weak-field deflection matches `4M/b`, confirm photon-sphere bracketing converges to `b_crit = 3*sqrt(3)*M`, confirm the impact parameter stays numerically constant along an integrated trajectory.
2. Render a small preview image (grid celestial sphere + horizon only, no disk) and visually confirm an Einstein-ring-like lensing pattern and a shadow whose angular size matches the analytic prediction for the chosen `r_cam`.
3. Add the accretion disk and re-render; visually confirm the lensed/warped disk image (light from behind the hole appearing above/below it).
4. Time a full-resolution render with the vectorized Phase 5 integrator and confirm it matches the per-pixel MVP's output on the preview resolution before trusting it at full res.
