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

## Phase 1 — Metric core (`metric.py`) ✅ done

- `f(r, M=1)` — the Schwarzschild lapse-squared function `1 - 2*M/r`.
- `r_horizon(M=1)`, `r_photon_sphere(M=1)` (`= 3*M`), `b_critical(M=1)` (`= 3*sqrt(3)*M`) — small closed-form helpers used everywhere downstream for validation and for the "did this ray fall in" cutoff.
- Docstring derivation of the photon effective potential `V_eff(r) = f(r)/r**2` and why circular photon orbits only exist at `r = 3M` (max of `V_eff`, hence unstable) — this is what Phase 2's validation checks against.
- **Validated**: `v_eff(r)` numerically maximized at `r=3.0` to 4 significant figures, and `1/sqrt(max(v_eff))` matches `b_critical()` to ~1e-8.

## Phase 2 — Planar photon geodesic integrator (`geodesics.py`) ✅ done

- The governing ODE, derived from the two photon conservation laws (`E`, `L`, with `b = L/E`) and standard for null geodesics in Schwarzschild: `d²u/dφ² = 3*M*u² - u`, where `u = 1/r`. Implemented as a first-order system `y = [u, du/dφ]` for `solve_ivp`.
- `integrate_ray(r0, phi0, b, sign0, M=1, phi_max=50*pi)`: integrates from an initial `(r0, φ0)` and initial radial direction (`sign0 = ±1`, since `du/dφ` at the start is `±sqrt(1/b**2 - u0**2*f(u0))`), using `solve_ivp` with **terminal events**: `u` reaching `1/r_horizon` (ray captured) or `u` dropping to `~0` (ray escapes to infinity). Returns an outcome tag (`"horizon"`/`"escaped"`/`"trapped"`) plus the full `(u, up, φ)` trajectory (`up` added beyond the original plan — needed for a clean conservation check without differentiating a non-uniform grid) and the total accumulated `φ`.
- `deflection_angle(b, M=1, r0=1e6)`: integrates a scattering orbit from a large-but-finite `r0` (a stand-in for infinity) back out to `r0`, and returns `Δφ_total - baseline`.
  - **Deviation from the original plan**: subtracting `π` (the `r0 -> ∞` baseline) instead of the *exact* finite-`r0` baseline `2*arccos(b/r0)` was numerically wrong — the finite-`r0` bias scales as `b/r0` while the true GR deflection scales as `M/b`, so for `b` not tiny compared to `r0` the bias overwhelmed the signal (confirmed empirically: the numeric/analytic ratio at `b=1000M`, `r0=1e6M` was 0.50, not ~1). Fixed by subtracting the exact finite-`r0` flat-space baseline instead of `π`; ratio then converges cleanly to 1.0 as `b` grows (0.9994 at `b=5000M`).
- **Validated** in `Geodesic Validation.ipynb`: (a) numeric/analytic deflection ratio converges to 1.0 for `b >> b_crit` (1.0006 at `b=5000M`); (b) bisection on `b` for an inward-aimed ray converges to `b_crit` to ~1e-10 relative error; (c) impact parameter recovered from `(u, up)` stays constant to ~1e-7 relative error along an integrated trajectory; (d) an orbit diagram across `b_crit` visually confirms captured vs. escaped rays whirl near `r=3M` as expected, more tightly the closer `b` is to `b_crit`.

## Phase 3 — Camera & ray-to-plane mapping (`camera.py`) ✅ done

- **Deviation from the original plan**: no separate `local_frame`/tetrad function. The radial unit vector `e1 = position / |position|` is all that's needed — decomposing a ray direction into "radial component" + "orthogonal remainder" via plain dot/cross products in the flat R^3 embedding gives the same numbers as doing it explicitly against the local proper tetrad `(e_r, e_theta, e_phi)` (they coincide; see `camera.py`'s module docstring for why). This simplified `ray_setup` to not need `e_theta`/`e_phi` at all.
- `camera_position(r_cam, theta_cam, phi_cam)`: camera position in the flat R^3 embedding, at *any* Schwarzschild `(r, theta, phi)` — not gauge-fixed to the equator, since the disk (Phase 4) fixes a preferred global equatorial plane, so a camera's polar angle relative to it is physically meaningful.
- `camera_basis(position, up_hint)`: standard look-at `(forward, right, up)` basis, looking toward the origin.
- `pixel_directions(resolution, fov_deg, forward, right, up)`: pinhole-camera unit ray direction per pixel.
- `ray_setup(direction, position, M=1)`: impact parameter `b = r_cam*sin(psi)/sqrt(f(r_cam))`, initial radial sign, and orbital-plane basis `(e1, e2)`.
- `plane_to_3d(r, phi, e1, e2)`: maps Phase 2's planar `(r, φ)` trajectory back into 3D.
- **Validated**: basis orthogonality to float precision; central (near-radial) pixel gives `b≈0`; `plane_to_3d(r_cam, 0, e1, e2)` exactly reproduces the camera position; a 15-ray fan from a tilted camera, integrated through `geodesics.integrate_ray` and reconstructed via `plane_to_3d`, visually whirls near the photon sphere and correctly separates captured (red) vs. escaped (blue) — see `media/camera_ray_fan.png`.

## Phase 4 — Scene model (`scene.py`) ✅ done

- `celestial_sphere_color(direction)`: procedural lat/lon checkerboard with hue set by longitude (a rainbow tint) — chosen over a plain grid because it makes lensing distortion (and which sky region maps where) immediately legible; a photographic skybox can be swapped in later without touching the raytracer.
- `disk_color(r, r_isco=6M, r_out=20M)`: thin equatorial accretion disk colored by a non-relativistic `T(r) ~ r**(-3/4)` profile through the `inferno` colormap — flagged as an approximation; true Doppler/gravitational redshift coloring remains a possible future extension.
- `disk_crossing(positions, r_isco, r_out)`: scans a ray's 3D trajectory in order for the first `z=0` crossing with radius in `[r_isco, r_out]`, linearly interpolating the exact crossing point.
- `HORIZON_COLOR = black`.
- **Validated**: the celestial texture's equirectangular unwrap and the disk colormap strip both look as intended (`media/celestial_sphere_texture.png`, `media/disk_temperature_profile.png`); `disk_crossing` correctly distinguishes a plane-crossing inside the disk annulus, one inside `r_isco` (correctly ignored), and no crossing at all.

## Phase 5 — Render loop (`raytrace.py`) ✅ done

- `render(resolution, r_cam, theta_cam, phi_cam, M=1, fov_deg=60, disk=True)`: per-pixel MVP wiring Phases 2-4 together. Correct, but ~11 ms/pixel (one `solve_ivp` call each) — a 240x135 preview would take minutes.
- `render_fast(...)`: same signature, but replaces per-pixel `solve_ivp` with a hand-written fixed-step vectorized RK4 advancing every pixel's `(u, du/dφ)` state together as `(N,)` numpy arrays (disk-crossing checked incrementally, vectorized, each step). ~50x faster (0.17s vs. 9.4s for a 40x22 cross-check render); a 320x180 render takes ~10s.
- **Deviation from the original plan**: `scene.disk_color` needed a fix for scalar radius input (matplotlib colormaps return a plain tuple, not an array, when called on a scalar) — added `np.atleast_1d`/unwrap handling.
- **Validated**: `render_fast` cross-checked against `render` at a shared preview resolution — they agree everywhere except a thin band right at the photon-ring/shadow edge (`media/mvp_vs_fast_diff.png`), which is expected, not a bug: `r=3M` is an *unstable* photon orbit, so nearby trajectories there are supposed to diverge chaotically under any numerical perturbation.

## Phase 6 — Validation & demo (`Schwarzschild_Raytracer.ipynb`) ✅ done

- Rendered the core "money shot": lensed grid-textured celestial sphere + shadow with no disk (`media/render_lensing_only.png`) — measured shadow angular radius matches the analytic `arcsin(b_crit*sqrt(f(r_cam))/r_cam)` prediction to within one pixel (ratio 1.000); then the full scene with the accretion disk added (`media/render_with_disk.png`), showing the characteristic lensed/warped disk arc above the shadow in addition to the near side below.
- Quantitative validation plot: full `deflection_angle(b)` curve down to `b` just above `b_crit`, showing the expected logarithmic divergence (`media/deflection_full_curve.png`).
- Stretch goal completed: a 20-frame orbiting-camera flythrough animation (`media/orbit_flythrough.gif`). Relativistic Doppler/redshift disk coloring was not done (flagged in Phase 4 as a possible future extension, not required here).
- All representative renders/plots saved to `media/`.

## Progress

- [x] Phase 1 — `metric.py`
- [x] Phase 2 — `geodesics.py` + `Geodesic Validation.ipynb`
- [x] Phase 3 — `camera.py`
- [x] Phase 4 — `scene.py`
- [x] Phase 5 — `raytrace.py` (correctness pass, then vectorized performance pass)
- [x] Phase 6 — `Schwarzschild_Raytracer.ipynb` + `media/`

## Verification

1. Run `Geodesic Validation.ipynb`: confirm weak-field deflection matches `4M/b`, confirm photon-sphere bracketing converges to `b_crit = 3*sqrt(3)*M`, confirm the impact parameter stays numerically constant along an integrated trajectory.
2. Render a small preview image (grid celestial sphere + horizon only, no disk) and visually confirm an Einstein-ring-like lensing pattern and a shadow whose angular size matches the analytic prediction for the chosen `r_cam`.
3. Add the accretion disk and re-render; visually confirm the lensed/warped disk image (light from behind the hole appearing above/below it).
4. Time a full-resolution render with the vectorized Phase 5 integrator and confirm it matches the per-pixel MVP's output on the preview resolution before trusting it at full res.
