# Kerr Black-Hole Raytracer — Design Plan

## Context

Direct sequel to `Schwarzschild_Raytracer/`: same "render a black hole"
project, now for a **rotating** black hole. This is a substantially
different numerical problem, not a small parameter extension --
Schwarzschild's whole architecture rests on spherical symmetry (every
photon's trajectory confined to a single plane through the center,
reducing the geodesic ODE to a clean 2D polar-orbit problem in `u=1/r`
vs. `phi`). Kerr is only **axisymmetric**, not spherically symmetric:
frame dragging pulls photons out of any fixed plane, so that reduction
does not exist here. The new payoff is worth it -- frame dragging, an
ergosphere, spin-dependent photon-sphere/ISCO radii, and a visibly
asymmetric (Doppler-boosted) disk image, none of which Schwarzschild can
show.

Same house style as `Schwarzschild_Raytracer/`: plain procedural
functions, `scipy.integrate.solve_ivp`, geometrized units `G=c=M=1`
(mass fixed at 1, spin parameter `a` in `[0, 1)` in the same units, `a=0`
reproducing Schwarzschild exactly -- the single most important
validation target throughout this whole project, since every Kerr
formula below was hand-derived and cross-checked against the already
-validated Schwarzschild equations at `a=0` before writing any code).

## The physics (derived and self-consistency-checked before implementation)

**Photon geodesics via the Carter constant, in Mino time.** A Kerr photon
has three conserved quantities: energy `E` (from `d/dt`), angular
momentum `L_z` (from `d/dphi`), and the Carter constant `Q` (from a
hidden second-rank Killing tensor -- with the null condition, `Q` is what
makes the `r` and `theta` equations separate into two independent 1D
problems instead of a coupled 2D one). Defining `P(r) = E(r^2+a^2) -
a*L_z`, `K = (L_z - aE)^2 + Q`, `Delta = r^2 - 2Mr + a^2`, `Sigma = r^2 +
a^2*cos^2(theta)`:

```
R(r)     = P(r)^2 - Delta*K
Theta(th) = Q + cos^2(th)*(a^2*E^2 - L_z^2/sin^2(th))
```

In ordinary affine parameter `lambda`: `Sigma*dr/dlambda = +-sqrt(R)`,
`Sigma*dtheta/dlambda = +-sqrt(Theta)`. Switching to **Mino time**
`tau` (`d(tau) = d(lambda)/Sigma`) cancels the shared `Sigma` factor and
makes `r` and `theta` fully independent ODEs:

```
dr/dtau     = +-sqrt(R(r))
dtheta/dtau = +-sqrt(Theta(theta))
dphi/dtau   = -(a*E - L_z/sin^2(th)) + a*P(r)/Delta(r)
```

**Second-order reformulation (this project's key numerical trick,
directly analogous to Schwarzschild's `d^2u/dphi^2 = 3Mu^2-u`)**:
differentiating `dr/dtau = sigma_r*sqrt(R)` once more with respect to
`tau` gives `d^2r/dtau^2 = (1/2)*dR/dr` -- the `sigma_r` sign and the
`sqrt(R)` singularity at turning points (`R=0`) both disappear
completely, exactly like Schwarzschild's second-order trick eliminated
the need to track `du/dphi`'s sign by hand. Same for `theta`:
`d^2theta/dtau^2 = (1/2)*dTheta/dtheta`. `geodesics.py` integrates the
five-variable first-order system `y = [r, dr/dtau, theta, dtheta/dtau,
phi]` built from these -- no explicit turning-point handling needed
anywhere, matching Schwarzschild's robustness.

**Validated by exact reduction to Schwarzschild at `a=0`, equatorial
(`Q=0`)**: substituting `a=0`, `theta=pi/2` into `R(r)` and the `phi`
equation and changing variables from Mino time to `phi` (`dr/dphi =
(dr/dtau)/(dphi/dtau)`, `u=1/r`) reproduces `Schwarzschild_Raytracer`'s
own validated equation `(du/dphi)^2 = 1/b^2 - u^2*(1-2Mu)` **exactly** --
worked by hand before any code was written, and re-verified numerically
in `Geodesic Validation.ipynb` by comparing a direct `a=0` Kerr
integration against `Schwarzschild_Raytracer.geodesics`'s own output for
the same ray.

**Camera setup via the ZAMO (zero-angular-momentum-observer) frame.**
Inside the ergosphere no observer can hold `(r,theta,phi)` fixed (frame
dragging forces even light to co-rotate), so Schwarzschild's "static
observer" camera tetrad doesn't generalize -- the standard replacement is
the locally-non-rotating ZAMO frame. With lapse `alpha = sqrt(Sigma*Delta/A)`,
shift `omega = 2*M*a*r/A` (`A = (r^2+a^2)^2 - a^2*Delta*sin^2(theta)`,
the frame-dragging angular velocity), and local photon energy normalized
to `E_hat=1`, a camera ray's local ZAMO-frame direction cosines
`(n_r, n_theta, n_phi)` (a unit 3-vector, obtained the same way
Schwarzschild's `camera.py` got `(radial, tangential)` components -- flat
-embedding spherical unit vectors coincide with the local orthonormal
tetrad once the metric's scale factors are accounted for, exactly as that
module's docstring argues) give the conserved quantities directly:

```
L_z = n_phi * sqrt(A/Sigma) * sin(theta)      [g_phiphi = A*sin^2(theta)/Sigma]
E   = alpha + omega*L_z                        [from E_hat = (E - omega*L_z)/alpha]
Q   = Sigma*n_theta^2 - cos^2(theta)*(a^2*E^2 - L_z^2/sin^2(theta))
```

with initial conditions `dr/dtau|_0 = n_r*sqrt(Delta*Sigma)`,
`dtheta/dtau|_0 = n_theta*sqrt(Sigma)` (both signed correctly with no
separate sign-tracking step, derived from `p_r_hat = n_r` and
`R = n_r^2 * Delta * Sigma` at the camera). All of this was worked
through by hand from the Kerr metric's ADM (lapse/shift) decomposition
before writing `camera.py`, for the same reason as the geodesic
derivation above: getting a sign or factor wrong here is easy and hard to
self-diagnose without an independent check, so the check (the `a=0`
reduction) was lined up first.

## File layout

```
General_Relativity/Kerr_Raytracer/Kerr_Raytracer_Plan.md   # this design document
General_Relativity/Kerr_Raytracer/metric.py                # Phase 1: Kerr metric quantities, horizon/ergosphere/photon-sphere/ISCO
General_Relativity/Kerr_Raytracer/geodesics.py              # Phase 2: Mino-time (r,theta,phi) photon geodesic integrator
General_Relativity/Kerr_Raytracer/camera.py                 # Phase 3: ZAMO-frame camera, ray -> (E, L_z, Q) mapping
General_Relativity/Kerr_Raytracer/scene.py                  # Phase 4: celestial sphere + spin-dependent ISCO disk (reused/adapted from Schwarzschild_Raytracer)
General_Relativity/Kerr_Raytracer/raytrace.py               # Phase 5: per-pixel + vectorized render loop
General_Relativity/Kerr_Raytracer/Geodesic Validation.ipynb # Phase 2 validation: a=0 reduction, equatorial photon sphere, weak-field deflection
General_Relativity/Kerr_Raytracer/Kerr_Raytracer.ipynb      # Phase 6: final renders across spins, prograde/retrograde asymmetry
General_Relativity/Kerr_Raytracer/Frame_Dragging.ipynb      # Phase 7: gyroscope precession (geodetic + Lense-Thirring), reusing the validated metric/Christoffel machinery, vs. Gravity Probe B
General_Relativity/Kerr_Raytracer/media/
```

## Phases

- [x] **Phase 1 — Metric core (`metric.py`)** ✅ done. `Sigma`, `Delta`, `A`, lapse `alpha`, frame-dragging `omega`; `r_horizon = M+sqrt(M^2-a^2)`; `r_ergo(theta) = M+sqrt(M^2-a^2*cos^2(theta))`; equatorial photon-sphere radii (prograde/retrograde, Bardeen 1973 closed form); ISCO radius (Bardeen-Press-Teukolsky 1972 closed form, prograde/retrograde). **Validated**: every formula reduces exactly to its already-validated Schwarzschild value at `a=0` (`r_horizon->2M`, `r_ergo->2M`, photon sphere -> `3M` both directions, ISCO -> `6M`), and at `a=0.9` gives physically sensible orderings (prograde photon sphere `1.558M` / ISCO `2.321M`, both smaller than Schwarzschild's `3M`/`6M`; retrograde `3.910M`/`8.717M`, both larger) plus correct extremal (`a->M`) limits (prograde photon sphere `->M`, retrograde `->4M`).
- [x] **Phase 2 — Geodesic integrator (`geodesics.py`)** ✅ done. The Mino-time second-order `(r, theta, phi)` system derived above.

  **A genuine bug found and fixed**: Boyer-Lindquist coordinates have a real coordinate singularity at the horizon (`dphi/dtau ~ 1/Delta -> infinity` as `Delta -> 0`) -- unlike `Schwarzschild_Raytracer.geodesics` (which sidesteps this entirely by using `phi` itself, not an affine-like parameter, as the independent variable), this project's Mino-time parameterization integrates `r`/`theta` *and* `phi` together, and `phi`'s blow-up forced the adaptive step size toward zero right at the horizon -- found as a genuine stall (a ray reported "trapped", sitting at `r` essentially equal to `r_horizon`, never firing the capture event within a generous `tau_max`). Fixed with a small horizon-crossing buffer (`horizon_buffer=1.001`, i.e. the capture event triggers at `1.001*r_horizon`, not exactly at the horizon) -- physically harmless, since a photon that close is captured for every practical rendering purpose regardless of the exact triggering radius.

  **Validated** in `Geodesic Validation.ipynb`, four independent checks: (a) `a=0` trajectories match `Schwarzschild_Raytracer.geodesics.integrate_ray` directly (not just formulas -- full trajectory comparison across 8 impact parameters spanning capture/escape, worst-case relative error `9.8e-4`, dominated by the small, expected horizon-buffer shift); (b) `metric.py`'s closed-form photon-sphere radius is a genuine double root (`R=0` *and* `dR/dr=0` simultaneously, to `~1e-11`-`1e-15`) of `geodesics.R_func` at `a=0.9`, for both prograde and retrograde -- two independently-implemented pieces of physics agreeing exactly; (c) integrating directly at that critical impact parameter produces genuine near-circular whirling at the predicted radius (`0.20%`/`0.047%` agreement, prograde/retrograde); (d) weak-field deflection converges cleanly toward the familiar `4M/b` limit as `b` grows (`1.160 -> 1.021` at `b=15` to `120`), spin-independent at leading order as expected.

  **Two validation-script bugs found and fixed along the way (in scratch testing, not the solver itself)**, worth recording since they nearly looked like solver bugs: a first deflection-angle attempt used a *different* radius for the escape threshold than the starting radius, silently invalidating the flat-space baseline formula `2*arccos(b/r0)` (which assumes symmetric start/end radii) -- the "excess" angle it showed wasn't GR deflection at all. A second attempt "fixed" this by pushing `r0` out to `1e6` (matching `Schwarzschild_Raytracer`'s own convention), which instead hit genuine numerical trouble at that scale specific to this Mino-time integrator (spurious "horizon" outcomes for rays nowhere near capture) -- not needed anyway, since no raytracer camera sits anywhere near that far out. A modest, self-consistent `r0` scale resolved both.
- [x] **Phase 3 — Camera (`camera.py`)** ✅ done. ZAMO tetrad, ray -> `(E, L_z, Q, dr0, dtheta0)`; `camera_position`/`camera_basis`/`pixel_directions` copied verbatim from Schwarzschild's `camera.py` (metric-independent, pure flat-R^3 pinhole geometry); `ray_setup` replaced entirely by the ZAMO mapping (Kerr rays have no single orbital plane to reduce to, unlike Schwarzschild's `(e1, e2)` plane basis). **Validated**: for an equatorial-confined ray direction (no `theta` component) at `a=0`, the Kerr camera's `(E, L_z)` gives `b=L_z/E` matching Schwarzschild's own `ray_setup` exactly (`0.0` relative error), with `Q=0`, `dtheta0=0`, and the radial-direction sign matching -- confirming the ZAMO-frame derivation (worked out by hand in this Plan's "The physics" section before writing any code) is correct.
- [x] **Phase 4 — Scene (`scene.py`)** ✅ done. `celestial_sphere_color` and `disk_color` copied verbatim from `Schwarzschild_Raytracer.scene` (metric-independent). `disk_crossing` reworked rather than copied: Schwarzschild's geodesic integrator tracks a single planar `(r, phi)` trajectory and has to reconstruct 3D Cartesian positions to find the disk plane; Kerr's integrator already tracks `(r, theta, phi)` directly in Mino time, so a disk crossing is just `theta` crossing `pi/2` (the global equatorial plane), and the disk radius at the crossing is simply the Boyer-Lindquist `r` there -- no embedding step needed, and actually simpler than Schwarzschild's own version. `r_isco` has no module-level default (unlike Schwarzschild's fixed `6M`): it's spin- and prograde/retrograde-dependent, computed via Phase 1's `metric.r_isco(a, M, prograde)` and passed in by the caller (`raytrace.py`).
- [x] **Phase 5 — Render loop (`raytrace.py`)** ✅ done. `render` is the per-pixel MVP (one `geodesics.integrate_ray` `solve_ivp` call per pixel). `render_fast` is a vectorized fixed-step RK4 over `(r, dr, theta, dtheta, phi)`, reusing `geodesics._dR_dr`/`_dTheta_dtheta`/`_dphi_dtau` directly as the derivative (these are already numpy-vectorized, so no re-derivation was needed) and batching `camera.ray_setup`'s formula over all pixels at once (valid since the camera position itself is fixed per render).

  **A genuine bug found and fixed**: the first version defaulted `render_fast` to `h=1e-2` (matching `Schwarzschild_Raytracer.raytrace`'s own default). Cross-checking against `render` at a small preview resolution showed large disagreement (mean abs pixel diff `0.14`, `54%` of pixels differing by more than `0.05`) plus `RuntimeWarning: overflow`. Isolating individual rays (`render` vs. a standalone single-ray `render_fast`-style integration) found the actual mechanism: for rays passing close to the critical impact parameter (the ones that matter most, since they trace the shadow boundary), `d^2r/dtau^2 = (1/2)dR/dr` is stiff enough near the true turning point that a fixed step of `h=1e-2` manufactures a *spurious* turning point well short of the real one (found `r_min=3.32` vs. the true `r_min=1.437`, both at `a=0.9`), after which the fixed-step RK4 diverges outward and eventually overflows to `inf`/`NaN` instead of correctly falling on to capture. A step-size sweep (`h` from `2e-2` down to `1e-4`) found `h<=5e-3` already recovers the correct outcome, `h=1e-3` matches `r_min` to `~1e-3` and was adopted as the default (`h=5e-4`/`1e-4` tighten further with no qualitative change). This is a genuinely stiffer equation than Schwarzschild's `du/dphi` system, which tolerates `h=1e-2` fine -- not a coding bug, a real numerical-stability difference between the two projects' governing equations.

  **Re-validated** at `h=1e-3`: `render` vs. `render_fast` on the same scene (`a=0.9`, off-equatorial camera, `48x27` preview) now agree with mean abs pixel diff `4.6e-4`, only `0.08%` of pixels differing by more than `0.05` (isolated shadow-boundary pixels, an expected finite-resolution disagreement, not a bug -- the boundary itself is a measure-zero curve). `render_fast` at `h=1e-3` was ~29x faster than `render` at this preview resolution (`1.8s` vs. `51s`) and remains practical at higher resolution (`400x225` in `~100s`), while `render`'s per-pixel `solve_ivp` cost would make that resolution impractical (would extrapolate to roughly an hour).
- [x] **Phase 6 — Validation & demo (`Kerr_Raytracer.ipynb`)** ✅ done. Five cells: (0) `render` vs. `render_fast` cross-check made a first-class notebook cell (not left in scratch testing) -- at `a=0.9`, off-equatorial camera, `48x27` preview: mean abs pixel diff `4.6e-4`, `0.08%` of pixels differing by `>0.05` (isolated shadow-boundary pixels), `render_fast` ~30x faster (`1.8s` vs. `52s`); (1) equatorial no-disk shadow across `a=[0, 0.5, 0.9, 0.998]` -- confirms the expected Kerr signature directly: the shadow visibly shrinks *and* deforms from a perfect circle into an asymmetric, flattened shape as spin increases, matching the prograde/retrograde photon-sphere split from Phase 1; (2) full lensed-disk scene at `a=0.9` (`r_cam=18M`, `theta_cam=80deg`, `400x225`) -- shows the classic strong-lensing "warped ring" (the disk's far side lensed up and over the shadow) plus a visible thin photon-ring sliver at the near edge; (3) prograde vs. retrograde disk at fixed `a=0.9`, same camera -- visually striking confirmation of the Phase 1 ISCO split (`2.32M` vs. `8.72M`): the prograde disk hugs the shadow closely with a small clean gap, while the retrograde disk's inner edge sits much farther out, leaving a large unlensed region between the shadow and the glow. All renders used `render_fast` at the validated `h=1e-3` default with a reduced `tau_max` (`80`-`150`, down from the `500` validation default) to bound worst-case per-pixel cost for whirling near-critical rays without affecting correctness (these rays are already indistinguishable from "horizon" at any practical resolution -- see Phase 5's writeup). Doppler-asymmetric disk coloring (stretch goal) was not attempted; flagged under Known limitations below.

  **One non-bug pitfall found during setup, worth recording**: an early test placed the camera exactly in the equatorial plane (`theta_cam=pi/2`) with the disk enabled and `r_cam` inside `r_out` -- i.e. the camera embedded in the disk's own plane. This caused a near-hang (many rays' `theta` hovers almost exactly at `pi/2` with `cos(theta)` oscillating near zero from floating-point noise, so `disk_crossing`'s sign-change test barely fires, and the fixed `tau_max=500`, `h=1e-3` loop grinds through hundreds of thousands of iterations for the whole image before those rays resolve). Not a solver bug: `Schwarzschild_Raytracer`'s own Phase 6 notebook independently avoids this exact configuration (shallow-angle camera for the disk shot, `theta_cam=pi/2` only for the no-disk lensing shot), and the same convention was adopted here once the cause was understood.
- [ ] **Phase 7 — Frame dragging (`Frame_Dragging.ipynb`)**. A gyroscope's spin 4-vector, parallel-transported along a circular timelike geodesic orbit (reusing this project's validated Kerr Christoffel-symbol machinery), precesses per orbit -- both the special-relativistic-adjacent geodetic (de Sitter) precession and the genuinely frame-dragging-driven Lense-Thirring precession. **Validation**: compare the computed precession rates against the real Gravity Probe B measurement (geodetic: ~6.6 arcsec/year; frame-dragging: ~0.039 arcsec/year, at Earth's actual mass/spin/orbit parameters rescaled into this project's geometrized units).

## Known limitations (anticipated, to confirm/document once built)

- **No Doppler/gravitational redshift disk coloring** by default (same limitation `Schwarzschild_Raytracer` flagged and left as a future extension) -- worth revisiting here specifically, since Kerr disks are famous for visibly asymmetric brightness from relativistic beaming; attempted as a stretch goal in Phase 6, not guaranteed.
- **Equatorial-plane accretion disk only** -- a tilted/precessing disk is a real possible extension, out of scope here.
- **No general polarization/redshift spectral rendering** -- purely a temperature-colormap disk, same as Schwarzschild.

## Verification

1. Run `Geodesic Validation.ipynb`: confirm the `a=0` reduction matches `Schwarzschild_Raytracer` directly (not just its formulas -- literally compare trajectories), confirm equatorial photon-sphere/ISCO formulas against Phase 1, confirm weak-field deflection.
2. Render a preview (grid celestial sphere, no disk) at a few spins and visually confirm the shadow shrinks/shifts as expected with increasing spin, and that `a=0` looks like `Schwarzschild_Raytracer`'s own render.
3. Add the disk and re-render; visually confirm the lensed disk image, and (if attempted) the Doppler brightness asymmetry.
4. Run `Frame_Dragging.ipynb` and confirm the computed geodetic/Lense-Thirring precession rates land in the right ballpark of the real Gravity Probe B measurement.
