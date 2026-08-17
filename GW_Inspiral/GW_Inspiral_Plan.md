# Post-Newtonian Binary Inspiral & Gravitational-Wave Waveform — Design Plan

## Context

The third of three deliverables from "do everything but the cosmology
demo" (alongside `Kerr_Raytracer/`), and the only one not extending an
existing raytracer: a two-body **post-Newtonian (PN) inspiral**
simulation. Two compact objects orbit under Newtonian gravity plus the
leading-order (2.5PN) **radiation-reaction** force -- the whole reason
"inspiral" happens at all: gravitational-wave emission carries away
orbital energy, the separation shrinks, the orbital frequency chirps
upward, until the PN approximation itself breaks down near merger. The
payoff is the actual `h+`/`h×` strain waveform this radiates, the same
kind of signal LIGO detects.

Same house style as the rest of `General_Relativity/`: plain procedural
functions, geometrized units (`G=c=1`), physics derived and
cross-validated against independent references before trusting any
number that goes into a plot. Units here are **solar masses** (`M_sun`)
rather than the raytracer projects' `M=1`, since a binary genuinely has
two different masses as free parameters -- "geometrized solar-mass
units", the standard convention in the PN/numerical-relativity
literature: 1 unit of length = `G*M_sun/c^2 = 1.4766 km`, 1 unit of time
= `G*M_sun/c^3 = 4.9255e-6 s`.

## The physics

**Newtonian + 2.5PN radiation reaction.** For relative separation
`r_vec = r1 - r2` (unit vector `n`, speed `v`, radial speed `rdot =
n.v`), total mass `M = m1+m2`, symmetric mass ratio `eta = m1*m2/M^2`:

```
a_Newtonian = -(M/r^2) n

a_2.5PN = (8/5) eta (M^2/r^3) [ (3v^2 + (17/3)(M/r)) rdot n - (v^2 + 3M/r) v ]
```

`a_2.5PN` is the standard leading-order (Burke-Thorne/Peters) radiation
-reaction acceleration for the relative two-body motion in harmonic
coordinates (quoted in this exact form in, e.g., Blanchet's Living
Reviews article on PN sources). It is **velocity-dependent and
non-conservative** -- unlike the raytracer projects' geodesic ODEs, this
is dissipative by design (it's the whole point: it drains orbital
energy). That rules out reusing `N_Body_Gravity`'s kick-drift-kick
leapfrog integrator as-is (leapfrog's symplectic accuracy guarantee
assumes velocity-*independent* forces; a velocity-dependent force
breaks the kick/drift split cleanly) -- an adaptive RK integrator
(`scipy.solve_ivp`) is used instead, which also matters practically
since the timescale itself shrinks by orders of magnitude between early
inspiral and merger (`da/dt` below scales as `1/a^3`), exactly the kind
of stiffening adaptive step control handles and fixed-step leapfrog
doesn't. Explicit 1PN conservative corrections (periastron advance) are
not included -- radiation reaction (the mechanism that makes this an
*inspiral* rather than a fixed ellipse, and the source of the waveform)
is the priority; noted under Known limitations.

**Validation target: the Peters (1964) formula.** Independent of the
instantaneous force law above -- derived instead by orbit-averaging the
quadrupole GW luminosity formula over one orbit -- the well-established
result for a binary's semi-major-axis decay rate is

```
da/dt = -(64/5) m1 m2 M / [a^3 (1-e^2)^(7/2)] * [1 + (73/24)e^2 + (37/96)e^4]   (G=c=1)
```

which for a circular orbit (`e=0`) integrates to a coalescence time
`T_c = (5/256) a0^4 / (m1 m2 M)`. If the direct-force-law simulation's
measured `da/dt` (and its total time to a near-merger cutoff) matches
this independently-derived formula, that's strong evidence the
2.5PN force's coefficient is right -- exactly the same
"cross-validate against something derived a completely different way"
discipline used throughout this whole `General_Relativity/` series.

**Waveform via the quadrupole formula.** For an observer along the
orbital-plane normal (face-on), the standard leading-order
(quadrupole-formula) strain is `h+ = (G/(c^4 D))(I_xx_ddot - I_yy_ddot)`,
`h_x = (2G/(c^4 D)) I_xy_ddot`, with the *reduced* mass quadrupole `I_ij
= mu(x_i x_j - delta_ij r^2/3)`. Differentiating twice analytically
(rather than by finite-differencing the trajectory, which would be
noisy) and noting the trace term (`propto delta_ij`) cancels
identically in `h+` and never appears in `h_x`:

```
I_xx_ddot = 2(vx^2 + x*ax)
I_yy_ddot = 2(vy^2 + y*ay)
I_xy_ddot = 2 vx vy + x*ay + y*ax
```

using the exact `(x,y,vx,vy,ax,ay)` the integrator already has at each
timestep (the same acceleration the 2.5PN force law computes) -- so the
waveform is built directly from the physical trajectory, not from a
separate closed-form chirp-frequency formula standing in for it.

## File layout

```
General_Relativity/GW_Inspiral/GW_Inspiral_Plan.md   # this design document
General_Relativity/GW_Inspiral/pn_dynamics.py         # relative EOM (Newtonian + 2.5PN RR), integration, orbital diagnostics
General_Relativity/GW_Inspiral/waveform.py            # quadrupole-formula h+/h_x from the trajectory, unit conversions
General_Relativity/GW_Inspiral/GW_Inspiral.ipynb       # Peters-formula validation + GW150914-parameter chirp waveform
General_Relativity/GW_Inspiral/media/
```

## Phases

- [x] **Phase 1 — Dynamics (`pn_dynamics.py`)** ✅ done. `relative_accel`
  (Newtonian + 2.5PN radiation reaction), `integrate_inspiral`
  (`solve_ivp`-based, with a terminal event at `r=6M`), orbital
  semi-major-axis/frequency diagnostics, plus the independent
  Peters-formula references (`peters_dadt`, `peters_coalescence_time`)
  used to validate it. **A real units bug found during scratch testing
  (not in the committed code)**: the first test used an initial
  separation smaller than the total mass (`a0=40` vs. `M=30`, i.e.
  *inside* `2M`) -- meaningless for a weak-field PN force law, and the
  integrator failed outright. Fixed by choosing `a0 >> M` (`a0=30M`),
  after which the physically sensible regime validated cleanly.
  **Validated**: at `m1=m2=15 M_sun`, `a0=30M`, measured early `da/dt`
  matches the Peters formula to `0.07%`, and total simulated time to
  the `r=6M` cutoff matches the Peters circular-orbit coalescence-time
  formula to `0.15%` -- strong evidence the 2.5PN force's coefficient
  is correct, since the two formulas were derived by entirely different
  methods (an instantaneous force law vs. orbit-averaged GW luminosity).
- [x] **Phase 2 — Waveform (`waveform.py`)** ✅ done. Quadrupole-formula
  `h+`/`h_x` from the trajectory's exact `(pos, vel, acc)` (the same
  acceleration the integrator computed, not a finite-difference
  estimate), plus geometrized-solar-mass <-> SI conversions. **Validated
  by hand** before implementation: substituting an exact circular-orbit
  trajectory (`x=r cos(wt)`, etc.) into the `I_ij_ddot` formulas reduces
  algebraically to the well-known textbook circular-binary strain
  amplitude `h0 = 4*mu*M/(D*r)`, confirming the derivative formulas.
  **Validated numerically**: for a GW150914-parameter run, peak `|h+|`
  came out to `~1.2e-21` at `410 Mpc` -- the right order of magnitude
  for a real detected signal -- and counting `h+`'s zero-crossings over
  a short segment gives a frequency matching the instantaneous
  `2x`-orbital-frequency formula to `0.2%`.
- [x] **Phase 3 — Demo notebook (`GW_Inspiral.ipynb`)** ✅ done. The
  Peters-formula validation as a first-class cell, a clean equal-mass
  (`15+15 M_sun`) circular-orbit chirp, and a GW150914-parameter
  (`36+29 M_sun`, `410 Mpc`) run. The GW150914 run's final (cutoff)
  frequency came out to `67.5 Hz`, visibly below the real event's
  measured peak (`~150-250 Hz`) -- expected and stated explicitly in
  the notebook: `r=6M` is a conservative "PN is no longer trustworthy"
  boundary chosen well short of the true strong-field merger, which
  needs higher PN orders or full numerical relativity to describe
  accurately. The chirp shape itself (slowly growing amplitude/frequency
  through most of the inspiral, then a sharp late upturn) closely
  matches the qualitative shape of LIGO's real published GW150914 plot,
  and the peak strain (`~1.2e-21`) lands at the right real-world order
  of magnitude.

## Known limitations (anticipated, to confirm/document once built)

- **No 1PN (or higher) conservative corrections** -- periastron advance
  is not captured; only the leading Newtonian + 2.5PN-radiation-reaction
  physics, sufficient for the chirp/inspiral phenomenon itself but not a
  precision waveform template.
- **Leading-order (restricted, quadrupole-only) waveform** -- no
  higher-order PN amplitude/phase corrections, no merger-ringdown (the
  simulation stops at a near-merger separation cutoff where PN itself
  becomes untrustworthy, well before the strong-field
  merger/ringdown regime that requires full numerical relativity).
- **Circular-orbit-focused** -- the code supports eccentric initial
  conditions (the force law makes no circularity assumption), but the
  primary validation and demo focus on circular/near-circular orbits.
- **Face-on-observer waveform only** -- no inclination-angle-dependent
  polarization mixing.

## Verification

1. Run `GW_Inspiral.ipynb`: confirm `da/dt` and total inspiral time
   match the Peters (1964) formula for a circular-orbit test binary.
2. Confirm the emitted GW frequency is exactly twice the orbital
   frequency (the standard quadrupole-formula relation) throughout the
   simulated inspiral.
3. Run the GW150914-parameter case and confirm the final GW frequency
   reached lands in the right ballpark of the real event's measured
   peak frequency (~150-250 Hz) before the PN-breakdown cutoff.
