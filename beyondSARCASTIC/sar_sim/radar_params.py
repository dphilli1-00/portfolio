"""
Ground the "500 pulses/collect" placeholder used throughout the compute-
scaling work in an actual system spec: 0.5 m resolution, X-band, spotlight.

Range resolution sets bandwidth directly:
    delta_r = c / (2B)  ->  B = c / (2 * delta_r)

Azimuth (cross-range) resolution in spotlight mode is set by the total
angle subtended by the synthetic aperture at the target, NOT by antenna
beamwidth (that's the whole point of spotlight -- steer the beam to buy a
bigger integration angle than a fixed beamwidth would give you for free):
    delta_az = lambda / (2 * d_theta)  ->  d_theta = lambda / (2*delta_az)

Minimum pulse count is set by Nyquist-sampling the synthetic aperture in
along-track distance (need at least 2 samples per azimuth resolution cell
to avoid azimuth ambiguities when focusing):
    PRF >= 2v / delta_az
    P = dwell_time * PRF = (R*d_theta/v) * (2v/delta_az)

Platform speed v cancels out completely -- P depends only on standoff
range, wavelength, and desired resolution:
    P = R * lambda / delta_az^2

ASSUMPTIONS (flagged, adjustable):
  delta_r = delta_az = 0.5 m         (isotropic resolution -- not stated,
                                       assumed for simplicity)
  fc = 10 GHz                        (X-band, matches the point-scatterer
                                       demo's fc=10e9 elsewhere in this work)
  R (standoff range) = 10 km          (typical airborne spotlight standoff --
                                       not given, flagged as adjustable)
"""

C = 299_792_458.0
delta_r = 0.5      # m
delta_az = 0.5     # m
fc = 10e9          # Hz, X-band
R = 10_000.0       # m, standoff range (assumption)

wavelength = C / fc
B = C / (2 * delta_r)
d_theta = wavelength / (2 * delta_az)          # rad
P = R * wavelength / (delta_az ** 2)           # pulses

print(f"Wavelength (X-band, fc={fc/1e9:.1f} GHz): {wavelength*100:.2f} cm")
print(f"Required bandwidth for {delta_r} m range res: {B/1e6:.1f} MHz")
print(f"Required integration angle for {delta_az} m az res: {d_theta:.4f} rad = {d_theta*180/3.14159265:.2f} deg")
print(f"Minimum Nyquist-sampled pulse count (R={R/1000:.0f} km standoff): {P:,.0f} pulses")

# sensitivity to standoff range assumption
print("\nSensitivity to standoff range:")
for R_test in [5000, 8000, 10000, 15000, 20000]:
    P_test = R_test * wavelength / (delta_az ** 2)
    print(f"  R={R_test/1000:5.0f} km -> P = {P_test:,.0f} pulses")

# --- fold into the break-even analysis: collects-to-break-even at real P ---
P_REAL = round(P)
B_WORST = 2.0
R_ANSYS = 75e-6
R_TIERED = 50e-6
R_ASC = 0.5e-6
N_CITY = 20000

P_star_worst_vs_ansys = B_WORST / (R_ANSYS - R_ASC)
P_star_worst_vs_tiered = B_WORST / (R_TIERED - R_ASC)
B_OPT = 0.5
P_star_opt_vs_tiered = B_OPT / (R_TIERED - R_ASC)

print(f"\nUsing a REAL spotlight collect (~{P_REAL:,} pulses instead of the round 500 placeholder):")
for label, P_star in [
    ("worst-case build (B=2s) vs Ansys dense rays", P_star_worst_vs_ansys),
    ("worst-case build (B=2s) vs my own tiered mode", P_star_worst_vs_tiered),
    ("optimistic build (B=0.5s) vs my own tiered mode", P_star_opt_vs_tiered),
]:
    collects_old = P_star / 500
    collects_real = P_star / P_REAL
    print(f"  {label}: break-even={P_star:,.0f} pulses -> "
          f"{collects_old:,.1f} collects @500ppc  |  {collects_real:,.1f} collects @{P_REAL:,}ppc (real spotlight)")

# ---------------------------------------------------------------------
# Satellite collection platform: standoff range jumps by ~50-80x vs
# airborne. LEO X-band SAR satellites (TerraSAR-X ~514km altitude,
# ICEYE/Capella ~500-575km altitude) sit at slant ranges roughly
# 500-1000 km depending on incidence angle. P = R*lambda/delta_az^2 still
# holds (platform speed cancelled out already) -- so pulse count scales
# directly with this much larger R.
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("SATELLITE PLATFORM (LEO X-band, slant range ~500-1000 km)")
print("=" * 70)

for R_sat in [500_000, 600_000, 700_000, 800_000, 1_000_000]:
    P_sat = R_sat * wavelength / (delta_az ** 2)
    print(f"  R={R_sat/1000:6.0f} km slant range -> P = {P_sat:,.0f} pulses")

R_SAT = 600_000.0  # representative LEO X-band standoff
P_SAT = round(R_SAT * wavelength / (delta_az ** 2))
print(f"\nUsing R={R_SAT/1000:.0f} km as representative: P_sat = {P_SAT:,} pulses/collect")
print(f"(vs P_air = {P_REAL:,} pulses/collect for the 10 km airborne case -- "
      f"{P_SAT/P_REAL:.1f}x more pulses per collect)")

print(f"\nBreak-even, re-expressed in COLLECTS at satellite pulse counts:")
for label, P_star in [
    ("worst-case build (B=2s) vs Ansys dense rays", P_star_worst_vs_ansys),
    ("worst-case build (B=2s) vs my own tiered mode", P_star_worst_vs_tiered),
    ("optimistic build (B=0.5s) vs my own tiered mode", P_star_opt_vs_tiered),
]:
    collects_sat = P_star / P_SAT
    pulses_short = "LESS THAN ONE COLLECT" if collects_sat < 1 else f"{collects_sat:,.2f} collects"
    print(f"  {label}: break-even={P_star:,.0f} pulses -> {pulses_short} "
          f"({collects_sat:.3f} collects @ {P_SAT:,} ppc)")

# ---------------------------------------------------------------------
# Reality check against a real system: TerraSAR-X Spotlight mode actually
# achieves ~2m resolution over a 10km x 10km scene -- NOT 0.5m. That's the
# real hardware tradeoff the earlier gotcha was pointing at: covering the
# full city footprint in one spotlight scene costs you resolution. Getting
# down to ~0.25m (TerraSAR-X "Staring Spotlight") means shrinking the scene
# to ~4km x 3.7km instead.
#
# Redo the pulse-count and break-even numbers using the achievable,
# TerraSAR-X-consistent combo: delta=2m over the full 10km x 10km / N=20,000
# structure city, rather than the unrealistic 0.5m-over-10km combo used above.
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("REALITY CHECK: TerraSAR-X Spotlight -- 10km x 10km scene @ ~2m res")
print("=" * 70)

delta_real = 2.0  # m, matches real TerraSAR-X spotlight over the full scene
B_real = C / (2 * delta_real)
d_theta_real = wavelength / (2 * delta_real)
P_city_real = R_SAT * wavelength / (delta_real ** 2)

print(f"Bandwidth for {delta_real} m res: {B_real/1e6:.1f} MHz (vs {B/1e6:.1f} MHz for 0.5m -- "
      f"much easier waveform)")
print(f"Pulse count @ R={R_SAT/1000:.0f} km, full 10km x 10km city: {P_city_real:,.0f} pulses "
      f"(vs {P_SAT:,.0f} for the unrealistic 0.5m-over-10km combo -- {P_SAT/P_city_real:.0f}x fewer)")

print(f"\nBreak-even collects using this REAL, achievable combo (2m res, full city, N=20,000):")
for label, P_star in [
    ("worst-case build (B=2s) vs Ansys dense rays", P_star_worst_vs_ansys),
    ("worst-case build (B=2s) vs my own tiered mode", P_star_worst_vs_tiered),
    ("optimistic build (B=0.5s) vs my own tiered mode", P_star_opt_vs_tiered),
]:
    collects = P_star / P_city_real
    print(f"  {label}: {collects:,.2f} collects")

print(f"\nAlternative: keep 0.5m res, shrink to what it actually buys (TerraSAR-X Staring")
print(f"Spotlight-class scene, ~4km x 3.7km ~= 15 km^2, vs 100 km^2 for the full city).")
print(f"Structure count scales with area: N_small ~= 20,000 * (15/100) = {20000*15/100:,.0f}")
N_SMALL = round(20000 * 15 / 100)
P_small = R_SAT * wavelength / (delta_az ** 2)  # same P as before -- pulse count is res-driven, not area-driven
B_WORST_small = N_SMALL * B_WORST
print(f"P is unchanged ({P_small:,.0f} pulses -- driven by resolution+range, not scene area),")
print(f"but N drops to {N_SMALL:,}, so build cost drops to {B_WORST_small:,.0f}s "
      f"(vs {N_CITY*B_WORST:,.0f}s at full city N)")
for label, r_opp in [("vs Ansys dense rays", R_ANSYS), ("vs my own tiered mode", R_TIERED)]:
    P_star_small = B_WORST / (r_opp - R_ASC)  # per-structure break-even is N-independent, unchanged
    collects_small = P_star_small / P_small
    print(f"  worst-case build {label}: {collects_small:,.3f} collects (unchanged -- P* is N-independent)")
