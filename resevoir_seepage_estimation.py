"""
Step 2: Build the empirical A-E-V (Area-Elevation-Volume) curve for GERD.

DAHITI gives us water level (H) and surface area (A) but never volume (V)
directly. This script:

  1. Uses the prismatoid formula on consecutive DAHITI H/A observations to
     estimate month-to-month volume CHANGE (relative, unknown baseline).
  2. Anchors that relative trajectory to absolute volume using 3 known
     GERD benchmarks (bed level, dead storage level, full supply level).
  3. Fits a 2D polynomial regression V = f(H, A) on the now-calibrated
     absolute volume series, which is what feeds the mass balance filter.

CAVEATS (worth flagging in your writeup):
  - Benchmark elevations are in m.a.s.l.; DAHITI altimetry is referenced
    to a geoid model (e.g. EGM2008) -- usually close, but a small
    systematic vertical offset is possible.
  - The benchmarks are design/theoretical capacities from engineering
    specs, not a direct bathymetric survey -- treat this as a
    research-grade calibration, not survey-grade ground truth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. Load the merged DAHITI monthly H/A dataset from the previous step
# =============================================================================
df = pd.read_csv("dahiti_27208_merged_monthly.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

H = df["water_level_m"].values
A = df["surface_area_km2"].values

# =============================================================================
# 2. Prismatoid formula -> relative volume change between consecutive months
# =============================================================================
# Standard prismatoid/conic frustum approximation for volume between two
# height/area cross-sections. Units: A in km^2, H in m -> V in km^2*m
# (= 0.001 km^3 = 1e6 m^3). We convert to BCM (billion cubic meters,
# 1 BCM = 1 km^3) at the end.

delta_H = np.diff(H)
A1 = A[:-1]
A2 = A[1:]

delta_V_km2m = delta_H * (A1 + A2 + np.sqrt(np.clip(A1 * A2, 0, None))) / 3
delta_V_bcm = delta_V_km2m * 0.001  # km^2*m -> km^3 (BCM)

# Cumulative relative volume trajectory (arbitrary baseline -- first point = 0)
relative_V = np.concatenate([[0.0], np.cumsum(delta_V_bcm)])
df["relative_V_bcm"] = relative_V

# =============================================================================
# 3. Anchor to absolute volume using known GERD benchmarks
# =============================================================================
# (elevation_m, volume_BCM) pairs from engineering design specs
BENCHMARKS = [
    (500.0, 0.0),    # bed level -> ~zero storage
    (590.0, 14.79),  # dead storage level
    (640.0, 74.0),   # full supply level
]

bench_H = np.array([b[0] for b in BENCHMARKS])
bench_V = np.array([b[1] for b in BENCHMARKS])

# Fit a quadratic V_benchmark = f(H) through the 3 known points -- this is
# the "official" elevation-volume curve implied by the design specs alone.
benchmark_coefs = np.polyfit(bench_H, bench_V, 2)
print("[INFO] Benchmark elevation-volume quadratic (V = aH^2 + bH + c):")
print(f"       {benchmark_coefs}")

# For each DAHITI month, compute what the benchmark curve says volume
# SHOULD be at that observed water level.
benchmark_V_at_H = np.polyval(benchmark_coefs, H)

# The offset between our relative (DAHITI-derived) trajectory and the
# benchmark-implied absolute volume. IMPORTANT: a single additive offset
# only corrects a vertical shift -- it can't fix a SCALE/curvature
# mismatch. Diagnostic plots showed exactly that (calibrated points
# diverge from the benchmark curve in the middle of the range, converging
# only near the endpoints), likely because the benchmark quadratic is
# extrapolated between 500m and 590m without any real anchor point in
# between, while most of our real DAHITI observations sit right in that
# unverified stretch. Fit BOTH a scale (a) and offset (b) via least
# squares: benchmark_V_at_H ~= a * relative_V + b
A_calib = np.column_stack([relative_V, np.ones_like(relative_V)])
(scale, offset), _, _, _ = np.linalg.lstsq(A_calib, benchmark_V_at_H, rcond=None)
df["absolute_V_bcm"] = scale * relative_V + offset

print(f"\n[INFO] Calibration fit: scale={scale:.4f}, offset={offset:+.3f} BCM")
print(f"[INFO] Calibrated volume range: "
      f"{df['absolute_V_bcm'].min():.2f} to {df['absolute_V_bcm'].max():.2f} BCM")
print(f"[INFO] Water level range: {H.min():.1f} to {H.max():.1f} m")

n_negative = (df["absolute_V_bcm"] < 0).sum()
if n_negative > 0:
    print(f"\n[WARNING] {n_negative} months have NEGATIVE calibrated volume -- "
          f"physically impossible. This means the benchmark-based calibration "
          f"still doesn't fully match the shape of the DAHITI-derived trajectory. "
          f"Consider: (a) fitting a higher-degree benchmark curve if more real "
          f"elevation-volume points become available, or (b) treating early "
          f"low-water-level months as unreliable given their large area/level "
          f"error bars, or (c) getting an actual bathymetric survey table "
          f"instead of the 3-point design-spec approximation.")

# Small negative dips in the pre-fill era (2019-2021, before GERD began
# retaining significant water around 2020) plausibly reflect calibration
# noise near true-zero storage rather than a real physical quantity.
# Clipping to zero is a defensible, common practice -- but it must be
# DOCUMENTED, not silently applied, since it changes the data.
CLIP_NEGATIVE_TO_ZERO = True  # <-- set False to keep raw (possibly negative) values
if CLIP_NEGATIVE_TO_ZERO and n_negative > 0:
    df["absolute_V_bcm_raw"] = df["absolute_V_bcm"]  # keep the unclipped version too
    df["absolute_V_bcm"] = df["absolute_V_bcm"].clip(lower=0)
    print(f"[INFO] Clipped {n_negative} negative-volume months to 0 BCM "
          f"(raw values preserved in 'absolute_V_bcm_raw' column). "
          f"This is documented here as a deliberate, visible choice -- "
          f"mention it in your methods writeup.")

# Sanity check: does the calibrated trajectory land near the benchmarks
# when H crosses them?
print("\n[INFO] Sanity check against benchmarks:")
for bh, bv in BENCHMARKS:
    nearby = df.iloc[(df["water_level_m"] - bh).abs().argsort()[:1]]
    if len(nearby):
        print(f"       Benchmark H={bh}m (V={bv} BCM) -> nearest DAHITI month "
              f"H={nearby['water_level_m'].values[0]:.1f}m, "
              f"calibrated V={nearby['absolute_V_bcm'].values[0]:.2f} BCM")

# =============================================================================
# 4. Fit the empirical polynomial A-E-V curve: V = f(H, A)
# =============================================================================
# 2nd-degree polynomial surface: V = c0 + c1*H + c2*A + c3*H^2 + c4*A^2 + c5*H*A
from numpy.polynomial import polynomial as P

def build_design_matrix(H, A, degree=2):
    """Build polynomial feature columns for a 2-variable regression."""
    terms = [np.ones_like(H)]
    if degree >= 1:
        terms += [H, A]
    if degree >= 2:
        terms += [H**2, A**2, H * A]
    if degree >= 3:
        terms += [H**3, A**3, H**2 * A, H * A**2]
    return np.column_stack(terms)

DEGREE = 2
X_design = build_design_matrix(H, A, degree=DEGREE)
y_target = df["absolute_V_bcm"].values

coefs, residuals, rank, sv = np.linalg.lstsq(X_design, y_target, rcond=None)

def empirical_aev_curve(H_input, A_input, coefs=coefs, degree=DEGREE):
    """The A-E-V function referenced in your mentor's Step 2 pseudocode --
    predicts absolute volume (BCM) from water level (m) and surface area (km2)."""
    H_input = np.asarray(H_input, dtype=float)
    A_input = np.asarray(A_input, dtype=float)
    X = build_design_matrix(H_input, A_input, degree=degree)
    return X @ coefs

df["V_fitted_bcm"] = empirical_aev_curve(H, A)

fit_residuals = df["absolute_V_bcm"] - df["V_fitted_bcm"]
r2 = 1 - np.sum(fit_residuals**2) / np.sum((y_target - y_target.mean())**2)
print(f"\n[INFO] Polynomial A-E-V fit (degree {DEGREE}): R^2 = {r2:.4f}, "
      f"RMSE = {np.sqrt(np.mean(fit_residuals**2)):.3f} BCM")

# =============================================================================
# 5. Diagnostic plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(df["date"], df["absolute_V_bcm"], color="black", linewidth=1.5,
          label="Calibrated volume (prismatoid + benchmark anchor)")
ax1.plot(df["date"], df["V_fitted_bcm"], color="steelblue", linestyle="--",
          label=f"Polynomial fit (degree {DEGREE})")
for bh, bv in BENCHMARKS:
    ax1.axhline(bv, color="gray", linewidth=0.7, linestyle=":")
ax1.set_title("GERD reservoir volume over time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Volume (BCM)")
ax1.legend(fontsize=8)

sort_idx = np.argsort(H)
ax2.scatter(H, df["absolute_V_bcm"], s=12, color="black", label="Calibrated (DAHITI-derived)")
ax2.scatter(bench_H, bench_V, s=60, color="darkorange", zorder=5, label="Design benchmarks (raw points)")
# Plot the ACTUAL fitted quadratic used for calibration (dense sweep), not
# a misleading straight-line connection between only 3 points -- those are
# visually different shapes and comparing against straight lines
# overstates/understates the real calibration error.
H_dense = np.linspace(H.min(), H.max(), 200)
ax2.plot(H_dense, np.polyval(benchmark_coefs, H_dense), color="darkorange",
         linestyle="-", alpha=0.8, label="Fitted benchmark quadratic (actual calibration target)")
ax2.set_title("Elevation-Volume relationship")
ax2.set_xlabel("Water level (m)")
ax2.set_ylabel("Volume (BCM)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("gerd_aev_curve_diagnostic.png", dpi=150)
plt.show()
print("[INFO] Diagnostic plot saved to gerd_aev_curve_diagnostic.png")

# =============================================================================
# 6. Save outputs for Step 3/4 (mass balance filter)
# =============================================================================
df["delta_V_surface"] = df["V_fitted_bcm"].diff()
df.to_csv("dahiti_27208_with_volume.csv", index=False)
print("[INFO] Saved dahiti_27208_with_volume.csv with V_fitted_bcm and delta_V_surface")
print(df[["date", "water_level_m", "surface_area_km2", "absolute_V_bcm",
          "V_fitted_bcm", "delta_V_surface"]].head())