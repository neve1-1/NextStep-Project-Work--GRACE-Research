"""
Reservoir Seepage Estimation Pipeline (Steps 2-4)
===================================================

Step 2: Build the empirical A-E-V (Area-Elevation-Volume) curve for the reservoir.
Step 3: Load the ensemble-averaged, gap-filled GRACE/GRACE-FO TWSA time series
        (CSR+JPL+GSFC ensemble, produced upstream by GERD_gapfill_model.py).
Step 4: Apply the mass balance filter, per mentor's formula:
            delta_TWSA_subsurface = csr_twsa_vol - delta_V_surface
            deep_seepage_loss     = delta_TWSA_subsurface - delta_V_surface
        then annualize.

============================ ASSUMPTIONS TO VERIFY ============================
  1. Column names in dahiti_df are GENERIC placeholders -- rename to match
     your actual altimetry file, or rename your file's columns to match these.
  2. csr_twsa_vol is used exactly as mentor specified -- as the raw monthly
     level/anomaly, NOT diffed into a monthly change, before the surface
     water correction below.
  3. Units are BCM (billion cubic meters, i.e. km^3) for all volume/storage
     columns. TWSA input arrives in cm (GERD_gapfill_model.py converts its
     native inches to cm before writing its output file) and is converted
     to BCM here via BASIN_AREA_KM2.
  4. TWSA source: run GERD_gapfill_model.py FIRST. It builds the CSR+JPL+GSFC
     ensemble mean over the GERD basin, DNN-gap-fills the pre-2020-07-01
     window (where real GRACE/GRACE-FO gaps exist), passes 2020-07-01+
     through raw (confirmed gap-free), stitches both into one continuous
     series, and writes gerd_twsa_gapfilled_dnn_full.csv (column:
     lwe_thickness_cm). This script reads that file directly -- it no
     longer reads basin_csr_mascons.csv or does its own gap-fill hook,
     since GERD_gapfill_model.py already produces a complete series.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 2: Build the empirical A-E-V curve
# =============================================================================

def build_design_matrix(H, A, degree=2):
    """Polynomial feature columns for the 2-variable (H, A) -> V regression."""
    terms = [np.ones_like(H)]
    if degree >= 1:
        terms += [H, A]
    if degree >= 2:
        terms += [H**2, A**2, H * A]
    if degree >= 3:
        terms += [H**3, A**3, H**2 * A, H * A**2]
    return np.column_stack(terms)


def fit_aev_curve(H, A, V_target, degree=2):
    """
    Fit V = f(H, A) via least-squares polynomial regression.
    Returns the fitted coefficients and a callable curve function.
    """
    X_design = build_design_matrix(H, A, degree=degree)
    coefs, residuals, rank, sv = np.linalg.lstsq(X_design, V_target, rcond=None)

    def curve_fn(H_input, A_input):
        H_input = np.asarray(H_input, dtype=float)
        A_input = np.asarray(A_input, dtype=float)
        X = build_design_matrix(H_input, A_input, degree=degree)
        return X @ coefs

    # Fit diagnostics
    V_fitted = curve_fn(H, A)
    resid = V_target - V_fitted
    r2 = 1 - np.sum(resid**2) / np.sum((V_target - V_target.mean())**2)
    rmse = np.sqrt(np.mean(resid**2))
    print(f"[INFO] A-E-V fit (degree {degree}): R^2 = {r2:.4f}, RMSE = {rmse:.3f} BCM")

    return coefs, curve_fn


def prismatoid_relative_volume(H, A):
    """
    Convert consecutive H/A observations into a relative volume trajectory
    (arbitrary baseline, first point = 0) using the prismatoid formula.
    Use this if you need to calibrate against known benchmarks before
    fitting the final A-E-V curve (see previous script for the benchmark
    anchoring approach) -- included here for completeness.
    """
    delta_H = np.diff(H)
    A1, A2 = A[:-1], A[1:]
    delta_V_km2m = delta_H * (A1 + A2 + np.sqrt(np.clip(A1 * A2, 0, None))) / 3
    delta_V_bcm = delta_V_km2m * 0.001  # km^2*m -> km^3 (BCM)
    return np.concatenate([[0.0], np.cumsum(delta_V_bcm)])


# --- Load your synchronized monthly altimetry dataset -----------------------

dahiti_df = pd.read_csv("dahiti_27208_merged_monthly.csv", parse_dates=["date"])
dahiti_df = dahiti_df.sort_values("date").reset_index(drop=True)

# --- Check for missing altimetry data before fitting -------------------------
n_nan_H = dahiti_df["water_level_m"].isna().sum()
n_nan_A = dahiti_df["surface_area_km2"].isna().sum()
if n_nan_H > 0 or n_nan_A > 0:
    print(f"[WARNING] {n_nan_H} months missing water_level_m, {n_nan_A} months "
          f"missing surface_area_km2 -- dropping these rows before curve fitting "
          f"(lstsq cannot handle NaN/Inf and will fail with a cryptic LAPACK error).")
    dahiti_df = dahiti_df.dropna(subset=["water_level_m", "surface_area_km2"]).reset_index(drop=True)

H = dahiti_df["water_level_m"].values
A = dahiti_df["surface_area_km2"].values

# Belt-and-suspenders: also confirm nothing non-finite slipped through
if not (np.all(np.isfinite(H)) and np.all(np.isfinite(A))):
    raise ValueError("H or A still contains NaN/Inf after dropna -- check the CSV directly.")

H = dahiti_df["water_level_m"].values
A = dahiti_df["surface_area_km2"].values

# If you already have a calibrated absolute volume column (e.g. from
# benchmark-anchoring, as in the earlier script), use it as V_target here.
# ADJUST ME: replace with your actual calibrated volume column if you have one
if "absolute_V_bcm" in dahiti_df.columns:
    V_target = dahiti_df["absolute_V_bcm"].values
else:
    print("[WARNING] No calibrated 'absolute_V_bcm' column found -- falling back "
          "to uncalibrated relative volume from the prismatoid formula. This "
          "trajectory has an ARBITRARY baseline (first month = 0 BCM), not "
          "true absolute volume. Anchor it to known reservoir benchmarks "
          "(bed level, dead storage, full supply level) before using it in "
          "the mass balance step below, or your seepage estimate will be "
          "biased by whatever the arbitrary baseline offset happens to be.")
    V_target = prismatoid_relative_volume(H, A)

aev_coefs, empirical_aev_curve = fit_aev_curve(H, A, V_target, degree=2)

dahiti_df["V_surface"] = empirical_aev_curve(H, A)
n_negative = (dahiti_df["V_surface"] < 0).sum()
if n_negative > 0:
    print(f"[WARNING] {n_negative} months have negative V_surface (physically "
          f"impossible) -- likely polynomial extrapolation in sparse regions "
          f"of (H, A) space. Clipping to zero; raw values preserved.")
    dahiti_df["V_surface_raw"] = dahiti_df["V_surface"]
    dahiti_df["V_surface"] = dahiti_df["V_surface"].clip(lower=0)

dahiti_df["delta_V_surface"] = dahiti_df["V_surface"].diff()


# =============================================================================
# STEP 3: Load the ensemble-averaged, gap-filled TWSA (CSR+JPL+GSFC)
# =============================================================================
# Source: GERD_gapfill_model.py -- RUN THAT SCRIPT FIRST. This step no longer
# reads basin_csr_mascons.csv or CSR alone; the ensemble file below already
# contains a complete, gap-free monthly series, so there's no separate
# gap-fill hook here anymore.
csr_df = pd.read_csv("gerd_twsa_gapfilled_dnn_full.csv", parse_dates=["date"])
csr_df = csr_df.rename(columns={csr_df.columns[0]: "date"})  # in case the index column name differs

if "lwe_thickness_cm" not in csr_df.columns:
    raise KeyError(
        f"Expected column 'lwe_thickness_cm' not found in "
        f"gerd_twsa_gapfilled_dnn_full.csv. Found columns: {list(csr_df.columns)} "
        f"-- check GERD_gapfill_model.py's output column naming (the cm-conversion "
        f"snippet should be renaming it to 'lwe_thickness_cm')."
    )

# --- Convert LWE thickness (cm) to volumetric anomaly (BCM) -----------------
BASIN_AREA_KM2 = 172250  # GERD tributary catchment (Blue Nile subbasin)

# cm thickness * km^2 area -> km^3 (BCM): (cm / 100000) * area_km2
csr_df["csr_twsa_vol"] = (csr_df["lwe_thickness_cm"] / 100000) * BASIN_AREA_KM2

n_nan_twsa = csr_df["csr_twsa_vol"].isna().sum()
if n_nan_twsa > 0:
    print(f"[WARNING] {n_nan_twsa} months have NaN csr_twsa_vol after loading "
          f"the ensemble file -- these should only occur if GERD_gapfill_model.py's "
          f"post-2020-07-01 pass-through assumption ('no gaps after this date') "
          f"turned out to be wrong. Check that script's own NaN warning before "
          f"trusting downstream results, since NaNs here will propagate into "
          f"the mass balance below.")

n_missing_months = pd.date_range(
    csr_df["date"].min(), csr_df["date"].max(), freq="MS"
).difference(csr_df["date"])
if len(n_missing_months) > 0:
    print(f"[WARNING] {len(n_missing_months)} calendar months are missing "
          f"entirely from the ensemble file's date range (not just NaN, but "
          f"absent rows) -- unexpected, since GERD_gapfill_model.py should "
          f"produce one continuous monthly series. Investigate before trusting "
          f"downstream results.")


# =============================================================================
# STEP 4: Apply the mass balance filter
# =============================================================================
# Mentor's exact formula:
#   delta_TWSA_subsurface = csr_twsa_vol - delta_V_surface
#   deep_seepage_loss     = delta_TWSA_subsurface - delta_V_surface
# csr_twsa_vol is used as-is (a level/anomaly, not diffed into a monthly
# change) per mentor's instructions.

# --- Merge altimetry-derived surface storage with GRACE TWSA ---------------
df = pd.merge(dahiti_df, csr_df, on="date", how="inner")

# --- Surface water correction (mentor's formula) ----------------------------
df["delta_TWSA_subsurface"] = df["csr_twsa_vol"] - df["delta_V_surface"]

# ---------------------------------------------------------------------------
# TOGGLE: flip this to True if your mentor says inflow/evaporation (or other
# subsurface compartments like soil moisture/groundwater) need to be
# accounted for separately, rather than the mass-balance filter below.
INCLUDE_INFLOW_EVAP = False  # ADJUST ME: set True once confirmed with mentor
# ---------------------------------------------------------------------------

if INCLUDE_INFLOW_EVAP:
    # ADJUST ME: replace file paths and column names with your actual files.
    inflow_df = pd.read_csv("inflow_data.csv", parse_dates=["date"])
    evap_df = pd.read_csv("evaporation_data.csv", parse_dates=["date"])

    # ADJUST ME: rename these to your actual column names, and confirm units
    # are BCM/month (or convert here, e.g. evap_mm * reservoir_area_km2 / 1e6).
    INFLOW_COL = "inflow_bcm"       # ADJUST ME
    EVAP_COL = "evaporation_bcm"    # ADJUST ME

    df = pd.merge(df, inflow_df[["date", INFLOW_COL]], on="date", how="left")
    df = pd.merge(df, evap_df[["date", EVAP_COL]], on="date", how="left")

    missing_inflow = df[INFLOW_COL].isna().sum()
    missing_evap = df[EVAP_COL].isna().sum()
    if missing_inflow or missing_evap:
        print(f"[WARNING] {missing_inflow} months missing inflow data, "
              f"{missing_evap} months missing evaporation data after merge -- "
              f"these months will produce NaN seepage estimates unless filled.")

    # Inflow/evaporation terms plus mentor's surface water correction.
    df["deep_seepage_loss"] = (
        df[INFLOW_COL]
        - df[EVAP_COL]
        - df["delta_TWSA_subsurface"]
        - df["delta_V_surface"]
    )
    print("[INFO] Mass balance computed WITH inflow/evaporation terms.")

else:
    # "Mass Balance Filter" here means smoothing/cleaning the seepage signal
    # (e.g. a rolling mean to remove monthly noise) -- NOT adding new terms.
    # Set FILTER_WINDOW to None/1 to skip smoothing entirely.
    FILTER_WINDOW = None  # ADJUST ME: e.g. 3 for a 3-month rolling mean, or None for no smoothing

    seepage_raw = df["delta_TWSA_subsurface"] - df["delta_V_surface"]

    if FILTER_WINDOW and FILTER_WINDOW > 1:
        df["deep_seepage_loss"] = seepage_raw.rolling(FILTER_WINDOW, center=True).mean()
        print(f"[INFO] Applied {FILTER_WINDOW}-month rolling mean as the mass balance filter.")
    else:
        df["deep_seepage_loss"] = seepage_raw
        print("[INFO] No smoothing filter applied -- deep_seepage_loss = "
              "delta_TWSA_subsurface - delta_V_surface directly (mentor's formula).")
    print("[INFO] Mass balance computed WITHOUT inflow/evaporation terms "
          "(INCLUDE_INFLOW_EVAP = False).")

# =============================================================================
# STEP 5: Annualize and summarize
# =============================================================================
df["year"] = df["date"].dt.year
annual_seepage = df.groupby("year")["deep_seepage_loss"].sum().rename("seepage_bcm_per_year")

print("\n[INFO] Monthly seepage estimate summary:")
print(df[["date", "deep_seepage_loss"]].describe())

print("\n[INFO] Annualized seepage estimate (BCM/year):")
print(annual_seepage)

mean_annual_seepage = annual_seepage.mean()
print(f"\n[RESULT] Mean estimated seepage: {mean_annual_seepage:.3f} BCM/year")

# =============================================================================
# Diagnostic plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(df["date"], df["deep_seepage_loss"], color="firebrick", linewidth=1.2,
          label="Deep seepage loss (BCM/month)")
ax1.plot(df["date"], df["csr_twsa_vol"], color="steelblue", linewidth=1.2,
          linestyle="--", label="GRACE TWSA volume (BCM)")
ax1.axhline(0, color="gray", linewidth=0.7, linestyle=":")
ax1.set_title("Monthly seepage estimate vs. TWSA volume")
ax1.set_xlabel("Date")
ax1.set_ylabel("BCM")
ax1.legend(loc="upper left", fontsize=8)

annual_seepage.plot(kind="bar", ax=ax2, color="steelblue")
ax2.set_title("Annualized seepage estimate")
ax2.set_xlabel("Year")
ax2.set_ylabel("Seepage (BCM/year)")

plt.tight_layout()
plt.savefig("seepage_estimate_diagnostic.png", dpi=150)
plt.show()
print("[INFO] Diagnostic plot saved to seepage_estimate_diagnostic.png")
n_below = (df["csr_twsa_vol"] < df["deep_seepage_loss"]).sum()
n_total = df["csr_twsa_vol"].notna().sum()


