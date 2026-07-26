import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Load Dataset & Compute Core Storage Variables
# =============================================================================
csv_filename = "dahiti_27208_with_volume.Check.forGemini.csv"
df = pd.read_csv(csv_filename)

# Parse Excel serial dates (Origin: 1899-12-30)
df["date_dt"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")
df = df.sort_values("date_dt").reset_index(drop=True)
df = df.dropna(subset=["water_level_m", "surface_area_km2", "TWSA_cm"]).reset_index(drop=True)

H = df["water_level_m"].values
A = df["surface_area_km2"].values

def prismatoid_relative_volume(H, A):
    """Compute surface storage target via prismatoid integration."""
    delta_H = np.diff(H)
    A1, A2 = A[:-1], A[1:]
    delta_V_bcm = delta_H * (A1 + A2 + np.sqrt(np.clip(A1 * A2, 0, None))) / 3 * 0.001
    return np.concatenate([[0.0], np.cumsum(delta_V_bcm)])

def build_design_matrix(H, A, degree=2):
    """Degree-2 polynomial design matrix."""
    return np.column_stack([np.ones_like(H), H, A, H**2, A**2, H * A])

# Solve A-E-V Polynomial Model
V_target = prismatoid_relative_volume(H, A)
X_design = build_design_matrix(H, A, degree=2)
coefs, _, _, _ = np.linalg.lstsq(X_design, V_target, rcond=None)

# Evaluated Cumulative Storage (BCM) and Monthly Flux delV (BCM/month)
df["V_surface_bcm"] = np.maximum(X_design @ coefs, 0)
df["delV_surface_bcm"] = df["V_surface_bcm"].diff()
df["delTWSA_cm"] = df["TWSA_cm"].diff()
df["year"] = df["date_dt"].dt.year
df["month"] = df["date_dt"].dt.month

valid = df.dropna(subset=["delV_surface_bcm", "delTWSA_cm", "TWSA_cm", "V_surface_bcm"]).copy()

# Pearson Correlation Coefficients
r_delV_delTWSA = np.corrcoef(valid["delV_surface_bcm"], valid["delTWSA_cm"])[0, 1]
r_V_TWSA = np.corrcoef(valid["V_surface_bcm"], valid["TWSA_cm"])[0, 1]


# =============================================================================
# FIGURE 1: Primary 3-Panel Reservoir Storage Dynamics
# =============================================================================
fig1, axes1 = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

# Panel 1.1: Monthly delV Bars vs Cumulative Volume
ax1_twin = axes1[0].twinx()
axes1[0].bar(valid["date_dt"], valid["delV_surface_bcm"], width=20, color="navy", alpha=0.55,
             label="Monthly Change ($\\Delta V_{\\mathrm{surface}}$)")
ax1_twin.plot(valid["date_dt"], valid["V_surface_bcm"], color="darkblue", lw=2,
              label="Cumulative Storage ($V_{\\mathrm{surface}}$)")
axes1[0].set_title("Figure 1.1: Monthly Surface Storage Change ($\\Delta V_{\\mathrm{surface}}$) & Cumulative Volume",
                   fontsize=11, fontweight="bold")
axes1[0].set_ylabel("$\\Delta V_{\\mathrm{surface}}$ (BCM/month)", fontsize=10)
ax1_twin.set_ylabel("Total Storage (BCM)", fontsize=10)
axes1[0].axhline(0, color="gray", linestyle=":", lw=1)

lines1, labels1 = axes1[0].get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
axes1[0].legend(lines1 + lines2, labels1 + labels2, loc="upper left")
axes1[0].grid(True, alpha=0.3)

# Panel 1.2: Dual-Axis Time Series delV vs ΔTWSA
ax2_twin = axes1[1].twinx()
axes1[1].plot(valid["date_dt"], valid["delV_surface_bcm"], color="navy", lw=2,
              label="Surface Change ($\\Delta V_{\\mathrm{surface}}$)")
ax2_twin.plot(valid["date_dt"], valid["delTWSA_cm"], color="teal", lw=1.5, linestyle="--",
              label="Monthly $\\Delta\\mathrm{TWSA}$")
axes1[1].axhline(0, color="gray", linestyle=":", lw=1)
axes1[1].set_title("Figure 1.2: Time-Series Comparison: $\\Delta V_{\\mathrm{surface}}$ vs. Monthly $\\Delta\\mathrm{TWSA}$",
                   fontsize=11, fontweight="bold")
axes1[1].set_ylabel("$\\Delta V_{\\mathrm{surface}}$ (BCM/month)", fontsize=10)
ax2_twin.set_ylabel("$\\Delta\\mathrm{TWSA}$ (cm/month)", fontsize=10)

lines3, labels3 = axes1[1].get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
axes1[1].legend(lines3 + lines4, labels3 + labels4, loc="upper left")
axes1[1].grid(True, alpha=0.3)

# Panel 1.3: Filling vs. Drawdown Categorization
fill_mask = valid["delV_surface_bcm"] >= 0
draw_mask = valid["delV_surface_bcm"] < 0
axes1[2].bar(valid.loc[fill_mask, "date_dt"], valid.loc[fill_mask, "delV_surface_bcm"],
             width=20, color="forestgreen", alpha=0.7, label="Impoundment / Filling ($\\Delta V > 0$)")
axes1[2].bar(valid.loc[draw_mask, "date_dt"], valid.loc[draw_mask, "delV_surface_bcm"],
             width=20, color="crimson", alpha=0.7, label="Release / Drawdown ($\\Delta V < 0$)")
axes1[2].axhline(0, color="black", linestyle="-", lw=0.8)
axes1[2].set_title("Figure 1.3: Operational Classification: Impoundment Pulses vs. Release Cycles",
                   fontsize=11, fontweight="bold")
axes1[2].set_xlabel("Date", fontsize=10)
axes1[2].set_ylabel("$\\Delta V_{\\mathrm{surface}}$ (BCM/month)", fontsize=10)
axes1[2].legend(loc="upper left")
axes1[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Fig1_surface_storage_dynamics.png", dpi=150)


# =============================================================================
# FIGURE 2: Correlation & Scatter Analysis
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Scatter 2.1: Monthly delV vs ΔTWSA
axes2[0].scatter(valid["delTWSA_cm"], valid["delV_surface_bcm"], color="darkcyan", alpha=0.7, edgecolors="k")
m1, b1 = np.polyfit(valid["delTWSA_cm"], valid["delV_surface_bcm"], 1)
x_vals1 = np.linspace(valid["delTWSA_cm"].min(), valid["delTWSA_cm"].max(), 100)
axes2[0].plot(x_vals1, m1 * x_vals1 + b1, color="firebrick", lw=2,
              label=f"Fit: y = {m1:.2f}x + {b1:.2f}\nr = {r_delV_delTWSA:.4f}")
axes2[0].axhline(0, color="gray", linestyle=":", lw=1)
axes2[0].axvline(0, color="gray", linestyle=":", lw=1)
axes2[0].set_title("Figure 2.1: Monthly Flux Correlation: $\\Delta V_{\\mathrm{surface}}$ vs. $\\Delta\\mathrm{TWSA}$",
                   fontsize=11, fontweight="bold")
axes2[0].set_xlabel("Monthly $\\Delta\\mathrm{TWSA}$ (cm/month)", fontsize=10)
axes2[0].set_ylabel("$\\Delta V_{\\mathrm{surface}}$ (BCM/month)", fontsize=10)
axes2[0].legend(loc="upper left")
axes2[0].grid(True, alpha=0.3)

# Scatter 2.2: Cumulative V_surface vs TWSA Anomaly
axes2[1].scatter(valid["TWSA_cm"], valid["V_surface_bcm"], color="navy", alpha=0.7, edgecolors="k")
m2, b2 = np.polyfit(valid["TWSA_cm"], valid["V_surface_bcm"], 1)
x_vals2 = np.linspace(valid["TWSA_cm"].min(), valid["TWSA_cm"].max(), 100)
axes2[1].plot(x_vals2, m2 * x_vals2 + b2, color="firebrick", lw=2,
              label=f"Fit: y = {m2:.2f}x + {b2:.2f}\nr = {r_V_TWSA:.4f}")
axes2[1].set_title("Figure 2.2: Cumulative Trend Correlation: $V_{\\mathrm{surface}}$ vs. Raw $\\mathrm{TWSA}$",
                   fontsize=11, fontweight="bold")
axes2[1].set_xlabel("$\mathrm{TWSA}$ Anomaly (cm)", fontsize=10)
axes2[1].set_ylabel("Cumulative Volume $V_{\\mathrm{surface}}$ (BCM)", fontsize=10)
axes2[1].legend(loc="upper left")
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Fig2_correlation_scatter.png", dpi=150)


# =============================================================================
# FIGURE 3: Annual Net Storage Gains (delV Progression)
# =============================================================================
annual_delV = valid.groupby("year")["delV_surface_bcm"].sum()

fig3, ax3 = plt.subplots(figsize=(8, 5))
bars = ax3.bar(annual_delV.index, annual_delV.values, color="steelblue", edgecolor="black", alpha=0.85)

# Highlight positive vs negative years
for bar in bars:
    if bar.get_height() < 0:
        bar.set_color("indianred")
        bar.set_edgecolor("black")

ax3.set_title("Figure 3: Annual Net Surface Storage Gain ($\sum\\Delta V_{\\mathrm{surface}}$)",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Year", fontsize=10)
ax3.set_ylabel("Net Annual Gain (BCM/year)", fontsize=10)
ax3.axhline(0, color="gray", linestyle="--", lw=1)
ax3.grid(True, alpha=0.3, axis="y")

# Add numerical values on top of bars
for bar in bars:
    yval = bar.get_height()
    va_setting = "bottom" if yval >= 0 else "top"
    ax3.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:+.2f}",
             ha="center", va=va_setting, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("Fig3_annual_storage_gains.png", dpi=150)


# =============================================================================
# FIGURE 4: Monthly Climatology (Monsoon Seasonality)
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(9, 5))

monthly_data = [valid.loc[valid["month"] == m, "delV_surface_bcm"].values for m in range(1, 13)]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

bp = ax4.boxplot(monthly_data, labels=month_labels, patch_artist=True,
                 medianprops=dict(color="black", lw=1.5))

# Highlight July-Sept monsoon months in dark blue
for i, patch in enumerate(bp["boxes"]):
    if i in [6, 7, 8]:  # Jul, Aug, Sep
        patch.set_facecolor("navy")
        patch.set_alpha(0.7)
    else:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.6)

ax4.axhline(0, color="gray", linestyle=":", lw=1)
ax4.set_title("Figure 4: Seasonal Climatology of Monthly Change ($\\Delta V_{\\mathrm{surface}}$)",
              fontsize=11, fontweight="bold")
ax4.set_xlabel("Month", fontsize=10)
ax4.set_ylabel("$\\Delta V_{\\mathrm{surface}}$ (BCM/month)", fontsize=10)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("Fig4_monthly_climatology.png", dpi=150)

plt.show()