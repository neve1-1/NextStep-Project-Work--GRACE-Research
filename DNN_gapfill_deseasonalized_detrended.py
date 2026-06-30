"""
GRACE TWSA gap-filling with a neural net, detrended/deseasonalized version.

Same pipeline as before (3-product ensemble -> lags -> MLP -> recursive
gap-fill -> STRAWS comparison), but the net is trained on the residual
(trend + seasonal cycle removed) instead of raw TWSA. Otherwise the MLP
just learns to repeat last month's value + climatology, which inflates
apparent skill and biases the recursive fill once the seasonal anchor
is gone. Trend/seasonal get added back in afterward since they're just
deterministic functions of the date.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import netCDF4 as nc
from datetime import datetime, timedelta

# =============================================================================
# 1. Load CSR/JPL/GSFC, build ensemble mean TWSA
# =============================================================================
FILES = {
    "CSR":  r"C:\Users\grace\Downloads\GRACE Research Project\3avg_files\CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
    "JPL":  r"C:\Users\grace\Downloads\GRACE Research Project\3avg_files\GRCTellus.JPL.nc",
    "GSFC": r"C:\Users\grace\Downloads\GRACE Research Project\3avg_files\gsfc.glb_.200204_202511_rl06v2.0_obp-ice6gd_halfdegree.nc",
}

TX = dict(lat_min=25.8, lat_max=36.5, lon_min=-106.6, lon_max=-93.5)


def decode_time(time_var):
    units = getattr(time_var, 'units', None) or getattr(time_var, 'Units', '')
    origin_str = units.split("since")[1].strip().replace("T", " ").replace("Z", "")
    origin = datetime.strptime(origin_str[:19], "%Y-%m-%d %H:%M:%S")
    return [origin + timedelta(days=float(d)) for d in time_var[:]]


def clip_and_mean(ds_name, fpath, bbox):
    ds   = nc.Dataset(fpath)
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    lwe  = ds.variables['lwe_thickness'][:]
    times = decode_time(ds.variables['time'])
    ds.close()

    lon_min_360 = bbox['lon_min'] + 360
    lon_max_360 = bbox['lon_max'] + 360

    lat_idx = np.where((lats >= bbox['lat_min']) & (lats <= bbox['lat_max']))[0]
    lon_idx = np.where((lons >= lon_min_360)     & (lons <= lon_max_360))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"{ds_name}: no grid cells found inside bounding box.")

    clipped  = lwe[:, lat_idx[:, None], lon_idx[None, :]]
    sub_lats = lats[lat_idx]

    w   = np.cos(np.deg2rad(sub_lats))
    w2d = np.tile(w[:, None], (1, len(lon_idx)))

    nt = clipped.shape[0]
    ts = np.full(nt, np.nan)
    for t in range(nt):
        frame   = clipped[t]
        mask    = np.ma.getmaskarray(frame)
        w_valid = np.where(mask, 0.0, w2d)
        total_w = w_valid.sum()
        if total_w > 0:
            ts[t] = (frame.filled(0.0) * w_valid).sum() / total_w

    print(f"  {ds_name}: {nt} months clipped, TX mean = {np.nanmean(ts):.2f} cm")
    return times, ts


def to_days(times, epoch):
    return np.array([(t - epoch).days for t in times], dtype=float)


print("Clipping CSR to Texas...")
csr_times, csr_ts = clip_and_mean("CSR", FILES["CSR"], TX)
print("Clipping JPL to Texas...")
jpl_times, jpl_ts = clip_and_mean("JPL", FILES["JPL"], TX)
print("Clipping GSFC to Texas...")
gsfc_times, gsfc_ts = clip_and_mean("GSFC", FILES["GSFC"], TX)

EPOCH = datetime(2002, 1, 1)

all_days = np.union1d(
    np.union1d(to_days(csr_times, EPOCH), to_days(jpl_times, EPOCH)),
    to_days(gsfc_times, EPOCH),
)

interp_csr  = np.interp(all_days, to_days(csr_times, EPOCH),  csr_ts,  left=np.nan, right=np.nan)
interp_jpl  = np.interp(all_days, to_days(jpl_times, EPOCH),  jpl_ts,  left=np.nan, right=np.nan)
interp_gsfc = np.interp(all_days, to_days(gsfc_times, EPOCH), gsfc_ts, left=np.nan, right=np.nan)

stack = np.vstack([interp_csr, interp_jpl, interp_gsfc])
ensemble_mean_cm = np.nanmean(stack, axis=0)

dates = pd.to_datetime([EPOCH + timedelta(days=int(d)) for d in all_days])
twsa_texas = ensemble_mean_cm

print(f"\n[INFO] Ensemble mean built from {len(FILES)} datasets "
      f"({', '.join(FILES.keys())}), {len(dates)} time steps total.")

df = pd.DataFrame({'Date': dates, 'Texas (in)': twsa_texas * 0.3937})
target_col = "Texas (in)"

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').drop_duplicates(subset='Date')
df = df.set_index('Date')[[target_col]].resample('MS').mean().reset_index()

# =============================================================================
# 2. Remove trend + seasonality -> Residual column
# =============================================================================
print("\n[INFO] Removing linear trend and monthly seasonal cycle...")

df['t_months'] = (df['Date'] - df['Date'].iloc[0]).dt.days / 30.4375
valid_mask = df[target_col].notna()

trend_coefs = np.polyfit(df.loc[valid_mask, 't_months'], df.loc[valid_mask, target_col], 1)
df['Trend'] = np.polyval(trend_coefs, df['t_months'])

detrended = df[target_col] - df['Trend']

df['Month'] = df['Date'].dt.month
climatology = detrended[valid_mask].groupby(df.loc[valid_mask, 'Month']).mean()
df['Seasonal'] = df['Month'].map(climatology)

df['Residual'] = df[target_col] - df['Trend'] - df['Seasonal']

print(f"[INFO] Trend: {trend_coefs[0]:.5f} in/month "
      f"({trend_coefs[0]*12:.4f} in/year)")
print("[INFO] Seasonal climatology (in, detrended anomaly by calendar month):")
for m in range(1, 13):
    if m in climatology.index:
        print(f"    Month {m:>2}: {climatology[m]:+.3f}")

residual_col = "Residual"

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(df['Date'], df[target_col], color='steelblue')
axes[0].set_title('Raw TWSA (with trend + seasonality)')
axes[1].plot(df['Date'], df['Trend'], color='darkorange', label='Trend')
axes[1].plot(df['Date'], df['Trend'] + df['Seasonal'], color='seagreen',
             label='Trend + Seasonal', alpha=0.7)
axes[1].set_title('Fitted trend and trend+seasonal climatology')
axes[1].legend()
axes[2].plot(df['Date'], df[residual_col], color='purple')
axes[2].axhline(0, color='gray', linewidth=0.8, linestyle=':')
axes[2].set_title('Residual (raw - trend - seasonal) -- this is the modeling target')
axes[2].set_xlabel('Date')
plt.tight_layout()
plt.savefig('twsa_decomposition.png', dpi=150)
plt.show()
print("[INFO] Decomposition plot saved to twsa_decomposition.png")

# =============================================================================
# 3. Lag CSVs (per-lag, residual-based) for inspection
# =============================================================================
for i in range(1, 7):
    lag_df = df[['Date', residual_col]].copy()
    lag_df[f'Lag_{i}'] = lag_df[residual_col].shift(i)
    lag_df.dropna().to_csv(f"twsa_residual_lag_{i}.csv", index=False)

print("[INFO] Generated 6 independent lag CSV files "
      "(twsa_residual_lag_1.csv to twsa_residual_lag_6.csv)")

# =============================================================================
# 4. Combined lag table for training
# =============================================================================
N_LAGS = 6

dataset = df[['Date', residual_col]].copy()
for i in range(1, N_LAGS + 1):
    dataset[f'Lag_{i}'] = dataset[residual_col].shift(i)
dataset = dataset.dropna()

X = dataset[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]].values
y = dataset[residual_col].values

# =============================================================================
# 5. Train/test split + normalize
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 6. Model: 6 lags -> 16 -> 16 -> 16 -> 1
# =============================================================================
def build_model(n_lags=N_LAGS):
    m = Sequential([
        Dense(16, activation='relu', input_shape=(n_lags,)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

# =============================================================================
# 7. Time-series CV, just to check the single split wasn't lucky
# =============================================================================
print("\n[INFO] Running time-series cross-validation (stability check)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_mae_scores = []

for fold_num, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(X_train), start=1):
    X_cv_train, X_cv_val = X_train[cv_train_idx], X_train[cv_val_idx]
    y_cv_train, y_cv_val = y_train[cv_train_idx], y_train[cv_val_idx]

    cv_scaler = StandardScaler()
    X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
    X_cv_val_scaled = cv_scaler.transform(X_cv_val)

    cv_model = build_model()
    cv_model.fit(X_cv_train_scaled, y_cv_train, epochs=30, batch_size=8, verbose=0)
    _, fold_mae = cv_model.evaluate(X_cv_val_scaled, y_cv_val, verbose=0)
    cv_mae_scores.append(fold_mae)
    print(f"  Fold {fold_num}: validation MAE = {fold_mae:.4f}")

print(f"[INFO] CV MAE across folds: mean={np.mean(cv_mae_scores):.4f}, "
      f"std={np.std(cv_mae_scores):.4f}")
print("  -> residual units now, not comparable to old raw-TWSA CV numbers.")

# =============================================================================
# 8. Train final model with early stopping
# =============================================================================
print("\n[INFO] Starting Vanilla Neural Network Training (with early stopping)...")
model = build_model()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=18,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    validation_split=0.1,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# 9. Learning curves
# =============================================================================
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss (residual units)')
plt.title('Training vs. validation loss (gap between lines = overfitting)')
plt.legend()
plt.tight_layout()
plt.savefig('twsa_learning_curves.png', dpi=150)
plt.show()
print("[INFO] Learning curve plot saved to twsa_learning_curves.png")

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
if final_val_loss > 1.5 * final_train_loss:
    print(f"[WARNING] Validation loss ({final_val_loss:.4f}) is notably higher "
          f"than training loss ({final_train_loss:.4f}). This is a sign of "
          f"overfitting -- consider fewer neurons, fewer lags, or more data.")

# =============================================================================
# 10. One-step-ahead accuracy (residual units)
# =============================================================================
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n[EVALUATION] One-step-ahead Test MSE: {test_loss:.4f} | "
      f"Test MAE: {test_mae:.4f} (residual units, in)")

# =============================================================================
# 11. Gap-filling helpers (operate on Residual)
# =============================================================================
def find_missing_value_gaps(series):
    gaps, current_gap_start = [], None
    for date, missing in series.isna().items():
        if missing and current_gap_start is None:
            current_gap_start = date
        elif not missing and current_gap_start is not None:
            gap_dates = series.index[(series.index >= current_gap_start) & (series.index < date)]
            gaps.append({"start_date": current_gap_start, "end_date": gap_dates[-1], "num_missing_months": len(gap_dates)})
            current_gap_start = None
    if current_gap_start is not None:
        gap_dates = series.index[series.index >= current_gap_start]
        gaps.append({"start_date": current_gap_start, "end_date": gap_dates[-1], "num_missing_months": len(gap_dates)})
    return gaps


def fill_gap_recursive_mlp(model, scaler, history_values, num_missing_months, n_lags=N_LAGS):
    history = list(history_values)
    predictions = []
    for _ in range(num_missing_months):
        feature_vec = history[-n_lags:][::-1]
        x_scaled = scaler.transform(np.array(feature_vec, dtype=float).reshape(1, -1))
        y_pred = model.predict(x_scaled, verbose=0)[0, 0]
        predictions.append(y_pred)
        history.append(y_pred)
    return predictions


def fill_all_gaps(model, scaler, series, n_lags=N_LAGS):
    filled_segments = {}
    for gap in find_missing_value_gaps(series):
        print(f"Filling gap: {gap['start_date'].date()} to {gap['end_date'].date()} ({gap['num_missing_months']} months)")
        before_gap = series[series.index < gap["start_date"]].dropna()
        if len(before_gap) < n_lags:
            print(f"  Skipping -- not enough real data before this gap (need {n_lags} months).")
            continue
        filled_values = fill_gap_recursive_mlp(model, scaler, before_gap.values[-n_lags:], gap["num_missing_months"], n_lags)
        gap_dates = pd.date_range(start=gap["start_date"], periods=gap["num_missing_months"], freq="MS")
        filled_segments[gap["start_date"]] = pd.Series(filled_values, index=gap_dates)
    return filled_segments

# =============================================================================
# 12. Multi-step recursive validation on the held-out test set
# =============================================================================
print("\n[INFO] Running multi-step recursive validation on the held-out test set...")

train_size = len(y_train)
seed_history = dataset[residual_col].values[:train_size][-N_LAGS:]

horizon = len(y_test)
recursive_preds = np.array(fill_gap_recursive_mlp(model, scaler, seed_history, horizon, n_lags=N_LAGS))

step_errors = np.abs(recursive_preds - y_test)

print("  Recursive forecast error by horizon length (how far into the 'gap'):")
for checkpoint in [1, 3, 6, 12]:
    if checkpoint <= horizon:
        mae_at_checkpoint = step_errors[:checkpoint].mean()
        print(f"    First {checkpoint:>2} month(s) ahead -> MAE = {mae_at_checkpoint:.4f} in")

print(f"  Full {horizon}-month horizon -> MAE = {step_errors.mean():.4f} in "
      f"(compare this to the one-step Test MAE of {test_mae:.4f} above --")
print(f"  a much bigger number here confirms the model's errors are compounding "
      f"over multi-month gaps.)")
print("  -> seasonal cycle is already removed, so this MAE is genuine")
print("     non-seasonal forecast skill, not just re-predicting climatology.")

plt.figure(figsize=(8, 4))
plt.plot(range(1, horizon + 1), step_errors, marker='o', markersize=3)
plt.xlabel('Months into the recursive forecast')
plt.ylabel('Absolute error, residual units (in)')
plt.title('Does error grow the longer the recursive forecast runs?')
plt.tight_layout()
plt.savefig('twsa_multistep_validation.png', dpi=150)
plt.show()
print("[INFO] Multi-step validation plot saved to twsa_multistep_validation.png")

# =============================================================================
# 13. Gap filling + reconstruction (residual -> back to actual TWSA)
# =============================================================================
residual_series = df.set_index('Date')[residual_col]

complete_residual = residual_series.copy()
for filled_values in fill_all_gaps(model, scaler, residual_series).values():
    complete_residual.update(filled_values)

trend_lookup = df.set_index('Date')['Trend']
seasonal_lookup = df.set_index('Date')['Seasonal']

complete_series = complete_residual + trend_lookup.reindex(complete_residual.index) \
                                     + seasonal_lookup.reindex(complete_residual.index)
complete_series.name = target_col

complete_residual.to_csv("twsa_residual_gapfilled_dnn.csv", header=[residual_col])
complete_series.to_csv("twsa_gapfilled_dnn.csv", header=[target_col])
print("Done. Complete gap-free monthly series (reconstructed: residual + "
      "trend + seasonal) saved to twsa_gapfilled_dnn.csv")
print("Residual-only series also saved to twsa_residual_gapfilled_dnn.csv")

# =============================================================================
# 14. Compare against actual STRAWS data
# =============================================================================
actual = pd.read_csv(r"C:\Users\grace\Downloads\GRACE Research Project\Texas-Statewide-English-1782253537240(in).csv")
actual.columns = actual.columns.str.strip()
actual['Date'] = pd.to_datetime(actual['Date']).dt.to_period('M').dt.to_timestamp()
actual = actual[['Date', 'Texas (in)']].rename(columns={'Texas (in)': 'Actual STRAWS'}).set_index('Date')

grace = complete_series.copy()
grace.index = grace.index.to_period('M').to_timestamp()

comparison = pd.DataFrame({
    'Actual STRAWS': actual['Actual STRAWS'],
    'GRACE DNN Gap-Filled': grace
})

comp = comparison.dropna()

comp_anom = comp - comp.mean()

diff_raw  = comp['Actual STRAWS'] - comp['GRACE DNN Gap-Filled']
diff_anom = comp_anom['Actual STRAWS'] - comp_anom['GRACE DNN Gap-Filled']

mae_raw  = np.mean(np.abs(diff_raw))
rmse_raw = np.sqrt(np.mean(diff_raw**2))
bias_raw = diff_raw.mean()
corr_raw = comp['Actual STRAWS'].corr(comp['GRACE DNN Gap-Filled'])

mae_anom  = np.mean(np.abs(diff_anom))
rmse_anom = np.sqrt(np.mean(diff_anom**2))
bias_anom = diff_anom.mean()
corr_anom = comp_anom['Actual STRAWS'].corr(comp_anom['GRACE DNN Gap-Filled'])

amp_straws = comp_anom['Actual STRAWS'].max() - comp_anom['Actual STRAWS'].min()
amp_grace  = comp_anom['GRACE DNN Gap-Filled'].max() - comp_anom['GRACE DNN Gap-Filled'].min()

metrics_text = (
    f"MAE:            {mae_anom:.3f} in\n"
    f"RMSE:           {rmse_anom:.3f} in\n"
    f"Bias:           {bias_anom:.3f} in\n"
    f"Correlation:    {corr_anom:.3f}\n"
    f"Amplitude diff: {abs(amp_straws - amp_grace):.3f} in"
)

print("=== Discrepancy Metrics (reconstructed TWSA vs. STRAWS) ===")
print(metrics_text)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax1.plot(comp.index, comp['Actual STRAWS'],        label='Actual STRAWS',        color='steelblue')
ax1.plot(comp.index, comp['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled', color='darkorange', linestyle='--')
ax1.set_title('Raw TWSA (reconstructed) — baseline offset is visible here')
ax1.set_ylabel('TWSA (inches)')
ax1.legend(loc='upper right')

ax2.plot(comp_anom.index, comp_anom['Actual STRAWS'],        label='Actual STRAWS (anomaly)',        color='steelblue')
ax2.plot(comp_anom.index, comp_anom['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled (anomaly)', color='darkorange', linestyle='--')
ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax2.set_title('Anomaly-from-mean TWSA — apples-to-apples comparison')
ax2.set_xlabel('Date')
ax2.set_ylabel('TWSA anomaly (inches)')
ax2.legend(loc='upper right')

ax2.text(0.01, 0.02, metrics_text, transform=ax2.transAxes,
         fontsize=7.5, verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('twsa_comparison.png', dpi=150)
plt.show()
print("Comparison plot saved to twsa_comparison.png")

if corr_anom > corr_raw:
    print("[INFO] Anomaly correlation is higher than raw -- confirms the offset was masking real agreement.")
if mae_anom < mae_raw * 0.6:
    print("[INFO] Anomaly MAE is dramatically lower than raw MAE -- the bias was dominating the error metric.")
print(f"[INFO] Amplitude ratio (GRACE/STRAWS): {amp_grace/amp_straws:.3f} "
      f"(1.0 = perfect, <1.0 = model is damping extremes as expected from recursive forecasting)")

# =============================================================================
# 15. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF VALIDATION DIAGNOSTICS (TREND + SEASONALITY REMOVED VERSION)")
print("=" * 70)
print(f"- Trend removed:                 {trend_coefs[0]*12:.4f} in/year")
print(f"- Seasonal cycle removed:        monthly climatology (see printout above)")
print(f"- CV stability (Section 7):      mean MAE {np.mean(cv_mae_scores):.4f}, "
      f"std {np.std(cv_mae_scores):.4f} across 5 folds (residual units)")
print(f"- Overfitting check (Section 9): train loss {final_train_loss:.4f} vs "
      f"val loss {final_val_loss:.4f} (residual units)")
print(f"- One-step test MAE (Section 10): {test_mae:.4f} in (residual units)")
print(f"- Multi-step recursive MAE (Section 12), full {horizon}-month horizon: "
      f"{step_errors.mean():.4f} in (residual units)")
print(f"- Final vs. independent STRAWS dataset (Section 14, reconstructed TWSA): "
      f"MAE {mae_anom:.3f} in, correlation {corr_anom:.3f}")
print("Residual-unit MAEs aren't directly comparable to a raw-TWSA model's MAEs --")
print("the seasonal cycle (usually the biggest source of month-to-month variance)")
print("is already subtracted out, so these numbers look smaller even if the")
print("underlying skill is the same. Section 14 is the fair comparison point.")