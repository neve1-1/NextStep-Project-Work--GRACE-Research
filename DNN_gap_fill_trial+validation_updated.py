"""
=============================================================================
GRACE TWSA (Total Water Storage Anomaly) Gap-Filling with a Neural Network
=============================================================================
This script trains a small feedforward neural net to predict next month's
Texas water storage anomaly from the previous 6 months, then uses that model
to fill gaps in the GRACE satellite record.
=============================================================================
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

# =============================================================================
# 1. LOAD DATA FROM GRACE NetCDF FILE
# =============================================================================
DATA_PATH = r"C:\Users\grace\Downloads\GRACE Research Project\3avg_files\CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"

dataset_nc = nc.Dataset(DATA_PATH)
print(list(dataset_nc.variables.keys()))
for name, var in dataset_nc.variables.items():
    print(f"{name}: {var.shape}")
# NOTE: dataset_nc.close() was being called here AND again further down right
# before we actually read lwe_thickness. That second close() would have made
# the lwe_thickness read below fail on a closed file handle, so it's removed
# from this spot and only kept after we're truly done reading.

# Get the dates
time_var = dataset_nc.variables['time']
times = nc.num2date(time_var[:], units=time_var.Units, calendar=getattr(time_var, 'calendar', 'standard'))
dates = pd.to_datetime([t.strftime('%Y-%m-%d') for t in times])

# Average all grid pixels inside Texas's bounding box
lats = dataset_nc.variables['lat'][:]
lons = dataset_nc.variables['lon'][:]  # 0-360 format

lat_idx = np.where((lats >= 25.8) & (lats <= 36.5))[0]
lon_idx = np.where((lons >= 360 - 106.6) & (lons <= 360 - 93.5))[0]

lwe = dataset_nc.variables['lwe_thickness']
twsa_texas = np.nanmean(lwe[:, lat_idx, :][:, :, lon_idx], axis=(1, 2))
dataset_nc.close()  # we're done reading from the file

df = pd.DataFrame({'Date': dates, 'Texas (in)': twsa_texas * 0.3937})  # cm to inches
target_col = "Texas (in)"

# Snap to strict monthly index so real gaps become explicit NaN rows
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').drop_duplicates(subset='Date')
df = df.set_index('Date')[[target_col]].resample('MS').mean().reset_index()

# =============================================================================
# 2. GENERATE 6 SEPARATE LAG CSV FILES (one lag at a time, for inspection)
# =============================================================================
for i in range(1, 7):
    lag_df = df[['Date', target_col]].copy()
    lag_df[f'Lag_{i}'] = lag_df[target_col].shift(i)
    lag_df.dropna().to_csv(f"twsa_lag_{i}.csv", index=False)

print("[INFO] Generated 6 independent lag CSV files (twsa_lag_1.csv to twsa_lag_6.csv)")

# =============================================================================
# 3. BUILD COMBINED LAG TABLE FOR TRAINING
# =============================================================================
N_LAGS = 6  # pulled out as a constant since it's reused

dataset = df[['Date', target_col]].copy()
for i in range(1, N_LAGS + 1):
    dataset[f'Lag_{i}'] = dataset[target_col].shift(i)
dataset = dataset.dropna()

X = dataset[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]].values
y = dataset[target_col].values

# =============================================================================
# 4. TRAIN/TEST SPLIT + NORMALIZE
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. BUILD MODEL
# 6 lag inputs -> 16 neurons -> 16 neurons -> 16 neurons -> 1 output
# =============================================================================
def build_model(n_lags=N_LAGS):
    """Factory function so we can build a fresh, identically-shaped model
    for the cross-validation folds in Section 6 as well as the final model
    in Section 7, without copy-pasting the architecture twice."""
    m = Sequential([
        Dense(16, activation='relu', input_shape=(n_lags,)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

# =============================================================================
# 6. TIME-SERIES CROSS-VALIDATION is the single train/test split lucky?
# =============================================================================

print("\n[INFO] Running time-series cross-validation (stability check)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_mae_scores = []

for fold_num, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(X_train), start=1):
    X_cv_train, X_cv_val = X_train[cv_train_idx], X_train[cv_val_idx]
    y_cv_train, y_cv_val = y_train[cv_train_idx], y_train[cv_val_idx]

    # fresh scaler per fold, same leakage logic as Section 4, just repeated
    # for each fold's own training slice
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
print("  -> Large std relative to the mean means performance is unstable")
print("     across different historical periods -- worth investigating before")
print("     trusting the final model below.")

# =============================================================================
# 7. TRAIN THE FINAL MODEL (with early stopping)
# =============================================================================

print("\n[INFO] Starting Vanilla Neural Network Training (with early stopping)...")
model = build_model()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,            # wait 10 epochs of no improvement before stopping
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=200,             # raised from 60 -- early stopping decides the
                            # real stopping point, so this is just a ceiling
    validation_split=0.1,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# 8. (NEW) PLOT LEARNING CURVES -- is the model overfitting?
# =============================================================================

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
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
# 9. EVALUATE: ONE-STEP-AHEAD ACCURACY 
# =============================================================================
#
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n[EVALUATION] One-step-ahead Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

# =============================================================================
# 10. GAP-FILLING HELPER FUNCTIONS
# (moved up from the bottom of the original script so Section 11 below can
#  reuse them for validation BEFORE we trust them for the real gap-filling
#  in Section 12)
# =============================================================================
def find_missing_value_gaps(series):
    """Find every stretch of consecutive NaN months and return their info."""
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
    """Predict one month at a time, feeding each guess back in as if it were
    real. This is the function responsible for the "snowball effect" -- the
    longer num_missing_months is, the more guesses get built on top of other
    guesses, and the shakier the result gets. Section 11 measures exactly how
    shaky, using real held-out data, before we use this on actual unknown gaps."""
    history = list(history_values)
    predictions = []
    for _ in range(num_missing_months):
        feature_vec = history[-n_lags:][::-1]  # [t-1, t-2, ..., t-6]
        x_scaled = scaler.transform(np.array(feature_vec, dtype=float).reshape(1, -1))
        y_pred = model.predict(x_scaled, verbose=0)[0, 0]
        predictions.append(y_pred)
        history.append(y_pred)
    return predictions


def fill_all_gaps(model, scaler, series, n_lags=N_LAGS):
    """Fill every gap in the series. Returns a dict of {gap_start: filled values}."""
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
# 11. (NEW) MULTI-STEP RECURSIVE VALIDATION -- the "hard" job, tested honestly
# =============================================================================

print("\n[INFO] Running multi-step recursive validation on the held-out test set...")

train_size = len(y_train)

seed_history = dataset[target_col].values[:train_size][-N_LAGS:]

horizon = len(y_test)
recursive_preds = np.array(fill_gap_recursive_mlp(model, scaler, seed_history, horizon, n_lags=N_LAGS))

step_errors = np.abs(recursive_preds - y_test)  # absolute error at each forecast step

print("  Recursive forecast error by horizon length (how far into the 'gap'):")
for checkpoint in [1, 3, 6, 12]:
    if checkpoint <= horizon:
        mae_at_checkpoint = step_errors[:checkpoint].mean()
        print(f"    First {checkpoint:>2} month(s) ahead -> MAE = {mae_at_checkpoint:.4f} in")

print(f"  Full {horizon}-month horizon -> MAE = {step_errors.mean():.4f} in "
      f"(compare this to the one-step Test MAE of {test_mae:.4f} above --")
print(f"  a much bigger number here confirms the model's errors are compounding "
      f"over multi-month gaps.)")

plt.figure(figsize=(8, 4))
plt.plot(range(1, horizon + 1), step_errors, marker='o', markersize=3)
plt.xlabel('Months into the recursive forecast')
plt.ylabel('Absolute error (in)')
plt.title('Does error grow the longer the recursive forecast runs?')
plt.tight_layout()
plt.savefig('twsa_multistep_validation.png', dpi=150)
plt.show()
print("[INFO] Multi-step validation plot saved to twsa_multistep_validation.png")
print("  -> If this line trends upward, long gaps should be trusted less than")
print("     short gaps. Consider flagging or excluding very long gap-fills.")

# =============================================================================
# 12. GAP FILLING (apply the validated model to the real missing months)
# =============================================================================
series = df.set_index('Date')[target_col]

complete_series = series.copy()
for filled_values in fill_all_gaps(model, scaler, series).values():
    complete_series.update(filled_values)

complete_series.to_csv("twsa_gapfilled_dnn.csv", header=[target_col])
print("Done. Complete gap-free monthly series saved to twsa_gapfilled_dnn.csv")

# =============================================================================
# 13. COMPARE AGAINST ACTUAL STRAWS DATA + DISCREPANCY METRICS
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

# Convert both series to anomalies from their own long-term mean
# before computing any metrics. This removes the reference-frame offset between
# GRACE (baseline: 2004-2009 satellite mean) and STRAWS (its own monitoring
# network baseline), so the metrics reflect shape/timing agreement rather than
# baseline mismatch.

comp_anom = comp - comp.mean()

diff_raw  = comp['Actual STRAWS'] - comp['GRACE DNN Gap-Filled']
diff_anom = comp_anom['Actual STRAWS'] - comp_anom['GRACE DNN Gap-Filled']

# Raw metrics (kept for reference -- bias here is expected and explainable bc using straws)
mae_raw  = np.mean(np.abs(diff_raw))
rmse_raw = np.sqrt(np.mean(diff_raw**2))
bias_raw = diff_raw.mean()
corr_raw = comp['Actual STRAWS'].corr(comp['GRACE DNN Gap-Filled'])

# Anomaly metrics (the defensible comparison for a research context)
mae_anom  = np.mean(np.abs(diff_anom))
rmse_anom = np.sqrt(np.mean(diff_anom**2))
bias_anom = diff_anom.mean()  # should be ~0 by construction
corr_anom = comp_anom['Actual STRAWS'].corr(comp_anom['GRACE DNN Gap-Filled'])

# Amplitude measured on anomaly series so damping comparison is
# baseline-neutral
amp_straws = comp_anom['Actual STRAWS'].max() - comp_anom['Actual STRAWS'].min()
amp_grace  = comp_anom['GRACE DNN Gap-Filled'].max() - comp_anom['GRACE DNN Gap-Filled'].min()

metrics_text = (
    f"MAE:            {mae_anom:.3f} in\n"
    f"RMSE:           {rmse_anom:.3f} in\n"
    f"Bias:           {bias_anom:.3f} in\n"
    f"Correlation:    {corr_anom:.3f}\n"
    f"Amplitude diff: {abs(amp_straws - amp_grace):.3f} in"
)

print("=== Discrepancy Metrics ===")
print(metrics_text)

# Two-panel plot: raw on top, anomaly on bottom
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Top panel — raw values (shows the baseline offset visually)
ax1.plot(comp.index, comp['Actual STRAWS'],        label='Actual STRAWS',        color='steelblue')
ax1.plot(comp.index, comp['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled', color='darkorange', linestyle='--')
ax1.set_title('Raw TWSA — baseline offset is visible here')
ax1.set_ylabel('TWSA (inches)')
ax1.legend(loc='upper right')

# Bottom panel — anomaly series (the apples-to-apples comparison)
ax2.plot(comp_anom.index, comp_anom['Actual STRAWS'],        label='Actual STRAWS (anomaly)',        color='steelblue')
ax2.plot(comp_anom.index, comp_anom['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled (anomaly)', color='darkorange', linestyle='--')
ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax2.set_title('Anomaly-from-mean TWSA — apples-to-apples comparison')
ax2.set_xlabel('Date')
ax2.set_ylabel('TWSA anomaly (inches)')
ax2.legend(loc='upper right')

# Metrics box on bottom panel
ax2.text(0.01, 0.02, metrics_text, transform=ax2.transAxes,
         fontsize=7.5, verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('twsa_comparison.png', dpi=150)
plt.show()
print("Comparison plot saved to twsa_comparison.png")

# Also print a quick plain-language interpretation
if corr_anom > corr_raw:
    print("[INFO] Anomaly correlation is higher than raw -- confirms the offset was masking real agreement.")
if mae_anom < mae_raw * 0.6:
    print("[INFO] Anomaly MAE is dramatically lower than raw MAE -- the bias was dominating the error metric.")
print(f"[INFO] Amplitude ratio (GRACE/STRAWS): {amp_grace/amp_straws:.3f} "
      f"(1.0 = perfect, <1.0 = model is damping extremes as expected from recursive forecasting)")
# =============================================================================
# 14. SUMMARY OF WHAT EACH NEW DIAGNOSTIC TELLS YOU
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF VALIDATION DIAGNOSTICS ADDED IN THIS VERSION")
print("=" * 70)
print(f"- CV stability (Section 6):      mean MAE {np.mean(cv_mae_scores):.4f}, "
      f"std {np.std(cv_mae_scores):.4f} across 5 folds")
print(f"- Overfitting check (Section 8): train loss {final_train_loss:.4f} vs "
      f"val loss {final_val_loss:.4f}")
print(f"- One-step test MAE (Section 9): {test_mae:.4f} in")
print(f"- Multi-step recursive MAE (Section 11), full {horizon}-month horizon: "
      f"{step_errors.mean():.4f} in")
print(f"- Final vs. independent STRAWS dataset (Section 13): MAE {mae:.3f} in, "
      f"correlation {corr:.3f}")
print("If the multi-step MAE is much worse than the one-step MAE, that gap is")
print("the main thing to bring back to your mentor -- it's the real measure of")
print("how trustworthy the gap-filled months are, not the one-step number alone.")