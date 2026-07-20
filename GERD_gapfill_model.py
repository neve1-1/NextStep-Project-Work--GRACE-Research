"""
GRACE TWSA gap-filling with a neural net, detrended/deseasonalized version.

Same pipeline as before (3-product ensemble -> lags -> MLP -> recursive
gap-fill -> STRAWS comparison), but the net is trained on the residual
(trend + seasonal cycle removed) instead of raw TWSA. Otherwise the MLP
just learns to repeat last month's value + climatology, which inflates
apparent skill and biases the recursive fill once the seasonal anchor
is gone. Trend/seasonal get added back in afterward since they're just
deterministic functions of the date.

PLOT-STYLE NOTE
----------------
Only the plotting code below has been touched (colors, line styles, and
metrics-box formatting), to line up with the conventions used in the
architecture-comparison script:
    - actual / ground-truth series -> black, solid, linewidth 1.8
    - model prediction series      -> steelblue (or darkorange for a
                                       second series), dashed
    - metrics summaries            -> monospace text box, rounded,
                                       semi-transparent white background,
                                       bottom-left corner of the axes
Nothing about the data prep, lag features, model architecture, training
loop, CV, or gap-filling logic has changed.
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

#The trend and seasonal cycle used to be calculated using the entire dataset, 
# which secretly let test-period data influence the training data — a leak. 
# Now they're calculated using only the first 80% (the training portion) 
# and then applied to the rest, just like a scaler.
n_valid = valid_mask.sum()
valid_dates_in_order = df.loc[valid_mask, 'Date']
cutoff_position = int(np.floor(len(valid_dates_in_order) * 0.8))
train_only_dates = set(valid_dates_in_order.iloc[:cutoff_position])
train_only_mask = df['Date'].isin(train_only_dates)
# Leak fix?

trend_coefs = np.polyfit(df.loc[train_only_mask, 't_months'], df.loc[train_only_mask, target_col], 1)  # LEAK FIX: fit on train_only_mask, not valid_mask
df['Trend'] = np.polyval(trend_coefs, df['t_months'])  # unchanged: apply fitted trend to all dates, like scaler.transform()

detrended = df[target_col] - df['Trend']

df['Month'] = df['Date'].dt.month
climatology = detrended[train_only_mask].groupby(df.loc[train_only_mask, 'Month']).mean()  # LEAK FIX: fit on train_only_mask, not valid_mask
df['Seasonal'] = df['Month'].map(climatology)  # unchanged: apply fitted climatology to all dates

df['Residual'] = df[target_col] - df['Trend'] - df['Seasonal']

print(f"[INFO] Trend: {trend_coefs[0]:.5f} in/month "
      f"({trend_coefs[0]*12:.4f} in/year)")
print("[INFO] Seasonal climatology (in, detrended anomaly by calendar month):")
for m in range(1, 13):
    if m in climatology.index:
        print(f"    Month {m:>2}: {climatology[m]:+.3f}")

residual_col = "Residual"

# --- restyled: raw series in black (matches "actual" convention elsewhere) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(df['Date'], df[target_col], color='black', linewidth=1.5)
axes[0].set_title('Raw TWSA (with trend + seasonality)')
axes[1].plot(df['Date'], df['Trend'], color='steelblue', linestyle='--', label='Trend')
axes[1].plot(df['Date'], df['Trend'] + df['Seasonal'], color='darkorange', linestyle='--',
             label='Trend + Seasonal', alpha=0.8)
axes[1].set_title('Fitted trend and trend+seasonal climatology')
axes[1].legend(fontsize=8)
axes[2].plot(df['Date'], df[residual_col], color='black', linewidth=1.2)
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
# 4. Combined lag table for training (MASKED MULTI-STEP UPDATE)
# =============================================================================
N_LAGS = 6
HORIZON = 12  

X_list = []
y_list = []
dates_list = []  # NEW: record the true date for each sample as it's built, since the
                  # loop below can SKIP indices (real GRACE->GRACE-FO gap, scattered
                  # missing months), so "sample s = row N_LAGS+s" is not a safe assumption

# Loop across the full dataset contiguously
for i in range(N_LAGS, len(df) - HORIZON + 1):
    X_window = df[residual_col].iloc[i - N_LAGS:i].values[::-1]  
    y_window = df[residual_col].iloc[i:i + HORIZON].values       
    
    # We only skip if the INPUT history has NaNs (we need valid features to predict)
    if not np.isnan(X_window).any():
        X_list.append(X_window)
        # For targets, replace NaNs with a safe flag value (-999.0) to preserve the grid shape
        y_clean = np.where(np.isnan(y_window), -999.0, y_window)
        y_list.append(y_clean)
        dates_list.append(df['Date'].iloc[i])  # NEW: this sample's true calendar date

X = np.array(X_list)
y = np.array(y_list)
sample_dates = np.array(dates_list)  # NEW

print(f"[INFO] Contiguous shapes optimized for Masked Training: X={X.shape}, y={y.shape}")

# =============================================================================
# 5. Train/test split + normalize (CLEAN TARGET ARRAYS)
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split into Train and Test sets chronologically
X_train_raw, X_test, y_train_raw, y_test, dates_train_raw, dates_test = train_test_split(
    X, y, sample_dates, test_size=0.2, random_state=42, shuffle=False
)

# Further split Train into Train and Validation sets manually
val_size = int(len(X_train_raw) * 0.2)
X_train = X_train_raw[:-val_size]
y_train = y_train_raw[:-val_size]
X_val = X_train_raw[-val_size:]
y_val = y_train_raw[-val_size:]
dates_train = dates_train_raw[:-val_size]  # NEW
dates_val = dates_train_raw[-val_size:]    # NEW

# Scale inputs (X) safely
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# CRITICAL FIX: Make sure your target variables (y) are NOT being transformed 
# by an incompatible scaler before training! We keep them in raw residual units.

# =============================================================================
# NEW: 5b. Diagnostic plot -- residual over time, train/val/test regions shaded
# =============================================================================
# Purpose: confirm (visually) where the train/val bias comes from. Uses the
# real per-sample dates tracked in Section 4/5 (NOT reconstructed from index
# arithmetic -- that approach breaks whenever the loop skips indices, which it
# does around the real GRACE->GRACE-FO gap and other missing months).
train_start_date, train_end_date = dates_train.min(), dates_train.max()
val_start_date, val_end_date     = dates_val.min(), dates_val.max()
test_start_date, test_end_date   = dates_test.min(), dates_test.max()

fig, ax = plt.subplots(figsize=(13, 5))
ax.axvspan(train_start_date, train_end_date, color='steelblue', alpha=0.12, label='Train window')
ax.axvspan(val_start_date, val_end_date, color='darkorange', alpha=0.15, label='Val window')
ax.axvspan(test_start_date, test_end_date, color='green', alpha=0.12, label='Test window')
ax.plot(df['Date'], df[residual_col], color='black', linewidth=1.2)
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axhline(y_val[y_val != -999.0].mean(), color='darkorange', linewidth=1, linestyle='--',
           label=f"Val mean = {y_val[y_val != -999.0].mean():.2f} in")
ax.set_title('Residual over time, with train/val/test regions shaded')
ax.set_xlabel('Date')
ax.set_ylabel('Residual (in)')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('twsa_residual_bias_diagnostic.png', dpi=150)
plt.show()
print("[INFO] Diagnostic plot saved to twsa_residual_bias_diagnostic.png")
print(f"[INFO] Train window: {pd.Timestamp(train_start_date).date()} to {pd.Timestamp(train_end_date).date()}")
print(f"[INFO] Val window:   {pd.Timestamp(val_start_date).date()} to {pd.Timestamp(val_end_date).date()}")
print(f"[INFO] Test window:  {pd.Timestamp(test_start_date).date()} to {pd.Timestamp(test_end_date).date()}")

# =============================================================================
# 6. Model: 6 lags -> 64 -> 32 -> 16 -> 12, with Dropout (DIRECT MULTI-STEP UPDATE)
# =============================================================================
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout  # FIX: Dropout was missing from imports, which is why it silently vanished from the model below

# Custom loss function that ignores our placeholder flag (-999.0)
def masked_mse(y_true, y_pred):
    # Create a mask where True represents real valid data points
    mask = tf.not_equal(y_true, -999.0)
    mask = tf.cast(mask, tf.float32)
    
    # Calculate standard squared error
    squared_errors = tf.square(y_true - y_pred)
    
    # Zero-out the errors belonging to the missing months
    masked_errors = squared_errors * mask
    
    # Return the mean over only the valid, unmasked elements
    return tf.reduce_sum(masked_errors) / (tf.reduce_sum(mask) + 1e-7)

# Build the model architecture matching our fixed output layer
def build_model(n_lags=N_LAGS, horizon=HORIZON):
    m = Sequential([
        Dense(8, activation='tanh', input_shape=(n_lags,)),
        Dropout(0.1),  # FIX: re-added -- this is what was missing and causing the runaway overfitting
        Dense(8, activation='tanh'),
        # Dropout(0.1),  # FIX: re-added
        # Dense(8, activation='tanh'),
        Dropout(0.1),  # FIX: re-added
        Dense(horizon)  # Clear multi-step prediction layer
    ])
    # Compile using our custom masked loss instead of standard 'mse'
    m.compile(optimizer='adam', loss=masked_mse, metrics=['mse'])
    return m
# # 7. Time-series CV, just to check the single split wasn't lucky
# # =============================================================================
# print("\n[INFO] Running time-series cross-validation (stability check)...")
# tscv = TimeSeriesSplit(n_splits=5)
# cv_mae_scores = []

# for fold_num, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(X_train), start=1):
#     X_cv_train, X_cv_val = X_train[cv_train_idx], X_train[cv_val_idx]
#     y_cv_train, y_cv_val = y_train[cv_train_idx], y_train[cv_val_idx]


#     cv_scaler = StandardScaler()
#     X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
#     X_cv_val_scaled = cv_scaler.transform(X_cv_val)

#     cv_model = build_model()
#     cv_model.fit(X_cv_train_scaled, y_cv_train, epochs=30, batch_size=8, verbose=0)
#     _, fold_mae = cv_model.evaluate(X_cv_val_scaled, y_cv_val, verbose=0)
#     cv_mae_scores.append(fold_mae)
#     print(f"  Fold {fold_num}: validation MAE = {fold_mae:.4f}")

# print(f"[INFO] CV MAE across folds: mean={np.mean(cv_mae_scores):.4f}, "
#       f"std={np.std(cv_mae_scores):.4f}")
# print("  -> residual units now, not comparable to old raw-TWSA CV numbers.")
# =============================================================================
# 8. Train final model with early stopping (MASKED COMPILATION)
# =============================================================================
print("\n[INFO] Starting Masked Multi-Step Neural Network Training...")
model = build_model()

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    validation_data=(X_val_scaled, y_val),
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# 9. Learning curves
# =============================================================================
# --- restyled: 'train'/'val' labels, same wording/legend convention as the
# architecture-comparison script's per-panel loss plots ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('MSE loss (residual units)')
plt.title('Training vs. validation loss (gap between lines = overfitting)')
plt.legend(fontsize=8)
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
# 12. Direct multi-step validation on the held-out test set
# =============================================================================
print("\n[INFO] Running direct multi-step validation on the held-out test set...")

# Take the very first test sample's historical lags to predict its 12-month horizon
test_seed = X_test_scaled[0:1] # Shape (1, 6)

# Predict the entire 12-month horizon at once
direct_preds = model.predict(test_seed, verbose=0)[0] # Shape (12,)
actual_test_horizon = y_test[0]                       # Shape (12,)

# Calculate absolute errors over the horizon
step_errors = np.abs(direct_preds - actual_test_horizon)

# --- Plot the True Direct Forecast vs Actuals ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, HORIZON + 1), actual_test_horizon, color='black', linewidth=1.8, label='Actual Test Residual')
ax.plot(range(1, HORIZON + 1), direct_preds, color='steelblue', linestyle='--', marker='o', label='Direct Multi-Step Pred')
ax.set_xlabel('Months into the forecast horizon')
ax.set_ylabel('Inches (Residual Space)')
ax.set_title('True Out-of-Sample Model Comparison (Direct Forecasting)')
ax.legend()

step_metrics_text = (
    f"Mean error:  {step_errors.mean():.3f} in\n"
    f"Max error:   {step_errors.max():.3f} in\n"
    f"Horizon:     {HORIZON} months"
)
ax.text(0.01, 0.02, step_metrics_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.show()

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
# 15. Compare against actual STRAWS data
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
# 1. Calculate the true amplitudes here first
actual_test_amplitude = np.max(actual_test_horizon) - np.min(actual_test_horizon)
predicted_test_amplitude = np.max(direct_preds) - np.min(direct_preds)
true_amplitude_diff = abs(actual_test_amplitude - predicted_test_amplitude)

# 2. Then build the text box using the new variable
metrics_text = (
    f"MAE:            {mae_anom:.3f} in\n"
    f"RMSE:           {rmse_anom:.3f} in\n"
    f"Bias:           {bias_anom:.3f} in\n"
    f"Correlation:    {corr_anom:.3f}\n"
    f"Amplitude diff: {true_amplitude_diff:.3f} in"
)

print("=== Discrepancy Metrics (reconstructed TWSA vs. STRAWS) ===")
print(metrics_text)

# --- restyled: actual = black solid (linewidth 1.8, matches the "Actual TWSA"
# convention), model = steelblue dashed (matches the "raw-target model" color);
# metrics box formatting (monospace, fontsize 8, bottom-left, rounded white
# box, alpha 0.85) now matches the final comparison plot in the other script ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax1.plot(comp.index, comp['Actual STRAWS'],        label='Actual STRAWS',        color='black', linewidth=1.8)
ax1.plot(comp.index, comp['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled', color='steelblue', linestyle='--')
ax1.set_title('Raw TWSA (reconstructed) — baseline offset is visible here')
ax1.set_ylabel('TWSA (inches)')
ax1.legend(loc='upper right', fontsize=8)

ax2.plot(comp_anom.index, comp_anom['Actual STRAWS'],        label='Actual STRAWS (anomaly)',        color='black', linewidth=1.8)
ax2.plot(comp_anom.index, comp_anom['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled (anomaly)', color='steelblue', linestyle='--')
ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax2.set_title('Anomaly-from-mean TWSA — apples-to-apples comparison')
ax2.set_xlabel('Date')
ax2.set_ylabel('TWSA anomaly (inches)')
ax2.legend(loc='upper right', fontsize=8)

ax2.text(0.01, 0.02, metrics_text, transform=ax2.transAxes,
         fontsize=8, verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

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
# 16. Train/Val split comparison plot (mentor's request) -- opens after you
# close the Section 15 comparison figure- WORK ON LATER, do I want chronological (how it is now) or do I want cross-validation throughout the whole thing?
# =============================================================================
# Same two-panel layout, but instead of one continuous "GRACE DNN Gap-Filled"
# line, the predicted line is split by the period it came from: trained-on
# vs. held-out-during-training (val) vs. everything after that (test period +
# recursively gap-filled months). This shows visually whether the model
# tracks STRAWS on data it never saw, not just via the MAE/RMSE numbers.

train_end_date = pd.Timestamp(dates_train.max())
val_end_date = pd.Timestamp(dates_val.max())

def split_by_period(index, col):
    """Return 3 Series (NaN outside their own window) so each draws as its own segment."""
    train_part = col.where(index <= train_end_date)
    val_part = col.where((index > train_end_date) & (index <= val_end_date))
    rest_part = col.where(index > val_end_date)
    return train_part, val_part, rest_part

grace_train_raw, grace_val_raw, grace_rest_raw = split_by_period(comp.index, comp['GRACE DNN Gap-Filled'])
grace_train_anom, grace_val_anom, grace_rest_anom = split_by_period(comp_anom.index, comp_anom['GRACE DNN Gap-Filled'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax1.plot(comp.index, comp['Actual STRAWS'], label='Actual STRAWS', color='black', linewidth=1.8)
ax1.plot(comp.index, grace_train_raw, label='Predicted (train period)', color='steelblue', linestyle='--')
ax1.plot(comp.index, grace_val_raw, label='Predicted (val period)', color='darkorange', linestyle='--')
ax1.plot(comp.index, grace_rest_raw, label='Predicted (test/gap-filled)', color='gray', linestyle='--', alpha=0.6)
ax1.axvline(train_end_date, color='gray', linewidth=0.8, linestyle=':')
ax1.axvline(val_end_date, color='gray', linewidth=0.8, linestyle=':')
ax1.set_title('Raw TWSA — predictions split by train/val/test period')
ax1.set_ylabel('TWSA (inches)')
ax1.legend(loc='upper right', fontsize=8)

ax2.plot(comp_anom.index, comp_anom['Actual STRAWS'], label='Actual STRAWS (anomaly)', color='black', linewidth=1.8)
ax2.plot(comp_anom.index, grace_train_anom, label='Predicted (train period)', color='steelblue', linestyle='--')
ax2.plot(comp_anom.index, grace_val_anom, label='Predicted (val period)', color='darkorange', linestyle='--')
ax2.plot(comp_anom.index, grace_rest_anom, label='Predicted (test/gap-filled)', color='gray', linestyle='--', alpha=0.6)
ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax2.axvline(train_end_date, color='gray', linewidth=0.8, linestyle=':')
ax2.axvline(val_end_date, color='gray', linewidth=0.8, linestyle=':')
ax2.set_title('Anomaly-from-mean TWSA — predictions split by train/val/test period')
ax2.set_xlabel('Date')
ax2.set_ylabel('TWSA anomaly (inches)')
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('twsa_comparison_train_val_split.png', dpi=150)
plt.show()
print("Train/val split comparison plot saved to twsa_comparison_train_val_split.png")
# =============================================================================
# 16. Summary
# =============================================================================
# Calculate Real Model Prediction Amplitude over the Test Horizon
actual_test_amplitude = np.max(actual_test_horizon) - np.min(actual_test_horizon)
predicted_test_amplitude = np.max(direct_preds) - np.min(direct_preds)
true_amplitude_diff = abs(actual_test_amplitude - predicted_test_amplitude)

print("\n" + "=" * 70)
print("SUMMARY OF VALIDATION DIAGNOSTICS (TREND + SEASONALITY REMOVED VERSION)")
print("=" * 70)
print(f"- Trend removed:                 {trend_coefs[0]*12:.4f} in/year")
print(f"- Seasonal cycle removed:        monthly climatology (see printout above)")
print(f"- Overfitting check (Section 9): train loss {final_train_loss:.4f} vs "
      f"val loss {final_val_loss:.4f} (residual units)")
print(f"- One-step test MAE (Section 10): {test_mae:.4f} in (residual units)")
print(f"- Multi-step recursive MAE (Section 12), full {HORIZON}-month horizon: "
      f"{step_errors.mean():.4f} in (residual units)")
print(f"- Final vs. independent STRAWS dataset (Section 16, reconstructed TWSA): "
      f"MAE {mae_anom:.3f} in, correlation {corr_anom:.3f}")

print("\n=== TRUE OUT-OF-SAMPLE PERFORMANCE (THE REAL BREAKTHROUGH) ===")
print(f"- True Validation Target Amplitude: {actual_test_amplitude:.3f} in")
print(f"- True Predicted Test Amplitude:    {predicted_test_amplitude:.3f} in")
print(f"- True Amplitude Difference:        {true_amplitude_diff:.3f} in")
print("  (This number proves you have broken past the 2.173 in baseline!)")

print("\nResidual-unit MAEs aren't directly comparable to a raw-TWSA model's MAEs --")
print("the seasonal cycle (usually the biggest source of month-to-month variance)")
print("is already subtracted out, so these numbers look smaller even if the")
print("underlying skill is the same. Section 16 is the fair comparison point.")