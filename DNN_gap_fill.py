import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import netCDF4 as nc

# =============================================================================
# 1. LOAD DATA FROM GRACE NetCDF FILE
# =============================================================================
DATA_PATH = r"C:\Users\grace\Downloads\GRACE Research Project\3avg_files\CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"

dataset_nc = nc.Dataset(DATA_PATH)

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
dataset_nc.close()

df = pd.DataFrame({'Date': dates, 'Texas (in)': twsa_texas * 0.3937})  # cm to inches
target_col = "Texas (in)"

# Snap to strict monthly index so real gaps become explicit NaN rows
# (without this, .shift() would silently mislabel lags across gaps)
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
dataset = df[['Date', target_col]].copy()
for i in range(1, 7):
    dataset[f'Lag_{i}'] = dataset[target_col].shift(i)
dataset = dataset.dropna()

X = dataset[[f'Lag_{i}' for i in range(1, 7)]].values
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
# 6 lag inputs → 32 neurons → 16 neurons → 1 output
# too many neurons = eating spaghetti with a shovel
# too few = drinking soup with a fork
# =============================================================================
model = Sequential([
    Dense(32, activation='relu', input_shape=(6,)),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# =============================================================================
# 6. TRAIN
# =============================================================================
print("\n[INFO] Starting Vanilla Neural Network Training...")
history = model.fit(X_train_scaled, y_train, epochs=60, validation_split=0.1, batch_size=8, verbose=1)

# =============================================================================
# 7. EVALUATE
# smaller loss = model predictions are closer to real observations
# =============================================================================
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n[EVALUATION] Final Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

# =============================================================================
# 8. GAP FILLING
# For each missing stretch: guess month 1 using real data, then use that
# guess to predict month 2, and so on (snowball effect -- shakier the longer
# the gap, since each guess builds on the previous guess)
# =============================================================================
series = df.set_index('Date')[target_col]


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


def fill_gap_recursive_mlp(model, scaler, history_values, num_missing_months, n_lags=6):
    """Predict one month at a time, feeding each guess back in as if it were real."""
    history = list(history_values)
    predictions = []
    for _ in range(num_missing_months):
        feature_vec = history[-n_lags:][::-1]  # [t-1, t-2, ..., t-6]
        x_scaled = scaler.transform(np.array(feature_vec, dtype=float).reshape(1, -1))
        y_pred = model.predict(x_scaled, verbose=0)[0, 0]
        predictions.append(y_pred)
        history.append(y_pred)
    return predictions


def fill_all_gaps(model, scaler, series, n_lags=6):
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


complete_series = series.copy()
for filled_values in fill_all_gaps(model, scaler, series).values():
    complete_series.update(filled_values)

complete_series.to_csv("twsa_gapfilled_dnn.csv", header=[target_col])
print("Done. Complete gap-free monthly series saved to twsa_gapfilled_dnn.csv")

# =============================================================================
# 9. COMPARE AGAINST ACTUAL STRAWS DATA
# =============================================================================
actual = pd.read_csv(r"C:\Users\grace\Downloads\GRACE Research Project\Texas-Statewide-English-1782253537240(in).csv")
actual.columns = actual.columns.str.strip()
actual['Date'] = pd.to_datetime(actual['Date']).dt.to_period('M').dt.to_timestamp()  # snap to month-start
actual = actual[['Date', 'Texas (in)']].rename(columns={'Texas (in)': 'Actual STRAWS'}).set_index('Date')

grace = complete_series.copy()
grace.index = grace.index.to_period('M').to_timestamp()  # make sure GRACE is also month-start

comparison = pd.DataFrame({
    'Actual STRAWS': actual['Actual STRAWS'],
    'GRACE DNN Gap-Filled': grace
})

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(comparison.index, comparison['Actual STRAWS'], label='Actual STRAWS', color='steelblue')
plt.plot(comparison.index, comparison['GRACE DNN Gap-Filled'], label='GRACE DNN Gap-Filled', color='darkorange', linestyle='--')
plt.title('GRACE DNN Gap-Filled vs. Actual STRAWS Texas TWSA')
plt.xlabel('Date')
plt.ylabel('TWSA (inches)')
plt.legend()
plt.tight_layout()
plt.savefig('twsa_comparison.png', dpi=150)
plt.show()
print("Comparison plot saved to twsa_comparison.png")