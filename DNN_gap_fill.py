import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the time-series data
df = pd.read_csv(r"C:\Users\grace\Downloads\GRACE Research Project\Texas-Statewide-English-1782253537240(in).csv")
df.columns = df.columns.str.strip()  # Clean whitespace

# Target variable (TWSA metric)
target_col = "Texas (in)"

# FIX (only change in this whole block): parse Date and sort chronologically,
# then snap to a strict monthly index so any REAL missing month becomes an
# explicit NaN row instead of just being absent entirely. Without this,
# every .shift(i) below assumes each row is exactly 1 month after the row
# before it -- not true if there's a real gap in the record (e.g. the
# 11-month GRACE -> GRACE-FO gap). If a month is missing, Lag_1 would
# silently mean "however many months back the next available row happens
# to be," not "1 month ago," and the model trains on mislabeled inputs
# with no warning.
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').drop_duplicates(subset='Date')
df = df.set_index('Date')[[target_col]].resample('MS').mean().reset_index()

# 2. Generate 6 distinct lag CSV files as requested
for i in range(1, 7):
    lag_df = df[['Date', target_col]].copy()
    lag_df[f'Lag_{i}'] = lag_df[target_col].shift(i)
    lag_df = lag_df.dropna()  # Remove truncation padding
    lag_df.to_csv(f"twsa_lag_{i}.csv", index=False)

print("[INFO] Generated 6 independent lag CSV files (twsa_lag_1.csv to twsa_lag_6.csv)")  # here we can add month number to make sure seasonality is considered

# 3. Consolidate lag inputs for network ingestion
dataset = df[['Date', target_col]].copy()
for i in range(1, 7):
    dataset[f'Lag_{i}'] = dataset[target_col].shift(i)

dataset = dataset.dropna()

# Extract matrices
X = dataset[[f'Lag_{i}' for i in range(1, 7)]].values
y = dataset[target_col].values

# 4. Train-Test Split (Chronological preservation for time series, 15%-25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Normalize inputs for stable neural network gradient descent ( so we will not get very large or vey small values as a result of the TWSA values being large or small. so we standardise with mean and standard deviation I would look in to the standard scaler equation it is simple equation

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build Vanilla Neural Network Architecture (MLP)

# I would ply with these numbers 32, and 16 there are the number of neurons and the 2 denses mean it has two layers

# 6 lagged inputs  go to 32 neurons then output from the 32 neurons goes to 16 neurons then output from the 16 neurons goes to 1 neuron ( out put) the process is basically iterating through weights that connect each neuron so that the final output is closer to the observed TWSA.
# too much depth and too many neurons will make the model heavy ( eating spagetti with a shovel), too small in adequate drinking soup with fork,
model = Sequential([
    Dense(32, activation='relu', input_shape=(6,)),  # 6 features representing Lag 1 to 6
    Dense(16, activation='relu'),
    Dense(1)                                          # 1 Linear output for regression prediction
])

# Compile with Mean Squared Error loss
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Model Training
print("\n[INFO] Starting Vanilla Neural Network Training...")
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=60,
    validation_split=0.1,
    batch_size=8,
    verbose=1
)

# 7. Model Evaluation
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
# smaller lass is what we are looking for it means what the neural network gives and observation are close which is good
print(f"\n[EVALUATION] Final Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f}")


# reshaped for a flat 6-value lag vector instead of a 3D RNN sequence input.
# not from mentor script
series = df.set_index('Date')[target_col]  # same calendar-complete series from the FIX above, NaN = real gap


def find_missing_value_gaps(series):
    """List every stretch of consecutive NaN (truly missing) months."""
    is_missing = series.isna()
    gaps = []
    current_gap_start = None

    for date, missing in is_missing.items():
        if missing and current_gap_start is None:
            current_gap_start = date
        elif not missing and current_gap_start is not None:
            gap_dates = series.index[(series.index >= current_gap_start) & (series.index < date)]
            gaps.append({
                "start_date": current_gap_start,
                "end_date": gap_dates[-1],
                "num_missing_months": len(gap_dates),
            })
            current_gap_start = None

    if current_gap_start is not None:
        gap_dates = series.index[series.index >= current_gap_start]
        gaps.append({
            "start_date": current_gap_start,
            "end_date": gap_dates[-1],
            "num_missing_months": len(gap_dates),
        })
    return gaps


def fill_gap_recursive_mlp(model, scaler, history_values, num_missing_months, n_lags=6):
    """
    Walk forward through one gap, one month at a time. Builds [Lag_1..Lag_6]
    = [t-1..t-6] from the most recent known-or-predicted values, scales it
    the same way training data was scaled, predicts, then feeds that
    prediction back in as if it were real for the next step.
    `history_values` = chronological (oldest -> newest) real values right
    before the gap, length >= n_lags.
    """
    history = list(history_values)
    predictions = []

    for _ in range(num_missing_months):
        last_n = history[-n_lags:]      # oldest -> newest
        feature_vec = last_n[::-1]      # newest -> oldest = [t-1, ..., t-n], matches Lag_1..Lag_6 order

        x = np.array(feature_vec, dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        y_pred = model.predict(x_scaled, verbose=0)[0, 0]

        predictions.append(y_pred)
        history.append(y_pred)

    return predictions


def fill_all_gaps(model, scaler, series, n_lags=6):
    """Find every gap and fill each one. Returns {gap_start_date: pd.Series of guesses}."""
    gaps = find_missing_value_gaps(series)
    filled_segments = {}

    for gap in gaps:
        print(f"Filling gap: {gap['start_date'].date()} to {gap['end_date'].date()} "
              f"({gap['num_missing_months']} missing months)")

        before_gap = series[series.index < gap["start_date"]].dropna()
        if len(before_gap) < n_lags:
            print(f"  Skipping -- only {len(before_gap)} real months before this gap "
                  f"(need {n_lags}). Gap is too close to the start of the record.")
            continue

        last_known = before_gap.values[-n_lags:]
        filled_values = fill_gap_recursive_mlp(model, scaler, last_known, gap["num_missing_months"], n_lags)

        gap_dates = pd.date_range(start=gap["start_date"], periods=gap["num_missing_months"], freq="MS")
        filled_segments[gap["start_date"]] = pd.Series(filled_values, index=gap_dates)

    return filled_segments


def build_complete_series(series, filled_segments):
    """Patch each filled segment into its matching blank spot in the original series."""
    complete = series.copy()
    for filled_values in filled_segments.values():
        complete.update(filled_values)
    return complete


filled_segments = fill_all_gaps(model, scaler, series, n_lags=6)
complete_series = build_complete_series(series, filled_segments)
complete_series.to_csv("twsa_gapfilled_dnn.csv", header=[target_col])
print("Done. Complete, gap-free monthly series saved to twsa_gapfilled_dnn.csv")