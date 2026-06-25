"""
rnn_gapfill.py
===============
NOTE: Grace here, there are a lot of comments on here to help explain to me what is going on, courtesy of Claude.

A simple RNN that fills in missing months of GRACE TWSA (terrestrial
water storage anomaly) data.

The idea: teach a model to guess "what's the water storage level THIS
month?" just by looking at the last few months -- nothing else, no
rainfall data, no soil moisture, just GRACE looking at its own past to
guess its own future. Once it's decent at that, point it at the actual
blank months in the real record and let it fill them in.

What's in here, in order:
    1. load_twsa_series()              -> open the GRACE file, average over our region of interest,
                                           and line everything up month by month (blanks stay blank)
    2. create_lag_windows()             -> chop the series into "last 3 months -> next month" examples
    3. scale_series()                   -> squish the numbers into a small, easy-to-learn range
    4. reshape_for_rnn()                -> repackage the data into the shape Keras expects
    5. build_rnn_model()                -> the actual network: 2 layers total --
                                           a "remember-y" layer with 16 little helper neurons that
                                           reads the past few months, then 1 final neuron that turns
                                           all those opinions into a single guess
    6. train_model()                    -> let the model practice, stop once it stops improving
    7. evaluate_on_holdout()            -> hide some real answers, see how close the guesses land
    8. find_missing_value_gaps()        -> find every blank stretch in the real data
       get_last_known_window_before()   -> grab the real months sitting right before each blank stretch
       fill_gap_recursive()             -> walk forward through one blank stretch, guessing month by month
       fill_all_gaps()                  -> do that for every blank stretch in the whole record
       build_complete_series()          -> stitch the real months and the guessed months into one record
    9. main()                           -> runs all of the above, start to finish

Install: pip install tensorflow numpy pandas scikit-learn matplotlib netCDF4 --break-system-packages
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import netCDF4 as nc

# LOAD
def load_twsa_series(nc_path,
                     lat_min=3.0, lat_max=18.0,
                     lon_min=32.0, lon_max=45.0):
    """
    Open a GRACE/GRACE-FO file and boil it down to one number/ month:
    the average water storage anomaly over our region of interest
    (defaults to a box around the Eastern Nile / GERD area -- change the
    lat/lon numbers above if you need a different region).

    Comes back as a simple monthly list of numbers (a pandas Series),
    with a blank (NaN) wherever a month is missing from the satellite
    record -- that's exactly what the rest of this script expects.
    """
    ds = nc.Dataset(nc_path)

    lwe = ds.variables['lwe_thickness'][:]   # (time, lat, lon), in cm of water
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]

    # --- figure out which date each "slice" of the file belongs to ---
    time_var = ds.variables['time']
    time_units = getattr(time_var, 'units', None) or getattr(time_var, 'Units')
    time_cal   = getattr(time_var, 'calendar', None) or getattr(time_var, 'Calendar', 'standard')
    dates = nc.num2date(time_var[:], units=time_units, calendar=time_cal)
    dates = pd.to_datetime([d.isoformat() for d in dates])

    # --- keep only the grid squares inside our region's lat/lon box ---
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lwe_box = lwe[:, lat_mask, :][:, :, lon_mask]   # just our region, every month

    # average all the grid squares in the box into one number per month
    # (ignoring any ocean/no-data pixels GRACE marks as "fill values")
    basin_avg = np.array([
        np.nanmean(np.ma.filled(month, np.nan))
        for month in lwe_box
    ])

    series = pd.Series(basin_avg, index=dates)
    series = series.resample("MS").mean()   # line up to the 1st of each month; missing months -> blank
    return series

# WINDOWING
def create_lag_windows(series, window_size=3):
    """
    Slide a 3-month window across the data to build practice examples:
    "here's months 1, 2, and 3 -- what's month 4?"

    If a window (or the answer right after it) touches even one blank
    month, we skip it entirely -- we only want to practice on stretches
    where every number is real.
    """
    values = series.values
    dates = series.index

    X, y, window_start_dates = [], [], []
    for i in range(len(values) - window_size):
        window = values[i : i + window_size]
        target = values[i + window_size]

        if np.isnan(window).any() or np.isnan(target):
            continue

        X.append(window)
        y.append(target)
        window_start_dates.append(dates[i])

    return np.array(X), np.array(y), window_start_dates


#SCALE- shrink numbers for network to work faster
def scale_series(X, y):
    """
    Neural networks learn much faster on small, tidy numbers than on raw
    centimeters of water. This squishes everything into a small range
    (roughly 0 to 1) so the math behind the scenes behaves nicely.

    NOTE (simplification to revisit later): this measures that "small
    range" using the FULL dataset, including the part we'll later hold
    back for testing. That's a shortcut that's fine for a first pass,
    but worth tightening up later so the test data stays truly unseen.

    Hands back the squished X and y, plus the "scaler" tool itself --
    you'll need that same tool later to convert predictions back into
    real water-storage units.
    """
    scaler = MinMaxScaler()
    combined = np.concatenate([X.flatten(), y.flatten()]).reshape(-1, 1)
    scaler.fit(combined)

    X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, scaler


def inverse_scale(values, scaler):
    """Undo the squishing -- turn small scaled numbers back into real TWSA (cm)."""
    values = np.array(values).reshape(-1, 1)
    return scaler.inverse_transform(values).flatten()


# RESHAPE FOR KERAS
def reshape_for_rnn(X):
    """
    Keras' RNN layers always expect 3 numbers describing the shape of the
    data: (how many examples, how many months per example, how many
    numbers per month). We only track one number per month (the TWSA
    value itself), so we just tack a "1" onto the end of the shape.
    """
    return X.reshape(X.shape[0], X.shape[1], 1)


# BUILD RNN
def build_rnn_model(window_size, hidden_units=16):
    """
    The network has 2 layers:
      - a SimpleRNN layer with 16 little helper neurons. Each helper reads
        the past few months one at a time (like reading a tiny diary,
        page by page) and keeps a running note of what it's seen so far.
        By the end, each of the 16 helpers has its own opinion about what
        comes next.
      - a Dense layer with just 1 neuron, whose only job is to listen to
        all 16 opinions and squish them into one final number: the guess.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        keras.layers.SimpleRNN(hidden_units, activation="tanh"),
        keras.layers.Dense(1),  # one number out: the predicted TWSA for next month
    ])

    model.compile(
        optimizer="adam",
        loss="mse",       # how training measures "how wrong was the guess"
        metrics=["mae"],  # an easier-to-read version of that same idea, just for watching progress
    )
    return model


# TRAIN
def train_model(model, X_train, y_train, X_val, y_val, max_epochs=200):
    """
    Run practice rounds (epochs). Since we don't have a huge amount of
    data, we stop early the moment the model stops getting better on
    data it hasn't trained on directly -- that helps avoid it just
    memorizing the training set instead of actually learning the pattern.
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


#EVAL--- see how close guesses were
def evaluate_on_holdout(model, X_holdout, y_holdout, scaler):
    """
    Show the model windows it never trained on, compare its guesses to
    the real (but withheld) answers, and report how far off it tends to
    be, in real units (RMSE = a kind of average "miss distance").
    """
    y_pred_scaled = model.predict(X_holdout, verbose=0).flatten()

    y_pred = inverse_scale(y_pred_scaled, scaler)
    y_true = inverse_scale(y_holdout, scaler)

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"Holdout RMSE: {rmse:.3f} (same units as your TWSA column)")

    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="actual")
    plt.plot(y_pred, label="predicted")
    plt.legend()
    plt.title("RNN: holdout predictions vs actual")
    plt.savefig("rnn_holdout_check.png", dpi=150)

    return rmse


# FILL REAL GAPS---------------
def find_missing_value_gaps(series):
    """
    Walk through the series like flipping through a calendar, and circle
    every stretch of blank months -- a tiny circle around a single blank
    month, a big circle around eleven blank months in a row (looking at
    you, GRACE -> GRACE-FO gap).

    Comes back as a list, one entry per blank stretch found, e.g.:
        {"start_date": ..., "end_date": ..., "num_missing_months": ...}
    """
    is_missing = series.isna()
    gaps = []
    current_gap_start = None

    for date, missing in is_missing.items():
        if missing and current_gap_start is None:
            # just stepped into a blank stretch
            current_gap_start = date
        elif not missing and current_gap_start is not None:
            # just stepped back out of a blank stretch -- close it off
            gap_dates = series.index[
                (series.index >= current_gap_start) & (series.index < date)
            ]
            gaps.append({
                "start_date": current_gap_start,
                "end_date": gap_dates[-1],
                "num_missing_months": len(gap_dates),
            })
            current_gap_start = None

    # if a blank stretch runs all the way to the end of the record, close that one off too
    if current_gap_start is not None:
        gap_dates = series.index[series.index >= current_gap_start]
        gaps.append({
            "start_date": current_gap_start,
            "end_date": gap_dates[-1],
            "num_missing_months": len(gap_dates),
        })

    return gaps


def get_last_known_window_before(series, gap_start_date, window_size, scaler):
    """
    Grab the last `window_size` REAL months sitting right before a blank
    stretch starts, and squish them down with the same scaler the model
    was trained on -- the model only understands those squished numbers,
    not raw centimeters.
    """
    before_gap = series[series.index < gap_start_date].dropna()
    last_values = before_gap.values[-window_size:]

    if len(last_values) < window_size:
        raise ValueError(
            "Not enough real data right before this gap to build a full "
            f"window of size {window_size}. Try a smaller window_size."
        )

    last_values_scaled = scaler.transform(last_values.reshape(-1, 1)).flatten()
    return last_values_scaled


def fill_gap_recursive(model, last_known_window, scaler, num_missing_months):
    """
    Walk forward through one blank stretch, one month at a time. Each
    guess gets added to the window and treated as if it were real data
    for the NEXT guess -- there's nothing real left inside the gap to
    use, so the model has to lean on its own previous guesses.
    """
    window = list(last_known_window)
    predictions_scaled = []

    for _ in range(num_missing_months):
        model_input = np.array(window[-len(last_known_window):]).reshape(1, -1, 1)
        next_pred_scaled = model.predict(model_input, verbose=0)[0, 0]

        predictions_scaled.append(next_pred_scaled)
        window.append(next_pred_scaled)  # pretend this guess was real, for the next step

    return inverse_scale(predictions_scaled, scaler)


def fill_all_gaps(model, series, scaler, window_size):
    """
    Find every blank stretch (find_missing_value_gaps) and fill each one
    (fill_gap_recursive). Comes back as {gap_start_date: pandas Series of
    guessed values}, so you can look at each filled stretch on its own.
    """
    gaps = find_missing_value_gaps(series)
    filled_segments = {}

    for gap in gaps:
        print(
            f"Filling gap: {gap['start_date'].date()} to {gap['end_date'].date()} "
            f"({gap['num_missing_months']} missing months)"
        )

        # make sure there's enough real data right before this gap to even start
        before_gap = series[series.index < gap["start_date"]].dropna()
        if len(before_gap) < window_size:
            print(
                f"  Skipping -- only {len(before_gap)} real months before this gap "
                f"(need {window_size}). Gap is too close to the start of the record."
            )
            continue

        last_window_scaled = get_last_known_window_before(
            series, gap["start_date"], window_size, scaler
        )
        filled_values = fill_gap_recursive(
            model, last_window_scaled, scaler, gap["num_missing_months"]
        )

        gap_dates = pd.date_range(
            start=gap["start_date"], periods=gap["num_missing_months"], freq="MS"
        )
        filled_segments[gap["start_date"]] = pd.Series(filled_values, index=gap_dates)

    return filled_segments


def build_complete_series(series, filled_segments):
    """
    Take the original series (the one with holes in it) and drop each
    filled stretch into its matching blank spot, so you end up with one
    continuous, hole-free monthly record.
    """
    complete = series.copy()
    for gap_start, filled_values in filled_segments.items():
        complete.update(filled_values)  # only fills in the blank slots, leaves real data untouched
    return complete


# MAIN-----------------
def main():
    WINDOW_SIZE = 3  # how many past months the model looks at before guessing the next one

    # load the GRACE data, point at wherever you saved your basin-averaged GRACE
    # export -- the .nc file with monthly water-storage values.
    DATA_PATH = r"C:\Users\grace\Downloads\GRACE Research Project\3avg_files\CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
    series = load_twsa_series(DATA_PATH)

    #chop it into "past 3 months -> next month" examples 
    X, y, window_dates = create_lag_windows(series, window_size=WINDOW_SIZE)

    #  shrink the numbers down for easier learning 
    X_scaled, y_scaled, scaler = scale_series(X, y)

    #repackage into the shape Keras wants 
    X_reshaped = reshape_for_rnn(X_scaled)

    # --- carve out a chunk of real data to test on later ---
    split = int(0.85 * len(X_reshaped))
    X_train, X_holdout = X_reshaped[:split], X_reshaped[split:]
    y_train, y_holdout = y_scaled[:split], y_scaled[split:]

    val_split = int(0.85 * len(X_train))
    X_train, X_val = X_train[:val_split], X_train[val_split:]
    y_train, y_val = y_train[:val_split], y_train[val_split:]

    # - build the model, then let it practice ---
    model = build_rnn_model(window_size=WINDOW_SIZE)
    train_model(model, X_train, y_train, X_val, y_val)

    # --see how good its guesses are on data it's never seen ---
    evaluate_on_holdout(model, X_holdout, y_holdout, scaler)

    # ---  find every real blank stretch and fill it in ---
    filled_segments = fill_all_gaps(model, series, scaler, window_size=WINDOW_SIZE)
    # glue the real data + the filled-in guesses into one record ---
    complete_series = build_complete_series(series, filled_segments)
    complete_series.to_csv("twsa_gapfilled_rnn.csv", header=["twsa_cm"])

    print("Done. Complete, gap-free monthly series saved to twsa_gapfilled_rnn.csv")


if __name__ == "__main__":
    main()