import pickle
import re
from pathlib import Path

import numpy as np

from feature_engineering import (
    DIAMETER_COLUMN,
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    add_diameter_trend_features,
)

_MODEL_CACHE = None
_TEST_MAE_CACHE = None
_TEST_MAE_LOADED = False
DEFAULT_RAW_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS
DEFAULT_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_PREDICTION_HORIZON = 10


def _load_prediction_artifacts():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    project_dir = Path(__file__).resolve().parent.parent
    models_dir = project_dir / "models"

    model_path = models_dir / "mlp_model.pkl"
    scaler_path = models_dir / "scalers.pkl"

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    with scaler_path.open("rb") as scaler_file:
        scalers = pickle.load(scaler_file)

    _MODEL_CACHE = model, scalers
    return _MODEL_CACHE


def _read_test_mae_mm_from_report():
    project_dir = Path(__file__).resolve().parent.parent
    report_path = project_dir / "models" / "hyperparameters.txt"
    if not report_path.exists():
        return None

    report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Test MAE:\s*([0-9]+(?:\.[0-9]+)?)\s*mm", report_text)
    if match is None:
        return None

    return float(match.group(1))


def get_test_mae_mm():
    global _TEST_MAE_CACHE, _TEST_MAE_LOADED
    if _TEST_MAE_LOADED:
        return _TEST_MAE_CACHE

    _model, scalers = _load_prediction_artifacts()
    test_mae_mm = scalers.get("test_mae_mm")
    if test_mae_mm is not None:
        _TEST_MAE_CACHE = float(test_mae_mm)
    else:
        _TEST_MAE_CACHE = _read_test_mae_mm_from_report()

    _TEST_MAE_LOADED = True
    return _TEST_MAE_CACHE


def _prepare_sequence_input(last_rows, feature_columns, sequence_length):
    if hasattr(last_rows, "columns"):
        prepared_rows = last_rows
        if DIAMETER_COLUMN in prepared_rows.columns:
            prepared_rows = add_diameter_trend_features(prepared_rows, DIAMETER_COLUMN)

        missing_columns = [column for column in feature_columns if column not in prepared_rows.columns]
        if missing_columns:
            raise ValueError(f"Folgende Feature-Spalten fehlen: {', '.join(missing_columns)}")
        input_values = prepared_rows[feature_columns].to_numpy()
    else:
        input_values = np.asarray(last_rows, dtype=float)
        expected_raw_shape = (sequence_length, len(DEFAULT_RAW_FEATURE_COLUMNS))
        expected_model_shape = (sequence_length, len(feature_columns))
        if input_values.shape == expected_raw_shape and expected_model_shape != expected_raw_shape:
            input_values = _add_diameter_trend_features_to_array(input_values)

    expected_shape = (sequence_length, len(feature_columns))
    if input_values.shape != expected_shape:
        raise ValueError(
            f"Die Vorhersage erwartet genau {expected_shape[0]} Zeilen "
            f"mit jeweils {expected_shape[1]} Features. Erhalten: {input_values.shape}"
        )

    return input_values


def _add_diameter_trend_features_to_array(raw_values):
    # Array-Variante fuer die Live-GUI: Aus den 5 Rohwerten pro Zeile werden
    # dieselben 7 Modellfeatures gebaut wie beim Training.
    diameter = raw_values[:, 0].astype(float)
    delta_5 = np.zeros_like(diameter)
    delta_5[5:] = diameter[5:] - diameter[:-5]
    slope_10 = np.zeros_like(diameter)
    slope_10[10:] = (diameter[10:] - diameter[:-10]) / 10.0

    return np.column_stack(
        [
            diameter,
            delta_5,
            slope_10,
            raw_values[:, 1],
            raw_values[:, 2],
            raw_values[:, 3],
            raw_values[:, 4],
        ]
    )


# Diese Funktion sagt den Durchmesser mehrere Zeilen in die Zukunft voraus.
# Der genaue Abstand wird im Training gespeichert, standardmaessig sind es 10 Zeilen.
def predict_diameter(last_rows):
    model, scalers = _load_prediction_artifacts()

    feature_scaler = scalers["feature_scaler"]
    diameter_scaler = scalers["diameter_scaler"]
    feature_columns = scalers.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    target_column = scalers.get("target_column", DIAMETER_COLUMN)
    target_mode = scalers.get("target_mode", "diameter")
    sequence_length = scalers.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
    prediction_horizon = scalers.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON)
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon muss mindestens 1 sein.")
    if target_mode not in {"diameter", "delta"}:
        raise ValueError("target_mode muss 'diameter' oder 'delta' sein.")

    # last_rows muss die letzten sequence_length Messzeilen enthalten.
    # Jede Zeile enthaelt den gemessenen Durchmesser plus die Prozess-Features.
    input_values = _prepare_sequence_input(last_rows, feature_columns, sequence_length)
    input_scaled = feature_scaler.transform(input_values)
    input_scaled = input_scaled.flatten().reshape(1, -1)

    try:
        prediction_scaled = model.predict(input_scaled, verbose=0)
    except TypeError:
        prediction_scaled = model.predict(input_scaled)

    if target_mode == "delta":
        # Das Modell sagt die Aenderung im skalierten Durchmesserraum voraus.
        # Fuer den absoluten Durchmesser addieren wir diese Aenderung zum letzten
        # gemessenen Durchmesser aus der Eingabesequenz.
        diameter_column_index = feature_columns.index(target_column)
        last_diameter_mm = float(input_values[-1][diameter_column_index])
        prediction_delta_mm = float(prediction_scaled[0][0]) * float(diameter_scaler.scale_[0])
        return last_diameter_mm + prediction_delta_mm

    prediction_mm = diameter_scaler.inverse_transform(prediction_scaled)

    return float(prediction_mm[0][0])


# Hier wird die Funktion mit Beispielwerten aufgerufen, um den vorhergesagten Durchmesser zu berechnen und auszugeben.
if __name__ == "__main__":
    example_row = [1.69, 187.9, 12.0, 23.56, 6.62]
    example_rows = np.tile(example_row, (DEFAULT_SEQUENCE_LENGTH, 1))
    diameter = predict_diameter(example_rows)
    test_mae_mm = get_test_mae_mm()

    print(f"Vorhergesagter Durchmesser in {DEFAULT_PREDICTION_HORIZON} Zeilen: {diameter:.3f} mm")
    if test_mae_mm is not None:
        print(f"Rueckskalierter Test MAE: +/- {test_mae_mm:.3f} mm")

# Aus Datensatz:
# 1.8,184.0,10.0,14.34,6.36
# 1.81,184.0,10.0,16.06,6.41

# 1.62,187.7,12.0,24.32,6.7     9500
# 1.69,187.9,12.0,23.56,6.62    9600
