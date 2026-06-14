import pickle
import re
from pathlib import Path

import numpy as np

_MODEL_CACHE = None
_TEST_MAE_CACHE = None
_TEST_MAE_LOADED = False
DEFAULT_FEATURE_COLUMNS = ['Temp Act (C)', 'Extruder RPM', 'Puller RPM', 'Sensor Pos']
DEFAULT_SEQUENCE_LENGTH = 60


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
        missing_columns = [column for column in feature_columns if column not in last_rows.columns]
        if missing_columns:
            raise ValueError(f"Folgende Feature-Spalten fehlen: {', '.join(missing_columns)}")
        input_values = last_rows[feature_columns].to_numpy()
    else:
        input_values = np.asarray(last_rows, dtype=float)

    expected_shape = (sequence_length, len(feature_columns))
    if input_values.shape != expected_shape:
        raise ValueError(
            f"Die Vorhersage erwartet genau {expected_shape[0]} Zeilen "
            f"mit jeweils {expected_shape[1]} Features. Erhalten: {input_values.shape}"
        )

    return input_values


# This function predicts the next filament diameter from the previous rows.
def predict_diameter(last_rows):
    model, scalers = _load_prediction_artifacts()

    feature_scaler = scalers["feature_scaler"]
    diameter_scaler = scalers["diameter_scaler"]
    feature_columns = scalers.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    sequence_length = scalers.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)

    input_values = _prepare_sequence_input(last_rows, feature_columns, sequence_length)
    input_scaled = feature_scaler.transform(input_values)
    input_scaled = input_scaled.flatten().reshape(1, -1)

    try:
        prediction_scaled = model.predict(input_scaled, verbose=0)
    except TypeError:
        prediction_scaled = model.predict(input_scaled)
    prediction_mm = diameter_scaler.inverse_transform(prediction_scaled)

    return float(prediction_mm[0][0])


# Hier wird die Funktion mit Beispielwerten aufgerufen, um den vorhergesagten Durchmesser zu berechnen und auszugeben.
if __name__ == "__main__":
    example_row = [187.9, 12.0, 23.56, 6.62]
    example_rows = np.tile(example_row, (DEFAULT_SEQUENCE_LENGTH, 1))
    diameter = predict_diameter(example_rows)
    test_mae_mm = get_test_mae_mm()

    print(f"Vorhergesagter Durchmesser: {diameter:.3f} mm")
    if test_mae_mm is not None:
        print(f"Rueckskalierter Test MAE: +/- {test_mae_mm:.3f} mm")

# Aus Datensatz:
# 1.8,184.0,10.0,14.34,6.36
# 1.81,184.0,10.0,16.06,6.41

# 1.62,187.7,12.0,24.32,6.7     9500
# 1.69,187.9,12.0,23.56,6.62    9600
