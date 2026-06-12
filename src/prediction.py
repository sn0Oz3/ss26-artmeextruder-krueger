import pickle
from pathlib import Path

import numpy as np

_MODEL_CACHE = None


def _load_prediction_artifacts():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    project_dir = Path(__file__).resolve().parent.parent
    models_dir = project_dir / "Models"

    model_path = models_dir / "mlp_model.pkl"
    scaler_path = models_dir / "scalers.pkl"

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    with scaler_path.open("rb") as scaler_file:
        scalers = pickle.load(scaler_file)

    _MODEL_CACHE = model, scalers
    return _MODEL_CACHE

# Funktion ist als Test des vorher trainierten Modells gedacht -> Parameter können ab Zeile 51 angepasst werden -> Sagt Durchmesser vorher
# This function predicts the diameter of a filament based on the input parameters und des zuvor trainierten Modells
def predict_diameter(temp_act, extruder_rpm, puller_rpm, sensor_pos):
    model, scalers = _load_prediction_artifacts()

    feature_scaler = scalers["feature_scaler"]
    diameter_scaler = scalers["diameter_scaler"]

    input_values = np.array([[temp_act, extruder_rpm, puller_rpm, sensor_pos]])

    input_scaled = feature_scaler.transform(input_values)
    try:
        prediction_scaled = model.predict(input_scaled, verbose=0)
    except TypeError:
        prediction_scaled = model.predict(input_scaled)
    prediction_mm = diameter_scaler.inverse_transform(prediction_scaled)

    return float(prediction_mm[0][0])

# Hier wird die Funktion mit Beispielwerten aufgerufen, um den vorhergesagten Durchmesser zu berechnen und auszugeben.
if __name__ == "__main__":
    diameter = predict_diameter(
        temp_act=187.9,
        extruder_rpm=12.0,
        puller_rpm=23.56,
        sensor_pos=6.62,
    )

    print(f"Vorhergesagter Durchmesser: {diameter:.3f} mm")

# Aus Datensatz:
# 1.8,184.0,10.0,14.34,6.36
# 1.81,184.0,10.0,16.06,6.41

# 1.62,187.7,12.0,24.32,6.7     9500
# 1.69,187.9,12.0,23.56,6.62    9600
