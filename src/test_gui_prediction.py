import argparse
from pathlib import Path
import tkinter as tk

import pandas as pd

from extruder_gui import BAD, DIA_TARGET, DIA_TOL, GOOD, INK, MonitorApp
from prediction import DEFAULT_FEATURE_COLUMNS, DEFAULT_SEQUENCE_LENGTH, predict_diameter

# ----------------------------------------------------------------------------------------------------------------------
# Test der Gui: CSV-Datei wird zeilenweise abgespielt; ab Zeile 41 wird auch aus den vorherigen 40 Zeilen der naechste Durchmesser vorhergesagt.
# ----------------------------------------------------------------------------------------------------------------------
REPLAY_INTERVAL_MS = 500
DEFAULT_TARGET_TEMP = 190.0
DEFAULT_FAN_PCT = 45
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "recording_2026-06-10_13-53-40_preprocessed.csv"
)


class CsvDiameterReader:
    def __init__(self):
        self.connected = True
        self.diameter = None

    def get(self):
        return self.diameter

    def set(self, diameter):
        if pd.isna(diameter):
            self.diameter = None
        else:
            self.diameter = float(diameter)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a preprocessed CSV in the GUI and predict the next diameter."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        type=Path,
        help="Path to a preprocessed CSV file.",
    )
    return parser.parse_args()


def _validate_csv(csv_df, csv_path):
    required_columns = ["Diameter (mm)", *DEFAULT_FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in csv_df.columns]
    if missing_columns:
        raise ValueError(
            f"{csv_path} enthaelt nicht alle benoetigten Spalten: "
            f"{', '.join(missing_columns)}"
        )

    if len(csv_df) <= DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(
            f"{csv_path} braucht mindestens {DEFAULT_SEQUENCE_LENGTH + 1} Zeilen, "
            "damit aus 40 vorherigen Zeilen der naechste Durchmesser vorhergesagt werden kann."
        )


def _row_to_gui_data(row, reading_no, prediction):
    return {
        "reading_no": reading_no,
        "elapsed": 0,
        "actual_temp": row["Temp Act (C)"],
        "target_temp": DEFAULT_TARGET_TEMP,
        "extruder_rpm": row["Extruder RPM"],
        "puller_speed": row["Puller RPM"],
        "fil_width": row["Sensor Pos"],
        "fil_width_predicted": prediction,
        "fan_pct": DEFAULT_FAN_PCT,
    }


def _update_actual_diameter(app, diameter):
    if pd.isna(diameter):
        app._metric_labels["fil_width"].config(text="--", fg=INK)
        return

    app._metric_labels["fil_width"].config(
        text=f"{diameter:.3f}",
        fg=GOOD if abs(diameter - DIA_TARGET) <= DIA_TOL else BAD,
    )


# Skript zum Testen der GUI -> Ich muss nicht immer zum Filamentextruder laufen, um die Vorhersage zu testen.
# Die CSV wird zeilenweise abgespielt; ab Zeile 41 wird aus den vorherigen 40 Zeilen der naechste Durchmesser vorhergesagt.
def main():
    args = _parse_args()
    csv_path = args.csv_path.resolve()
    csv_df = pd.read_csv(csv_path)
    _validate_csv(csv_df, csv_path)

    root = tk.Tk()
    app = MonitorApp(root)
    root.title(f"CSV Prediction Test - {csv_path.name}")
    csv_diameter_reader = CsvDiameterReader()
    app.messuhr_obj = csv_diameter_reader
    app.messuhr_online = True

    replay_state = {
        "row_index": 0,
        "prediction_error_reported": False,
    }

    def replay_next_row():
        row_index = replay_state["row_index"]
        row = csv_df.iloc[row_index]
        prediction = None

        if row_index >= DEFAULT_SEQUENCE_LENGTH:
            previous_rows = csv_df.iloc[
                row_index - DEFAULT_SEQUENCE_LENGTH:row_index
            ][DEFAULT_FEATURE_COLUMNS]
            try:
                prediction = predict_diameter(previous_rows)
                replay_state["prediction_error_reported"] = False
            except Exception as exc:
                if not replay_state["prediction_error_reported"]:
                    print(f"Prediction failed: {exc}")
                    replay_state["prediction_error_reported"] = True

        data = _row_to_gui_data(row, row_index + 1, prediction)
        csv_diameter_reader.set(row["Diameter (mm)"])
        app._update_metrics(data)
        _update_actual_diameter(app, row["Diameter (mm)"])

        replay_state["row_index"] += 1
        if replay_state["row_index"] < len(csv_df):
            root.after(REPLAY_INTERVAL_MS, replay_next_row)

    root.after(0, replay_next_row)
    root.mainloop()


if __name__ == "__main__":
    main()
