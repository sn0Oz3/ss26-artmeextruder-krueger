import argparse
from io import StringIO
from pathlib import Path
import tkinter as tk

import pandas as pd

from extruder_gui import BAD, DIA_TARGET, DIA_TOL, GOOD, INK, MonitorApp


# Wie schnell die CSV-Zeilen nacheinander in der GUI angezeigt werden.
REPLAY_INTERVAL_MS = 500

# Diese Fallback-Werte werden genutzt, falls eine CSV die Spalten nicht enthaelt.
DEFAULT_TARGET_TEMP = 190.0
DEFAULT_FAN_PCT = 45

# Standarddatei, falls beim Starten kein anderer CSV-Pfad angegeben wird.
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "recording_2026-06-10_13-53-40.csv"
)

REQUIRED_COLUMNS = [
    "Diameter (mm)",
    "Temp Act (C)",
    "Extruder RPM",
    "Puller RPM",
    "Sensor Pos",
]


class CsvDiameterReader:
    # Ersatz fuer die echte Messuhr. Der aktuelle Durchmesser kommt aus der CSV.
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
    # Beispiel:
    # python src/test_gui_prediction.py data/meine_aufnahme.csv
    parser = argparse.ArgumentParser(
        description="Spielt eine Aufnahme in der GUI ab und berechnet Vorhersagen."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        type=Path,
        help="Pfad zu einer Recording-CSV-Datei.",
    )
    return parser.parse_args()


def _load_recording_csv(csv_path):
    # Recording-Dateien koennen vor der eigentlichen CSV-Kopfzeile Metadaten enthalten.
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("Timestamp,") or line.startswith("Diameter (mm),")
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"{csv_path} enthaelt keine CSV-Kopfzeile mit Messwerten.")

    data_lines = []
    for line in lines[header_index:]:
        if not line.strip():
            continue
        if line.startswith("---"):
            break
        if line.startswith(
            (
                "Ended:",
                "Total samples:",
                "Within tolerance:",
                "Exceeded tolerance:",
                "STATUS:",
            )
        ):
            break
        data_lines.append(line)

    csv_df = pd.read_csv(StringIO("\n".join(data_lines)))
    optional_numeric_columns = ["Temp Tgt (C)", "Fan %"]
    for column in [*REQUIRED_COLUMNS, *optional_numeric_columns]:
        if column in csv_df.columns:
            csv_df[column] = pd.to_numeric(csv_df[column], errors="coerce")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in csv_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{csv_path} enthaelt nicht alle benoetigten Spalten: "
            f"{', '.join(missing_columns)}"
        )

    return csv_df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def _validate_csv(csv_df, csv_path):
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in csv_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{csv_path} enthaelt nicht alle benoetigten Spalten: "
            f"{', '.join(missing_columns)}"
        )
    if csv_df.empty:
        raise ValueError(f"{csv_path} enthaelt keine gueltigen Messzeilen.")


def _row_to_gui_data(row, reading_no):
    # Uebersetzt eine CSV-Zeile in das Format der MonitorApp.
    return {
        "reading_no": reading_no,
        "elapsed": reading_no - 1,
        "actual_temp": row["Temp Act (C)"],
        "target_temp": row.get("Temp Tgt (C)", DEFAULT_TARGET_TEMP),
        "extruder_rpm": row["Extruder RPM"],
        "puller_speed": row["Puller RPM"],
        "fil_width": row["Sensor Pos"],
        "fil_width_predicted": None,
        "fan_pct": row.get("Fan %", DEFAULT_FAN_PCT),
    }


def _update_actual_diameter(app, diameter):
    # Die normale GUI aktualisiert dieses Feld ueber eine echte Messuhr.
    # Beim Replay wird stattdessen der Durchmesser aus der CSV angezeigt.
    if pd.isna(diameter):
        app._metric_labels["fil_width"].config(text="--", fg=INK)
        return

    diameter = float(diameter)
    app._metric_labels["fil_width"].config(
        text=f"{diameter:.3f}",
        fg=GOOD if abs(diameter - DIA_TARGET) <= DIA_TOL else BAD,
    )


def main():
    args = _parse_args()
    csv_path = args.csv_path.resolve()
    csv_df = _load_recording_csv(csv_path)
    _validate_csv(csv_df, csv_path)

    root = tk.Tk()
    app = MonitorApp(root)
    root.title(f"CSV Prediction Test - {csv_path.name}")

    # Die echte Messuhr wird durch den Durchmesser aus der CSV ersetzt.
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

        data = _row_to_gui_data(row, row_index + 1)
        csv_diameter_reader.set(row["Diameter (mm)"])

        # Im opt-Branch erwartet das Modell pro Vorhersage genau eine Zeile
        # mit Temperatur, Extruder-RPM, Puller-RPM und Sensorposition.
        try:
            data["fil_width_predicted"] = app._predict_current_diameter(data)
            replay_state["prediction_error_reported"] = False
        except Exception as exc:
            if not replay_state["prediction_error_reported"]:
                print(f"Prediction failed: {exc}")
                replay_state["prediction_error_reported"] = True

        app._update_metrics(data)
        _update_actual_diameter(app, row["Diameter (mm)"])

        replay_state["row_index"] += 1
        if replay_state["row_index"] < len(csv_df):
            root.after(REPLAY_INTERVAL_MS, replay_next_row)

    root.after(0, replay_next_row)
    root.mainloop()


if __name__ == "__main__":
    main()
