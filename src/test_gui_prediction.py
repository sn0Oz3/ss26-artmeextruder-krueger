import argparse
from pathlib import Path
import tkinter as tk

import pandas as pd

from extruder_gui import BAD, DIA_TARGET, DIA_TOL, GOOD, INK, MonitorApp
from prediction import DEFAULT_FEATURE_COLUMNS, DEFAULT_SEQUENCE_LENGTH, predict_diameter

# ----------------------------------------------------------------------------------------------------------------------
# Test der Gui: CSV-Datei wird zeilenweise abgespielt; ab Zeile 41 wird auch aus den vorherigen 40 Zeilen der naechste Durchmesser vorhergesagt.
# ----------------------------------------------------------------------------------------------------------------------
# Wie schnell die CSV-Zeilen nacheinander in der GUI angezeigt werden.
REPLAY_INTERVAL_MS = 500

# Diese Werte stehen nicht in der preprocessed CSV, werden aber von der GUI erwartet.
DEFAULT_TARGET_TEMP = 190.0
DEFAULT_FAN_PCT = 45

# Standarddatei, falls beim Starten des Skripts kein anderer CSV-Pfad angegeben wird.
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "recording_2026-06-10_13-53-40_preprocessed.csv"
)


class CsvDiameterReader:
    # Kleine Ersatz-Messuhr fuer den GUI-Test.
    # Die echte GUI erwartet ein Objekt mit get(), das den aktuellen Durchmesser liefert.
    # Hier kommt dieser Durchmesser stattdessen aus der CSV-Datei.
    def __init__(self):
        self.connected = True
        self.diameter = None

    def get(self):
        # Die GUI ruft diese Methode regelmaessig auf, um den Live-Durchmesser zu bekommen.
        return self.diameter

    def set(self, diameter):
        # Beim CSV-Replay wird hier pro Zeile der aktuelle CSV-Durchmesser gesetzt.
        if pd.isna(diameter):
            self.diameter = None
        else:
            self.diameter = float(diameter)


def _parse_args():
    # Erlaubt optional einen eigenen CSV-Pfad beim Starten:
    # python src/test_gui_prediction.py data/meine_datei_preprocessed.csv
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
    # Prueft, ob die CSV alle Spalten enthaelt, die GUI und Prediction brauchen.
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
    # Uebersetzt eine CSV-Zeile in das Dictionary-Format, das MonitorApp._update_metrics erwartet.
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
    # Die normale GUI zeigt den echten Durchmesser nur an, wenn eine echte Messuhr laeuft.
    # Im Test schreiben wir den CSV-Durchmesser deshalb direkt in das passende GUI-Feld.
    if pd.isna(diameter):
        app._metric_labels["fil_width"].config(text="--", fg=INK)
        return

    app._metric_labels["fil_width"].config(
        text=f"{diameter:.3f}",
        fg=GOOD if abs(diameter - DIA_TARGET) <= DIA_TOL else BAD,
    )


def _update_predicted_diameter_with_error(app, prediction, actual_diameter):
    # Zeigt die Vorhersage zusammen mit dem absoluten Fehler an:
    # Beispiel: 1.734 +/- 0.012
    if prediction is None or pd.isna(actual_diameter):
        app._metric_labels["fil_width_predicted"].config(text="--", fg=INK)
        return

    prediction_error = abs(prediction - actual_diameter)
    app._metric_labels["fil_width_predicted"].config(
        text=f"{prediction:.3f} +/- {prediction_error:.3f}",
        fg=GOOD if prediction_error <= DIA_TOL else BAD,
    )


# Skript zum Testen der GUI -> Ich muss nicht immer zum Filamentextruder laufen, um die Vorhersage zu testen.
# Die CSV wird zeilenweise abgespielt; ab Zeile 41 wird aus den vorherigen 40 Zeilen der naechste Durchmesser vorhergesagt.
def main():
    # CSV laden und pruefen, bevor die GUI gestartet wird.
    args = _parse_args()
    csv_path = args.csv_path.resolve()
    csv_df = pd.read_csv(csv_path)
    _validate_csv(csv_df, csv_path)

    # Normale Monitor-GUI erstellen.
    root = tk.Tk()
    app = MonitorApp(root)
    root.title(f"CSV Prediction Test - {csv_path.name}")

    # Die echte Messuhr wird durch den CSV-Durchmesser ersetzt.
    # Dadurch kann auch das Live-Diagramm den CSV-Durchmesser zeichnen.
    csv_diameter_reader = CsvDiameterReader()
    app.messuhr_obj = csv_diameter_reader
    app.messuhr_online = True

    # row_index merkt sich, welche CSV-Zeile als naechstes abgespielt wird.
    # prediction_error_reported verhindert, dass derselbe Fehler dauernd ins Terminal geschrieben wird.
    replay_state = {
        "row_index": 0,
        "prediction_error_reported": False,
    }

    def replay_next_row():
        # Diese Funktion wird von Tkinter immer wieder aufgerufen und spielt jeweils eine CSV-Zeile ab.
        row_index = replay_state["row_index"]
        row = csv_df.iloc[row_index]
        prediction = None

        if row_index >= DEFAULT_SEQUENCE_LENGTH:
            # Fuer eine Vorhersage braucht das Modell die vorherigen DEFAULT_SEQUENCE_LENGTH Zeilen.
            # Die aktuelle Zeile ist dann der Wert, mit dem der Fehler verglichen wird.
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

        # CSV-Zeile in GUI-Daten umwandeln und alle Anzeigen aktualisieren.
        data = _row_to_gui_data(row, row_index + 1, prediction)
        csv_diameter_reader.set(row["Diameter (mm)"])
        app._update_metrics(data)
        _update_actual_diameter(app, row["Diameter (mm)"])
        _update_predicted_diameter_with_error(app, prediction, row["Diameter (mm)"])

        replay_state["row_index"] += 1
        if replay_state["row_index"] < len(csv_df):
            # root.after startet den naechsten Replay-Schritt nach REPLAY_INTERVAL_MS Millisekunden,
            # ohne die Tkinter-Oberflaeche zu blockieren.
            root.after(REPLAY_INTERVAL_MS, replay_next_row)

    # Replay starten, sobald die GUI bereit ist.
    root.after(0, replay_next_row)
    root.mainloop()


if __name__ == "__main__":
    main()
