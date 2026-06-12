import tkinter as tk

from extruder_gui import MonitorApp

# Skript zum Testen der GUI -> Ich muss nicht immer zum Filamentextruder laufen, um die Vorhersage zu testen. Es werden Beispielwerte übergeben, um die Funktionalität der Vorhersage und der GUI-Integration zu überprüfen.
def main():
    root = tk.Tk()
    app = MonitorApp(root)

    data = {
        "reading_no": 1,
        "elapsed": 0,
        "actual_temp": 187.9,
        "target_temp": 190.0,
        "extruder_rpm": 12.0,
        "puller_speed": 23.56,
        "fil_width": 6.62,
        "fan_pct": 45,
    }

    data["fil_width_predicted"] = app._predict_current_diameter(data)
    app._update_metrics(data)

    root.mainloop()


if __name__ == "__main__":
    main()
