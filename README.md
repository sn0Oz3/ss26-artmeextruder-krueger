# artme Filamentextruder 

## Beschreibung

Ziel dieses Projekts ist die datenbasierte Vorhersage des Filamentdurchmessers eines Artme-Extruders mithilfe von Künstlicher Intelligenz. Dadurch soll der Key-Performance-Indikator Qualität verbessert werden, indem mehr Filament mit dem gewünschten Durchmesser produziert und gleichzeitig weniger Ausschuss erzeugt wird. Im Extrusionsprozess ist ein möglichst konstanter Durchmesser wichtig, da Abweichungen die Qualität des gedruckten oder weiterverarbeiteten Materials beeinflussen können. Das Projekt untersucht daher, ob aus Prozessdaten wie Temperatur, Extruder-Drehzahl, Puller-Drehzahl, Sensorposition und bisherigen Durchmesserwerten eine Vorhersage des zukünftigen Durchmessers möglich ist.

Im Kontext von **Künstliche Intelligenz und Digitalisierung im Maschinenbau** zeigt das Projekt einen praktischen Anwendungsfall für Predictive Monitoring: Prozessdaten aus einem realen technischen System werden aufgezeichnet, bereinigt, gefiltert und anschließend mit einem Machine-Learning-Modell ausgewertet.

Verwendet werden CSV-Aufzeichnungen des Extruders und einer Messuhr. Die wichtigsten Eingangsgrößen sind:

- `Temp Act (C)`: aktuelle Extrudertemperatur
- `Extruder RPM`: Drehzahl des Extruders
- `Puller RPM`: Geschwindigkeit des Pullers
- `Sensor Pos`: Sensorposition bzw. Filamentbreiten-Signal des Extruders

Die Zielgröße ist:

- `Diameter (mm)`: tatsächlich gemessener Filamentdurchmesser

Als KI-Methode wird ein Multi-Layer-Perceptron (MLP) mit Keras/TensorFlow eingesetzt. Das Modell bekommt nicht nur eine einzelne Messzeile, sondern eine Sequenz der vorherigen 60 Messpunkte. Daraus wird der nächste Filamentdurchmesser vorhergesagt.

## Verwendung

### Training und Evaluation

Die komplette Pipeline wird mit folgendem Befehl gestartet:

```bash
python src/main.py
```

Dabei werden nacheinander folgende Schritte ausgeführt:

1. `data/recording_2026-06-10_13-53-40.csv` wird eingelesen.
2. Metadaten, konstante Spalten und fehlerhafte Messpunkte werden entfernt.
3. Der Durchmesser wird mit einem Butterworth-Tiefpassfilter geglättet.
4. Die bereinigten Daten werden als `data/recording_2026-06-10_13-53-40_preprocessed.csv` gespeichert.
5. Features und Zielwert werden mit `StandardScaler` skaliert.
6. Aus jeweils 60 vorherigen Messzeilen wird eine Eingabesequenz für den nächsten Durchmesser gebildet.
7. Optuna sucht geeignete Hyperparameter für das MLP.
8. Das beste Modell wird trainiert, bewertet und gespeichert.
9. Trainings- und Validierungsverlauf werden geplottet.

Die trainierten Artefakte werden im Ordner `models/` abgelegt:

- `mlp_model.pkl`: trainiertes Keras-Modell
- `scalers.pkl`: Feature- und Zielwert-Scaler sowie Metadaten zur Sequenzlänge
- `training_history.pkl`: Trainingshistorie
- `optuna_study.pkl`: gespeicherte Optuna-Studie
- `hyperparameters.txt`: dokumentierte Hyperparameter und Kennzahlen

### Einzelne Vorhersage

Eine Beispielvorhersage kann direkt über `prediction.py` gestartet werden:

```bash
python src/prediction.py
```

Das Skript lädt das gespeicherte Modell aus `models/`, skaliert eine Beispielsequenz und gibt den vorhergesagten Durchmesser in Millimetern aus.

### GUI-Test mit CSV-Replay

Die GUI kann ohne angeschlossenen Extruder mit einer gespeicherten CSV-Datei getestet werden:

```bash
python src/test_gui_prediction.py
```

Optional kann ein anderer CSV-Pfad übergeben werden:

```bash
python src/test_gui_prediction.py data/meine_aufnahme.csv
```

Die CSV wird zeilenweise abgespielt. Sobald genug vorherige Messpunkte vorhanden sind, zeigt die GUI neben dem aktuellen Durchmesser auch die vorhergesagte nächste Filamentbreite inklusive Test-MAE an.

### Live-GUI

Die Live-GUI für den Extruder kann direkt gestartet werden:

```bash
python src/extruder_gui.py
```

Die seriellen Ports sind in `src/extruder_gui.py` konfiguriert:

- `SERIAL_PORT = "COM4"` für den ArtME-Extruder
- `MESSUHR_PORT = "COM6"` für die Messuhr

Bei anderer Hardwarekonfiguration müssen diese Werte angepasst werden.

## Modell und Ergebnisse

Das finale Modell ist ein MLP mit einer versteckten Schicht und 16 Neuronen. Die besten Hyperparameter wurden mit 80 Optuna-Trials bestimmt.

Wichtige Trainingsdaten aus `models/hyperparameters.txt`:

- Feature-Spalten: `Temp Act (C)`, `Extruder RPM`, `Puller RPM`, `Sensor Pos`
- Ziel-Spalte: `Diameter (mm)`
- Sequenzlänge: 60 Messzeilen
- Input-Dimension: 240 Werte
- Optimizer: Adam
- Loss: Mean Squared Error
- Metrik: Mean Absolute Error
- Early Stopping: aktiv, `patience=5`

Finale Modellgüte:

- Test MSE: `0.1990`
- Test MAE, skaliert: `0.3389`
- Test MAE, rückskaliert: `0.0238 mm`

Der rückskalierte Test-MAE bedeutet, dass die mittlere absolute Abweichung der Vorhersage auf den Testdaten bei ungefähr `0.024 mm` liegt. Damit kann die Vorhersage als Orientierung für die Prozessüberwachung genutzt werden, ersetzt aber keine physische Messung des Durchmessers.

## Projektstruktur

```text
.
+-- README.md
+-- requirements.txt
+-- data/
|   +-- recording_2026-06-10_13-53-40.csv
|   +-- recording_2026-06-10_13-53-40_preprocessed.csv
+-- models/
|   +-- hyperparameters.txt
+-- src/
    +-- data_preprocessing.py
    +-- evaluation.py
    +-- extruder_gui.py
    +-- main.py
    +-- model.py
    +-- prediction.py
    +-- test_gui_prediction.py
```

## Anforderungen

- Python >= 3.10
- Abhängigkeiten aus `requirements.txt`
- TensorFlow/Keras für das MLP
- Optuna für die Hyperparameter-Suche
- PySerial für die Live-Kommunikation mit Extruder und Messuhr
- Tkinter für die GUI

## Datenquelle

Die verwendeten Daten stammen aus einer eigenen Messaufzeichnung am ArtME-3D-MK2-Filamentextruder. Die Rohdaten liegen unter:

```text
data/recording_2026-06-10_13-53-40.csv
```

Die vorverarbeitete Version wird automatisch erzeugt und liegt unter:

```text
data/recording_2026-06-10_13-53-40_preprocessed.csv
```

## Autoren

- Simon Krüger
