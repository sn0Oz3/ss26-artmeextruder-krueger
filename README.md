# artme-Filamentextruder

## Beschreibung

Ziel dieses Projekts ist die datenbasierte Vorhersage des Filamentdurchmessers eines Artme-Extruders mithilfe von Künstlicher Intelligenz. Dadurch soll der Key-Performance-Indikator Qualität verbessert werden, indem mehr Filament mit dem gewünschten Durchmesser produziert und gleichzeitig weniger Ausschuss erzeugt wird. Im Extrusionsprozess ist ein möglichst konstanter Durchmesser wichtig, da Abweichungen die Qualität des gedruckten oder weiterverarbeiteten Materials beeinflussen können.

Im Kontext von **Künstliche Intelligenz und Digitalisierung im Maschinenbau** zeigt das Projekt einen praktischen Anwendungsfall für Predictive Monitoring: Prozessdaten aus einem realen technischen System werden aufgezeichnet, bereinigt, gefiltert und anschließend mit einem Machine-Learning-Modell ausgewertet. Eine solche Vorhersage kann perspektivisch helfen, ungünstige Prozessentwicklungen früher zu erkennen und Bedienpersonen oder spätere Regelungsstrategien zu unterstützen.

Als Datengrundlage werden aufgezeichnete CSV-Dateien des Extrusionsprozesses verwendet. Diese enthalten unter anderem den gemessenen Filamentdurchmesser, die aktuelle Temperatur, die Extruder-Drehzahl, die Puller-Drehzahl und die Sensorposition. Die Rohdaten werden zunächst vorverarbeitet: Metadaten und nicht benötigte Spalten werden entfernt, fehlerhafte Messpunkte werden bereinigt und der Durchmesser wird geglättet.

Die wichtigsten Eingangsgrößen des aktuellen Modells sind:

- `Temp Act (C)`: aktuelle Extrudertemperatur
- `Extruder RPM`: Drehzahl des Extruders
- `Puller RPM`: Geschwindigkeit des Pullers
- `Sensor Pos`: Sensorposition bzw. Filamentbreiten-Signal des Extruders

Die Zielgröße ist:

- `Diameter (mm)`: tatsächlich gemessener Filamentdurchmesser

Für die Vorhersage wird im aktuellen Branch `opt` ein Multi-Layer-Perceptron mit Keras/TensorFlow eingesetzt. Das Modell verwendet die vier Prozessgrößen als Eingabe und schätzt daraus den Filamentdurchmesser. Die Hyperparameter des neuronalen Netzes wurden mithilfe von Optuna optimiert. Anschließend werden das trainierte Modell, die verwendeten Scaler, die Trainingshistorie, die Optuna-Studie und ein Plot des Training- und Validation-Loss gespeichert.

## Verwendung

### Training und Evaluation

Die komplette Pipeline wird mit folgendem Befehl gestartet:

```bash
python src/main.py
```

Dabei werden nacheinander folgende Schritte ausgeführt:

1. Die Rohdaten aus `data/recording_2026-06-10_13-53-40.csv` werden eingelesen.
2. Die Daten werden vorverarbeitet und als `data/recording_2026-06-10_13-53-40_preprocessed.csv` gespeichert.
3. Features und Zielwert werden mit `StandardScaler` skaliert.
4. Optuna sucht geeignete Hyperparameter für das MLP.
5. Das beste Modell wird trainiert und auf den Testdaten bewertet.
6. Modell, Scaler, Trainingshistorie, Optuna-Studie und Trainingsplot werden im Ordner `models/` gespeichert.

Die wichtigsten gespeicherten Artefakte sind:

- `models/mlp_model.pkl`: trainiertes Keras-Modell
- `models/scalers.pkl`: Feature- und Zielwert-Scaler
- `models/training_history.pkl`: Trainingshistorie
- `models/optuna_study.pkl`: gespeicherte Optuna-Studie
- `models/hyperparameters.txt`: dokumentierte Hyperparameter und Kennzahlen
- `models/training_loss.png`: Plot von Training Loss und Validation Loss

### Einzelne Vorhersage

Eine Beispielvorhersage kann direkt über `prediction.py` gestartet werden:

```bash
python src/prediction.py
```

Das Skript lädt das gespeicherte Modell aus `models/`, skaliert die Eingabewerte und gibt den vorhergesagten Durchmesser in Millimetern aus.

### GUI-Test mit CSV-Replay

Die GUI kann ohne angeschlossenen Extruder mit einer gespeicherten CSV-Datei getestet werden:

```bash
python src/test_gui_prediction.py
```

Optional kann ein anderer CSV-Pfad übergeben werden:

```bash
python src/test_gui_prediction.py data/meine_aufnahme.csv
```

Die CSV wird zeilenweise abgespielt. Die GUI zeigt die Messwerte und den vom Modell vorhergesagten Filamentdurchmesser an.

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

Das finale Modell im aktuellen Branch `opt` ist ein MLP mit fünf versteckten Schichten. Die besten Hyperparameter wurden mit 30 Optuna-Trials bestimmt.

Wichtige Trainingsdaten aus `models/hyperparameters.txt`:

- Feature-Spalten: `Temp Act (C)`, `Extruder RPM`, `Puller RPM`, `Sensor Pos`
- Ziel-Spalte: `Diameter (mm)`
- Input-Dimension: 4 Werte
- Hidden Layer: `[8, 16, 128, 32, 32]`
- Hidden Activation: ReLU
- Output Activation: linear
- Dropout Rate: `0.0`
- Optimizer: Adam
- Learning Rate: `0.001829`
- Loss: Mean Squared Error
- Metrik: Mean Absolute Error
- Batch Size: `8`
- Early Stopping: aktiv, `patience=10`

Finale Modellgüte:

- Final Validation Loss: `0.0554`
- Test MSE: `0.0583`
- Test MAE, skaliert: `0.1570`

Der Test-MAE ist in `models/hyperparameters.txt` im skalierten Zielraum angegeben. Die tatsächliche Abweichung in Millimetern hängt vom gespeicherten Zielwert-Scaler in `models/scalers.pkl` ab.

<p align="center">
  <img src="models/training_loss.png" alt="Training Loss" width="720">
</p>

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
|   +-- mlp_model.pkl
|   +-- optuna_study.pkl
|   +-- scalers.pkl
|   +-- training_history.pkl
|   +-- training_loss.png
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

## Autor

- Simon Krüger
