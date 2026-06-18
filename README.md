# artme-Filamentextruder: Delta-Vorhersage (10 Sekunden)

## Beschreibung

Ziel dieses Projekts ist die datenbasierte Vorhersage des zukünftigen Filamentdurchmessers eines Artme-Extruders mithilfe von Künstlicher Intelligenz. Im Branch `delta-Vorhersage-(10-Sekunden)` wird nicht direkt der absolute Durchmesser gelernt, sondern die Änderung des Durchmessers gegenüber dem letzten bekannten Messwert. Dadurch soll abgeschätzt werden, wie sich der Filamentdurchmesser etwa 10 Sekunden in die Zukunft entwickelt.

Im Extrusionsprozess ist ein möglichst konstanter Durchmesser wichtig, da Abweichungen die Qualität des gedruckten oder weiterverarbeiteten Materials beeinflussen können. Das Projekt unterstützt damit den Key-Performance-Indikator Qualität: Mehr Filament soll im gewünschten Durchmesserbereich produziert werden, während Ausschuss und ungünstige Prozesszustände früher erkannt werden können.

Im Kontext von **Künstliche Intelligenz und Digitalisierung im Maschinenbau** zeigt das Projekt einen praktischen Anwendungsfall für Predictive Monitoring. Messdaten aus einem realen technischen Prozess werden aufgezeichnet, vorverarbeitet, um Trendinformationen ergänzt und anschließend mit einem Machine-Learning-Modell ausgewertet. Eine solche Vorhersage kann perspektivisch helfen, Bedienpersonen frühzeitig auf kritische Entwicklungen hinzuweisen oder spätere Regelungsstrategien zu unterstützen.

## Daten und Features

Als Datengrundlage werden CSV-Aufzeichnungen des Extrusionsprozesses verwendet. Die Rohdaten enthalten unter anderem den gemessenen Filamentdurchmesser, die aktuelle Temperatur, die Extruder-Drehzahl, die Puller-Drehzahl und die Sensorposition.

Die Rohdaten werden zunächst vorverarbeitet. Dabei werden Metadaten und nicht benötigte Spalten entfernt, fehlerhafte Messpunkte bereinigt und der Durchmesser geglättet. Anschließend werden zusätzliche Trend-Features aus dem bisherigen Durchmesserverlauf berechnet.

Die Modellfeatures sind:

- `Diameter (mm)`: bisher gemessener Filamentdurchmesser
- `Diameter Delta 5`: Änderung des Durchmessers gegenüber fünf Zeilen zuvor
- `Diameter Slope 10`: Steigung des Durchmessers über zehn Zeilen
- `Temp Act (C)`: aktuelle Extrudertemperatur
- `Extruder RPM`: Drehzahl des Extruders
- `Puller RPM`: Geschwindigkeit des Pullers
- `Sensor Pos`: Sensorposition bzw. Filamentbreiten-Signal des Extruders

Die Zielgröße ist:

- Delta zum zukünftigen `Diameter (mm)`: Durchmesser in 10 Zeilen minus letzter Eingabe-Durchmesser

Da die Messdaten ungefähr im Sekundentakt verarbeitet werden, entspricht der Prediction Horizon von 10 Zeilen näherungsweise einer Vorhersage 10 Sekunden in die Zukunft.

## Modellansatz

Für die Vorhersage wird ein Multi-Layer-Perceptron mit Keras/TensorFlow eingesetzt. Das Modell betrachtet Sequenzen der vergangenen 60 Messzeilen. Jede Zeile enthält 7 Features, wodurch ein flacher Eingabevektor mit 420 Werten entsteht.

Das Modell gibt ein Durchmesser-Delta aus. Für die absolute Vorhersage wird dieses Delta zum letzten gemessenen Durchmesser der Eingabesequenz addiert. Diese Vorgehensweise wird mit einer einfachen Baseline verglichen, bei der angenommen wird, dass der Durchmesser unverändert bleibt.

Die Hyperparameter des neuronalen Netzes werden mit Optuna optimiert. Dabei werden die kleinen Modellarchitekturen `[16]`, `[16, 8]` und `[32, 16, 8]` sowie verschiedene Werte für Dropout, L2-Regularisierung, Lernrate und Batchgröße getestet. Der Branch verwendet außerdem Early Stopping und Learning-Rate-Reduktion bei stagnierender Validation Loss.

## Verwendung

### Training und Evaluation

Die komplette Pipeline wird mit folgendem Befehl gestartet:

```bash
python src/main.py
```

Dabei werden nacheinander folgende Schritte ausgeführt:

1. `data/recording_2026-06-10_13-53-40.csv` wird eingelesen.
2. Die Daten werden vorverarbeitet und als `data/recording_2026-06-10_13-53-40_preprocessed.csv` gespeichert.
3. Die Durchmesser-Trendfeatures werden ergänzt.
4. Features und Zielwert werden mit `StandardScaler` skaliert.
5. Aus jeweils 60 vorherigen Messzeilen wird eine Eingabesequenz gebildet.
6. Ziel ist das Durchmesser-Delta in 10 Zeilen.
7. Optuna sucht geeignete Hyperparameter für das MLP.
8. Das beste Modell wird trainiert, bewertet und gespeichert.
9. Training Loss und Validation Loss werden als Plot gespeichert.

Die wichtigsten Artefakte im Branch sind:

- `models/mlp_model.pkl`: trainiertes Keras-Modell
- `models/scalers.pkl`: Feature- und Zielwert-Scaler sowie Metadaten zu Sequenzlänge, Prediction Horizon und Zielmodus
- `models/hyperparameters.txt`: dokumentierte Hyperparameter und Kennzahlen
- `models/training_loss.png`: Plot von Training Loss und Validation Loss

Beim erneuten Training können zusätzlich lokale Dateien wie `models/training_history.pkl` und `models/optuna_study.pkl` entstehen.

### Einzelne Vorhersage

Eine Beispielvorhersage kann direkt über `prediction.py` gestartet werden:

```bash
python src/prediction.py
```

Das Skript lädt das gespeicherte Modell aus `models/`, berechnet bei Bedarf die Trendfeatures, skaliert die Eingabesequenz und gibt den vorhergesagten Durchmesser in Millimetern aus.

### GUI-Test mit CSV-Replay

Die GUI kann ohne angeschlossenen Extruder mit einer gespeicherten CSV-Datei getestet werden:

```bash
python src/test_gui_prediction.py
```

Optional kann ein anderer CSV-Pfad übergeben werden:

```bash
python src/test_gui_prediction.py data/meine_aufnahme.csv
```

Die CSV wird zeilenweise abgespielt. Sobald mindestens 60 vorherige Messzeilen vorhanden sind, zeigt die GUI den vorhergesagten Durchmesser für 10 Zeilen in die Zukunft an.

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

Das finale Modell im Branch `delta-Vorhersage-(10-Sekunden)` ist ein MLP mit zwei versteckten Schichten. Die besten Hyperparameter des dokumentierten Trainingslaufs wurden mit 40 Optuna-Trials bestimmt.

Wichtige Trainingsdaten aus `models/hyperparameters.txt`:

- Feature-Spalten: `Diameter (mm)`, `Diameter Delta 5`, `Diameter Slope 10`, `Temp Act (C)`, `Extruder RPM`, `Puller RPM`, `Sensor Pos`
- Ziel-Modus: Delta zum zukünftigen Durchmesser
- Sequenzlänge: 60 Messzeilen
- Prediction Horizon: 10 Zeilen
- Input-Dimension: 420 Werte
- Bester Optuna-Trial: `34`
- Beste Validation-Metrik (`val_loss`): `0.0245`
- Architektur: `small_16_8`
- Hidden Layer: `[16, 8]`
- Hidden Activation: ReLU
- Output Activation: linear
- Dropout Rate: `0.25`
- L2 Rate: `0.001`
- Optimizer: Adam
- Learning Rate: `0.001`
- Loss: Huber
- Metrik: Mean Absolute Error
- Epochen: maximal `100`, tatsächlich trainiert `33`
- Batch Size: `32`
- Random State: `42`
- Final Training Seed: `76`
- Early Stopping: aktiv, Monitor `val_loss`, `patience=7`, `min_delta=5e-05`
- Learning-Rate-Reduktion: Faktor `0.5`, `patience=3`, minimale Lernrate `1e-06`

Finale Modellgüte:

- Final Training Loss: `0.0144`
- Final Validation Loss: `0.0245`
- Test Loss: `0.0428`
- Test MAE, skaliert: `0.1704`
- Test MAE, rückskaliert: `0.0120 mm`

Baseline: letzter Durchmesser bleibt gleich:

- Baseline Test Loss: `0.0616`
- Baseline Test MAE, skaliert: `0.2033`
- Baseline Test MAE, rückskaliert: `0.0143 mm`

Das Modell verbessert die rückskalierte Testabweichung gegenüber der Baseline von `0.0143 mm` auf `0.0120 mm`.

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
|   +-- scalers.pkl
|   +-- training_loss.png
+-- src/
    +-- data_preprocessing.py
    +-- evaluation.py
    +-- extruder_gui.py
    +-- feature_engineering.py
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

Die vorverarbeitete Version liegt unter:

```text
data/recording_2026-06-10_13-53-40_preprocessed.csv
```

## Autor

- Simon Krüger
