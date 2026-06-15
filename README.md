# artme-Filamentextruder

## Beschreibung

Ziel dieses Projekts ist die datenbasierte Vorhersage des Filamentdurchmessers eines Artme-Extruders mithilfe von Künstlicher Intelligenz. Dadurch soll der Key-Performance-Indikator Qualitaet verbessert werden, indem mehr Filament mit dem gewuenschten Durchmesser produziert und gleichzeitig weniger Ausschuss erzeugt wird. Im Extrusionsprozess ist ein moeglichst konstanter Durchmesser wichtig, da Abweichungen die Qualitaet des gedruckten oder weiterverarbeiteten Materials beeinflussen koennen. Das Projekt untersucht daher, ob aus Prozessdaten wie Temperatur, Extruder-Drehzahl, Puller-Drehzahl, Sensorposition und bisherigen Durchmesserwerten eine Vorhersage des zukuenftigen Durchmessers moeglich ist.

Im Kontext von Kuenstliche Intelligenz und Digitalisierung im Maschinenbau zeigt das Projekt, wie Messdaten aus einem technischen Prozess genutzt werden koennen, um Prozessverhalten zu analysieren und Qualitaetsabweichungen fruehzeitig zu erkennen. Eine solche Vorhersage kann perspektivisch dabei helfen, Regelungsstrategien zu verbessern oder Bedienpersonen rechtzeitig auf unguenstige Prozessentwicklungen hinzuweisen.

Als Datengrundlage werden aufgezeichnete CSV-Dateien des Extrusionsprozesses verwendet. Diese enthalten unter anderem den gemessenen Filamentdurchmesser, die aktuelle Temperatur, die Extruder-Drehzahl, die Puller-Drehzahl und die Sensorposition. Die Daten werden zunaechst vorverarbeitet und fuer das Modell in eine geeignete Trainings- und Teststruktur ueberfuehrt. Im aktuellen Modellstand werden keine zusaetzlichen Trend-Features wie kurzfristige Aenderungen oder Steigungen des Durchmessers ergaenzt.

Fuer die Vorhersage wird ein Multi-Layer-Perceptron eingesetzt. Das Modell betrachtet Sequenzen vergangener Messwerte und sagt die Aenderung des Durchmessers mehrere Zeilen in die Zukunft voraus. Die Hyperparameter des neuronalen Netzes werden mithilfe von Optuna optimiert. Anschliessend werden das trainierte Modell, die verwendeten Scaler, die Trainingshistorie und ein Plot des Training- und Validation-Loss gespeichert.

Im aktuellen Branch wird ein MLP-Modell verwendet, das aus den vier Prozessgroessen Temperatur, Extruder-Drehzahl, Puller-Drehzahl und Sensorposition den Filamentdurchmesser abschaetzt. Der aktuelle Trainingslauf erreicht einen finalen Validation Loss von `0.0554`, einen Test-MSE von `0.0583` und einen Test-MAE von `0.1570` im skalierten Zielraum. Rueckgerechnet entspricht das einem mittleren absoluten Fehler von etwa `0.0110 mm`.

<p align="center">
  <img src="models/training_loss.png" alt="Training Loss" width="720">
</p>

## Autor

- Simon Krüger
