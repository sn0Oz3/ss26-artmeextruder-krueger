# artme-Filamentextruder

## Beschreibung

Ziel dieses Projekts ist die datenbasierte Vorhersage des Filamentdurchmessers eines Artme-Extruders mithilfe von Künstlicher Intelligenz. Dadurch soll der Key-Performance-Indikator Qualitaet verbessert werden, indem mehr Filament mit dem gewuenschten Durchmesser produziert und gleichzeitig weniger Ausschuss erzeugt wird. Im Extrusionsprozess ist ein moeglichst konstanter Durchmesser wichtig, da Abweichungen die Qualitaet des gedruckten oder weiterverarbeiteten Materials beeinflussen koennen. Das Projekt untersucht daher, ob aus Prozessdaten wie Temperatur, Extruder-Drehzahl, Puller-Drehzahl, Sensorposition und bisherigen Durchmesserwerten eine Vorhersage des zukuenftigen Durchmessers moeglich ist.

Im Kontext von Kuenstliche Intelligenz und Digitalisierung im Maschinenbau zeigt das Projekt, wie Messdaten aus einem technischen Prozess genutzt werden koennen, um Prozessverhalten zu analysieren und Qualitaetsabweichungen fruehzeitig zu erkennen. Eine solche Vorhersage kann perspektivisch dabei helfen, Regelungsstrategien zu verbessern oder Bedienpersonen rechtzeitig auf unguenstige Prozessentwicklungen hinzuweisen.

Als Datengrundlage werden aufgezeichnete CSV-Dateien des Extrusionsprozesses verwendet. Diese enthalten unter anderem den gemessenen Filamentdurchmesser, die aktuelle Temperatur, die Extruder-Drehzahl, die Puller-Drehzahl und die Sensorposition. Die Daten werden zunaechst vorverarbeitet und um zusaetzliche Trend-Features ergaenzt, zum Beispiel kurzfristige Aenderungen und Steigungen des Durchmessers.

Fuer die Vorhersage wird ein Multi-Layer-Perceptron eingesetzt. Das Modell betrachtet Sequenzen vergangener Messwerte und sagt die Aenderung des Durchmessers mehrere Zeilen in die Zukunft voraus. Die Hyperparameter des neuronalen Netzes werden mithilfe von Optuna optimiert. Anschliessend werden das trainierte Modell, die verwendeten Scaler, die Trainingshistorie und ein Plot des Training- und Validation-Loss gespeichert.

Erwartet wird ein Modell, das den zukuenftigen Filamentdurchmesser zuverlaessiger abschaetzt als eine einfache Baseline, bei der angenommen wird, dass der Durchmesser unveraendert bleibt. Im aktuellen Trainingslauf erreicht das Modell einen Test-MAE von etwa `0.0124 mm` und liegt damit unter der Baseline von etwa `0.0143 mm`.

<p align="center">
  <img src="models/training_loss.png" alt="Training Loss" width="720">
</p>

## Autor

- Simon Krüger
