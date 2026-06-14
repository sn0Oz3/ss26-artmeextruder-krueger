import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _keras_import_error(exc):
    # Einheitliche Fehlermeldung, falls Keras/TensorFlow nicht installiert ist.
    raise ImportError(
        "TensorFlow/Keras wird fuer das Multi-Layer-Perceptron benoetigt. "
        "Installiere es z.B. mit: pip install tensorflow"
    ) from exc


def _create_optimizer(learning_rate):
    # Adam ist der Optimizer, der die Gewichte des Netzes beim Training anpasst.
    # Die learning_rate bestimmt, wie gross diese Anpassungen pro Schritt sind.
    try:
        from keras.optimizers import Adam
    except ImportError as exc:
        _keras_import_error(exc)

    return Adam(learning_rate=learning_rate)


def _create_early_stopping(
    monitor,
    mode,
    patience,
    min_delta,
    restore_best_weights,
):
    # EarlyStopping beendet das Training automatisch, wenn sich die Validation
    # Loss ueber mehrere Epochen nicht mehr verbessert. Dadurch wird Overfitting reduziert.
    try:
        from keras.callbacks import EarlyStopping
    except ImportError as exc:
        _keras_import_error(exc)

    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
    )


def _clear_keras_session():
    # Bei der Optuna-Suche werden viele Modelle nacheinander gebaut.
    # clear_session raeumt alte Keras-Modelle aus dem Speicher.
    try:
        from keras import backend

        backend.clear_session()
    except ImportError:
        pass


def build_mlp_model(
    input_dim=4,
    hidden_units=None,
    dropout_rate=0.1,
    l2_rate=0.0,
    hidden_activation="relu",
    output_activation="linear",
    optimizer="adam",
    loss="mse",
    metrics=None,
):
    # Baut ein MLP mit beliebig vielen versteckten Schichten.
    # hidden_units ist z.B. [32, 16] fuer zwei Hidden Layer mit 32 und 16 Neuronen.
    try:
        from keras import Sequential
        from keras.layers import Dense, Dropout, Input
        from keras.regularizers import l2
    except ImportError as exc:
        _keras_import_error(exc)

    if hidden_units is None:
        hidden_units = [32, 16]
    if metrics is None:
        metrics = ["mae"]

    # Die Layer-Liste wird Schritt fuer Schritt aufgebaut:
    # Input -> beliebig viele Dense/Dropout-Bloecke -> Output.
    kernel_regularizer = l2(l2_rate) if l2_rate > 0 else None
    layers = [Input(shape=(input_dim,))]
    for units in hidden_units:
        # L2 bestraft zu grosse Gewichte und kann dadurch Overfitting reduzieren.
        layers.append(
            Dense(
                units,
                activation=hidden_activation,
                kernel_regularizer=kernel_regularizer,
            )
        )
        if dropout_rate > 0:
            # Dropout deaktiviert zufaellig Neuronen im Training und hilft gegen Overfitting.
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(1, activation=output_activation))

    model = Sequential(layers)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


def create_sequences(features, targets, sequence_length):
    # Aus einzelnen Messzeilen werden Sequenzen gebaut.
    # Beispiel bei sequence_length=60:
    # Zeilen 0-59 werden zum Input, der Durchmesser aus Zeile 60 ist das Ziel.
    if len(features) <= sequence_length:
        raise ValueError(
            f"Mindestens {sequence_length + 1} Datenzeilen werden benoetigt, "
            "um aus den vorherigen Zeilen den naechsten Durchmesser zu lernen."
        )

    sequence_features = []
    sequence_targets = []

    for row_index in range(sequence_length, len(features)):
        previous_rows = features[row_index - sequence_length:row_index]
        # Ein MLP erwartet einen flachen Vektor. Deshalb wird aus (60, 4) ein Vektor mit 240 Werten.
        sequence_features.append(previous_rows.flatten())
        sequence_targets.append(targets[row_index])

    return np.array(sequence_features), np.array(sequence_targets)


def _suggest_hyperparameters(trial):
    # Optuna bekommt hier den Suchraum: Es darf z.B. Anzahl Layer,
    # Neuronen, Dropout, L2-Regularisierung, Lernrate, Batch Size
    # und Sequenzlaenge ausprobieren.
    hidden_layer_count = trial.suggest_int("hidden_layer_count", 1, 4)
    hidden_units = [
        trial.suggest_categorical(f"hidden_layer_{i + 1}_units", [8, 16, 32, 64])
        for i in range(hidden_layer_count)
    ]

    return {
        "hidden_units": hidden_units,
        "dropout_rate": trial.suggest_float("dropout_rate", 0.15, 0.5, step=0.05),
        "l2_rate": trial.suggest_float("l2_rate", 1e-5, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "sequence_length": trial.suggest_categorical("sequence_length", [40, 60]),
    }


def _split_sequence_data(sequence_features, sequence_targets, test_size, validation_size):
    # Zeitreihendaten werden nicht gemischt. Dadurch trainiert das Modell auf frueheren
    # Messungen und wird auf spaeteren Messungen getestet.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequence_features,
        sequence_targets,
        test_size=test_size,
        shuffle=False,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_size,
        shuffle=False,
    )

    # Ergebnis: Training fuer das Lernen, Validation fuer Optuna/EarlyStopping,
    # Test erst ganz am Ende fuer die finale Bewertung.
    return X_train, X_val, X_test, y_train, y_val, y_test


def _optimize_hyperparameters(
    feature_array_scaled,
    diameter_array_scaled,
    random_state,
    n_trials,
    test_size,
    validation_size,
    epochs,
    loss,
    metrics,
    hidden_activation,
    output_activation,
    early_stopping_settings,
):
    # Diese Funktion fuehrt die Optuna-Suche aus.
    # Jeder Trial ist ein kompletter Testlauf mit einer anderen Modellarchitektur.
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna wird fuer die automatische Hyperparameter-Suche benoetigt. "
            "Installiere es z.B. mit: pip install optuna"
        ) from exc

    def objective(trial):
        # Ein Trial fragt zuerst neue Hyperparameter aus dem Suchraum ab.
        params = _suggest_hyperparameters(trial)

        # Weil auch die sequence_length gesucht wird, muessen die Trainingssequenzen
        # fuer jeden Trial neu erstellt werden.
        sequence_features, sequence_targets = create_sequences(
            feature_array_scaled,
            diameter_array_scaled,
            params["sequence_length"],
        )
        X_train, X_val, _, y_train, y_val, _ = _split_sequence_data(
            sequence_features,
            sequence_targets,
            test_size,
            validation_size,
        )

        # Fuer jeden Trial wird ein frisches Modell gebaut und trainiert.
        _clear_keras_session()
        model = build_mlp_model(
            input_dim=X_train.shape[1],
            hidden_units=params["hidden_units"],
            dropout_rate=params["dropout_rate"],
            l2_rate=params["l2_rate"],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimizer=_create_optimizer(params["learning_rate"]),
            loss=loss,
            metrics=metrics,
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=params["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[_create_early_stopping(**early_stopping_settings)],
            verbose=0,
        )

        # Optuna minimiert diesen Wert. Je kleiner die beste Validation Loss,
        # desto besser war dieser Trial.
        best_validation_loss = min(history.history["val_loss"])
        trial.set_user_attr("trained_epochs", len(history.history["loss"]))
        trial.set_user_attr("input_dim", X_train.shape[1])
        _clear_keras_session()
        return best_validation_loss

    # TPESampler ist Optunas Standard-Strategie fuer eine intelligente Suche.
    # Der Seed macht die Suche besser reproduzierbar.
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study


def _format_best_params(best_params):
    # Macht aus den besten Optuna-Parametern lesbaren Text fuer hyperparameters.txt.
    lines = []
    for key, value in best_params.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def train_model(data, n_trials=80):
    # Hauptfunktion: wird aus main.py aufgerufen und trainiert am Ende das Modell.
    # n_trials bestimmt, wie viele Modellvarianten Optuna testen soll.
    csv_path = Path(data)

    feature_columns = ['Temp Act (C)', 'Extruder RPM', 'Puller RPM', 'Sensor Pos']
    target_column = 'Diameter (mm)'

    hidden_activation = "relu"
    output_activation = "linear"
    optimizer_name = "adam"
    loss = "mse"
    metrics = ["mae"]
    test_size = 0.2
    validation_size = 0.2
    random_state = 42
    shuffle = True
    epochs = 100
    early_stopping_settings = {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 5,
        "min_delta": 0.0003,
        "restore_best_weights": True,
    }

    # CSV-Datei aus main.py einlesen.
    csv_df = pd.read_csv(csv_path)

    # Features sind die Eingabewerte fuer das Modell.
    feature_array = csv_df[feature_columns].to_numpy()

    # Target ist der Wert, den das Modell vorhersagen soll.
    diameter_array = csv_df[[target_column]].to_numpy()

    # Scaler -> StandardScaler, da die Features sehr unterschiedliche Wertebereiche haben.
    # Wichtig: Erst skalieren, danach werden daraus Sequenzen gebildet.
    scaler_feature = StandardScaler()
    scaler_diameter = StandardScaler()
    feature_array_scaled = scaler_feature.fit_transform(feature_array)
    diameter_array_scaled = scaler_diameter.fit_transform(diameter_array)

    print("Optuna-Suche nach besten Hyperparametern gestartet...")
    # Optuna sucht auf einem Validation-Split nach einer guten Architektur.
    # Der Test-Split bleibt dabei unangetastet.
    study = _optimize_hyperparameters(
        feature_array_scaled=feature_array_scaled,
        diameter_array_scaled=diameter_array_scaled,
        random_state=random_state,
        n_trials=n_trials,
        test_size=test_size,
        validation_size=validation_size,
        epochs=epochs,
        loss=loss,
        metrics=metrics,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        early_stopping_settings=early_stopping_settings,
    )

    # Die besten gefundenen Parameter werden aus der Optuna-Study gelesen.
    best_params = study.best_params
    hidden_layer_count = best_params["hidden_layer_count"]
    hidden_units = [
        best_params[f"hidden_layer_{i + 1}_units"]
        for i in range(hidden_layer_count)
    ]
    dropout_rate = best_params["dropout_rate"]
    l2_rate = best_params["l2_rate"]
    optimizer_learning_rate = best_params["learning_rate"]
    batch_size = best_params["batch_size"]
    sequence_length = best_params["sequence_length"]

    # Mit der besten sequence_length werden die finalen Sequenzen gebaut.
    sequence_features, sequence_targets = create_sequences(
        feature_array_scaled,
        diameter_array_scaled,
        sequence_length,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = _split_sequence_data(
        sequence_features,
        sequence_targets,
        test_size,
        validation_size,
    )

    # Jetzt wird das finale Modell mit den besten Hyperparametern gebaut.
    _clear_keras_session()
    mlp_model = build_mlp_model(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        l2_rate=l2_rate,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        optimizer=_create_optimizer(optimizer_learning_rate),
        loss=loss,
        metrics=metrics,
    )

    print("Training des besten Models gestartet...")
    # Finales Training. Validation wird hier weiterhin fuer EarlyStopping genutzt.
    history = mlp_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[_create_early_stopping(**early_stopping_settings)],
        verbose=0,
    )

    # Finale Bewertung auf Training, Validation und Test.
    # Die Testdaten wurden vorher nicht fuer Optuna oder EarlyStopping genutzt.
    train_loss, train_mae = mlp_model.evaluate(X_train, y_train, verbose=0)
    validation_loss, validation_mae = mlp_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_mae = mlp_model.evaluate(X_test, y_test, verbose=0)

    # MAE liegt nach dem Training im skalierten Durchmesserraum.
    # Multiplikation mit scale_[0] macht daraus wieder Millimeter.
    diameter_scale_mm = scaler_diameter.scale_[0]
    final_training_mae_mm = train_mae * diameter_scale_mm
    final_validation_mae_mm = validation_mae * diameter_scale_mm
    test_mae_mm = test_mae * diameter_scale_mm

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "mlp_model.pkl"
    scaler_path = models_dir / "scalers.pkl"
    history_path = models_dir / "training_history.pkl"
    hyperparameter_path = models_dir / "hyperparameters.txt"
    optuna_study_path = models_dir / "optuna_study.pkl"

    # Modell und alle Hilfsdaten speichern, damit prediction.py spaeter dieselbe
    # Skalierung und Sequenzlaenge verwenden kann.
    with model_path.open("wb") as model_file:
        pickle.dump(mlp_model, model_file)

    with scaler_path.open("wb") as scaler_file:
        pickle.dump(
            {
                "feature_scaler": scaler_feature,
                "diameter_scaler": scaler_diameter,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "sequence_length": sequence_length,
                "test_mae_mm": test_mae_mm,
            },
            scaler_file,
        )

    with history_path.open("wb") as history_file:
        pickle.dump(history.history, history_file)

    # Die Optuna-Study wird mitgespeichert, damit die Trials spaeter analysiert werden koennen.
    with optuna_study_path.open("wb") as study_file:
        pickle.dump(study, study_file)

    # Textdatei fuer die Dokumentation des Trainingslaufs.
    hyperparameter_text = (
        "Hyperparameter des MLP-Modells\n"
        "==============================\n\n"
        f"Datenquelle: {csv_path}\n"
        f"Feature-Spalten: {', '.join(feature_columns)}\n"
        f"Ziel-Spalte: {target_column}\n\n"
        "Optuna\n"
        "------\n"
        f"Trials: {n_trials}\n"
        f"Best Validation Loss: {study.best_value:.4f}\n"
        f"Study gespeichert unter: {optuna_study_path}\n\n"
        "Beste Optuna-Parameter\n"
        "----------------------\n"
        f"{_format_best_params(best_params)}\n\n"
        "Modellarchitektur\n"
        "-----------------\n"
        f"Input Dimension: {X_train.shape[1]}\n"
        f"Original Feature Count: {len(feature_columns)}\n"
        f"Sequence Length: {sequence_length}\n"
        f"Sequenz-Ziel: vorherige {sequence_length} Zeilen -> naechster Durchmesser\n"
        f"Hidden Layers: {len(hidden_units)}\n"
        f"Hidden Layer Neuronen: {hidden_units}\n"
        f"Hidden Activation: {hidden_activation}\n"
        f"Output Activation: {output_activation}\n"
        f"Dropout Rate: {dropout_rate}\n"
        f"L2 Rate: {l2_rate}\n\n"
        "Training\n"
        "--------\n"
        f"Optimizer: {optimizer_name}\n"
        f"Learning Rate: {optimizer_learning_rate}\n"
        f"Loss: {loss}\n"
        f"Metrics: {', '.join(metrics)}\n"
        f"Epochs: {epochs}\n"
        f"Tatsaechlich trainierte Epochen: {len(history.history['loss'])}\n"
        f"Batch Size: {batch_size}\n"
        f"Test Size: {test_size}\n"
        f"Validation Size vom Trainingsanteil: {validation_size}\n"
        f"Random State: {random_state}\n"
        f"Shuffle: {shuffle}\n\n"
        "Early Stopping\n"
        "--------------\n"
        f"Monitor: {early_stopping_settings['monitor']}\n"
        f"Mode: {early_stopping_settings['mode']}\n"
        f"Patience: {early_stopping_settings['patience']}\n"
        f"Min Delta: {early_stopping_settings['min_delta']}\n"
        f"Restore Best Weights: {early_stopping_settings['restore_best_weights']}\n\n"
        "Finale Ergebnisse\n"
        "-----------------\n"
        f"Final Training Loss: {train_loss:.4f}\n"
        f"Final Validation Loss: {validation_loss:.4f}\n"
        f"Test MSE: {test_loss:.4f}\n"
        f"Test MAE: {test_mae:.4f}\n\n"
        "Rueckskalierte Fehler in mm\n"
        "--------------------------\n"
        f"Final Training MAE: {final_training_mae_mm:.4f} mm\n"
        f"Final Validation MAE: {final_validation_mae_mm:.4f} mm\n"
        f"Test MAE: {test_mae_mm:.4f} mm\n"
    )
    hyperparameter_path.write_text(hyperparameter_text, encoding="utf-8")

    # Kompakte Zusammenfassung fuer die Konsolenausgabe in main.py.
    message = (
        f"Model trained on {len(csv_df)} rows "
        f"with feature array shape {feature_array_scaled.shape}"
        f"and diameter array shape {diameter_array_scaled.shape}"
        f", sequence shape {sequence_features.shape}/{sequence_targets.shape}"
        f", sequence length {sequence_length}"
        f", train/validation/test shape "
        f"{X_train.shape}/{y_train.shape}, {X_val.shape}/{y_val.shape}, {X_test.shape}/{y_test.shape}"
        f", optuna trials {n_trials}"
        f", best optuna validation loss {study.best_value:.4f}"
        f", trained epochs {len(history.history['loss'])}"
        f", final validation loss {validation_loss:.4f}"
        f", test MSE {test_loss:.4f}"
        f", test MAE {test_mae:.4f}"
        f", test MAE {test_mae_mm:.4f} mm"
        f", model saved to {model_path}"
        f", scalers saved to {scaler_path}"
        f", training history saved to {history_path}"
        f", hyperparameters saved to {hyperparameter_path}"
        f", optuna study saved to {optuna_study_path}"
    )

    return {
        "message": message,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "history_path": history_path,
        "hyperparameter_path": hyperparameter_path,
        "optuna_study_path": optuna_study_path,
        "data_path": csv_path,
    }
