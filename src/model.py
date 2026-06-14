import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _import_keras():
    # Keras wird erst bei Bedarf importiert, damit das Modul auch ohne
    # installierte ML-Abhaengigkeiten geladen werden kann.
    try:
        from keras import Sequential
        from keras.layers import Dense, Dropout, Input
        from keras.optimizers import Adam, RMSprop, SGD
        from keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras wird fuer das Multi-Layer-Perceptron benoetigt. "
            "Installiere es z.B. mit: pip install tensorflow keras"
        ) from exc

    return Sequential, Dense, Dropout, Input, Adam, RMSprop, SGD, EarlyStopping


def _build_optimizer(name, learning_rate):
    _, _, _, _, Adam, RMSprop, SGD, _ = _import_keras()
    # Die Namen aus den Hyperparametern werden hier auf konkrete Keras-
    # Optimizer-Klassen abgebildet.
    optimizers = {
        "adam": Adam,
        "rmsprop": RMSprop,
        "sgd": SGD,
    }
    return optimizers[name](learning_rate=learning_rate)


def build_mlp_model(
    input_dim=4,
    hidden_units=None,
    hidden_layer_1_units=32,
    hidden_layer_2_units=16,
    dropout_rate=0.1,
    hidden_activation="relu",
    output_activation="linear",
    optimizer="adam",
    learning_rate=0.001,
    loss="mse",
    metrics=None,
):
    Sequential, Dense, Dropout, Input, _, _, _, _ = _import_keras()

    # Standardwerte: zwei versteckte Layer und MAE als zusaetzliche Metrik.
    if metrics is None:
        metrics = ["mae"]
    if hidden_units is None:
        hidden_units = [hidden_layer_1_units, hidden_layer_2_units]

    # Das MLP besteht aus einem Eingabelayer, variabel vielen Dense-Layern
    # mit optionalem Dropout und einem linearen Ausgabeneuron fuer Regression.
    layers = [Input(shape=(input_dim,))]
    for units in hidden_units:
        layers.append(Dense(units, activation=hidden_activation))
        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(1, activation=output_activation))

    model = Sequential(layers)
    model.compile(
        optimizer=_build_optimizer(optimizer, learning_rate),
        loss=loss,
        metrics=metrics,
    )

    return model


def _suggest_hyperparameters(trial):
    # Optuna waehlt pro Trial Architektur und Trainingsparameter aus diesem
    # Suchraum aus.
    layer_count = trial.suggest_int("hidden_layer_count", 1, 6)                             #--------------------------------- Anzahl hidden Layer (zw. 1-6)------------------------------
    hidden_units = [
        trial.suggest_categorical(f"hidden_layer_{i + 1}_units", [8, 16, 32, 64, 128])      #---------------------------- Anzahl Neuronen pro hidden Layer ------------------------------
        for i in range(layer_count)
    ]

    return {
        "hidden_units": hidden_units,
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.05),
        "hidden_activation": trial.suggest_categorical(
            "hidden_activation", ["relu", "tanh", "elu"]
        ),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
    }


def _create_early_stopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True,
):
    *_, EarlyStopping = _import_keras()
    # Early Stopping beendet ein Training, wenn sich der Validierungsfehler
    # nicht mehr ausreichend verbessert, und stellt optional die besten
    # Gewichte wieder her.
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
    )


def _optimize_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    loss,
    metrics,
    epochs,
    early_stopping_settings,
    n_trials,
    random_state,
):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna wird fuer die automatische Hyperparameter-Suche benoetigt. "
            "Installiere es z.B. mit: pip install optuna"
        ) from exc

    try:
        import tensorflow as tf
    except ImportError:
        tf = None

    def objective(trial):
        # Vor jedem Trial wird die TensorFlow-Session geleert, damit alte
        # Modellgraphen keinen Speicher belegen und der Seed reproduzierbar ist.
        if tf is not None:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(random_state)

        params = _suggest_hyperparameters(trial)
        # Jeder Trial trainiert ein neues Modell mit den von Optuna
        # vorgeschlagenen Hyperparametern.
        model = build_mlp_model(
            input_dim=input_dim,
            hidden_units=params["hidden_units"],
            dropout_rate=params["dropout_rate"],
            hidden_activation=params["hidden_activation"],
            optimizer=params["optimizer"],
            learning_rate=params["learning_rate"],
            loss=loss,
            metrics=metrics,
        )

        history = model.fit(                                # model.fit trainiert MOdel
            X_train,
            y_train,
            epochs=epochs,
            batch_size=params["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[_create_early_stopping(**early_stopping_settings)],
            verbose=0,
        )

        # Optuna minimiert den besten Validierungs-Loss des jeweiligen Trials.
        return min(history.history["val_loss"])

    # Der TPE-Sampler nutzt die bisherigen Ergebnisse, um nach und nach
    # vielversprechendere Hyperparameter-Kombinationen vorzuschlagen.
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = _suggestion_dict_to_model_params(study.best_trial.params)

    return study, best_params


def _suggestion_dict_to_model_params(params):
    # Optuna speichert Layer einzeln im Trial-Dictionary; fuer den Modellbau
    # werden sie wieder zu einer Liste zusammengefuehrt.
    layer_count = params["hidden_layer_count"]
    return {
        "hidden_units": [
            params[f"hidden_layer_{i + 1}_units"] for i in range(layer_count)
        ],
        "dropout_rate": params["dropout_rate"],
        "hidden_activation": params["hidden_activation"],
        "optimizer": params["optimizer"],
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
    }


def train_model(data, n_trials=30):
    csv_path = Path(data)

    # Eingabegroessen fuer das Modell und Zielwert, der vorhergesagt werden soll.
    feature_columns = ["Temp Act (C)", "Extruder RPM", "Puller RPM", "Sensor Pos"]
    target_column = "Diameter (mm)"

    # Zentrale Trainingskonfiguration. Die eigentliche Architektur wird spaeter
    # durch Optuna bestimmt.
    output_activation = "linear"
    loss = "mse"
    metrics = ["mae"]
    test_size = 0.2
    validation_size = 0.2
    random_state = 42
    shuffle = True                 # Shuffle = True, damit Gewichtungen nicht hin und her springen koennen, wenn Daten nicht in zufaelliger Reihenfolge vorliegen (z.B. erst alle kleinen Durchmesser, dann alle grossen)
    epochs = 100
    early_stopping_settings = {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 10,
        "min_delta": 0.0001,
        "restore_best_weights": True,
    }

    # CSV laden und die benoetigten Spalten in NumPy-Arrays fuer Keras wandeln.
    csv_df = pd.read_csv(csv_path)
    feature_array = csv_df[feature_columns].to_numpy()
    diameter_array = csv_df[[target_column]].to_numpy()

    # Features und Zielwert werden standardisiert, damit das neuronale Netz auf
    # vergleichbaren Skalen trainiert.
    scaler_feature = StandardScaler()
    scaler_diameter = StandardScaler()
    feature_array_scaled = scaler_feature.fit_transform(feature_array)
    diameter_array_scaled = scaler_diameter.fit_transform(diameter_array)

    # Zuerst wird ein Testdatensatz abgetrennt. Aus dem verbleibenden Anteil
    # entsteht anschliessend ein Validierungsdatensatz fuer Optuna und Training.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        feature_array_scaled,
        diameter_array_scaled,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # Die Hyperparameter-Suche trainiert mehrere kleine Kandidatenmodelle und
    # liefert die beste gefundene Konfiguration zurueck.
    print("Optuna-Hyperparameter-Suche gestartet...")
    study, best_params = _optimize_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=X_train.shape[1],
        loss=loss,
        metrics=metrics,
        epochs=epochs,
        early_stopping_settings=early_stopping_settings,
        n_trials=n_trials,
        random_state=random_state,
    )

    # Mit den besten Hyperparametern wird das finale Modell noch einmal sauber
    # auf dem Trainingssplit trainiert.
    print("Training des besten Models gestartet...")
    mlp_model = build_mlp_model(
        input_dim=X_train.shape[1],
        hidden_units=best_params["hidden_units"],
        dropout_rate=best_params["dropout_rate"],
        hidden_activation=best_params["hidden_activation"],
        output_activation=output_activation,
        optimizer=best_params["optimizer"],
        learning_rate=best_params["learning_rate"],
        loss=loss,
        metrics=metrics,
    )
    history = mlp_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=best_params["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=[_create_early_stopping(**early_stopping_settings)],
        verbose=0,
    )

    # Die finale Bewertung erfolgt nur auf dem Testsplit, der bei Suche und
    # Training nicht verwendet wurde.
    test_loss, test_mae = mlp_model.evaluate(X_test, y_test, verbose=0)

    # Alle Ergebnisse werden im Projektordner models abgelegt, damit Prediction
    # und Auswertung spaeter dieselben Artefakte verwenden koennen.
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "mlp_model.pkl"
    scaler_path = models_dir / "scalers.pkl"
    history_path = models_dir / "training_history.pkl"
    hyperparameter_path = models_dir / "hyperparameters.txt"
    optuna_study_path = models_dir / "optuna_study.pkl"

    with model_path.open("wb") as model_file:
        pickle.dump(mlp_model, model_file)

    with scaler_path.open("wb") as scaler_file:
        pickle.dump(
            {
                "feature_scaler": scaler_feature,
                "diameter_scaler": scaler_diameter,
            },
            scaler_file,
        )

    with history_path.open("wb") as history_file:
        pickle.dump(history.history, history_file)

    with optuna_study_path.open("wb") as study_file:
        pickle.dump(study, study_file)

    # Textdatei mit den wichtigsten Einstellungen und Kennzahlen fuer eine
    # schnelle Nachvollziehbarkeit ausserhalb von Python.
    hyperparameter_text = (
        "Hyperparameter des MLP-Modells\n"
        "==============================\n\n"
        f"Datenquelle: {csv_path}\n"
        f"Feature-Spalten: {', '.join(feature_columns)}\n"
        f"Ziel-Spalte: {target_column}\n\n"
        "Optuna\n"
        "------\n"
        f"Trials: {n_trials}\n"
        f"Best Trial: {study.best_trial.number}\n"
        f"Best Validation Loss: {study.best_value:.4f}\n"
        f"Study gespeichert unter: {optuna_study_path}\n\n"
        "Modellarchitektur\n"
        "-----------------\n"
        f"Input Dimension: {X_train.shape[1]}\n"
        f"Hidden Layer Anzahl: {len(best_params['hidden_units'])}\n"
        f"Hidden Layer Neuronen: {best_params['hidden_units']}\n"
        f"Hidden Activation: {best_params['hidden_activation']}\n"
        f"Output Activation: {output_activation}\n"
        f"Dropout Rate: {best_params['dropout_rate']}\n\n"
        "Training\n"
        "--------\n"
        f"Optimizer: {best_params['optimizer']}\n"
        f"Learning Rate: {best_params['learning_rate']:.6f}\n"
        f"Loss: {loss}\n"
        f"Metrics: {', '.join(metrics)}\n"
        f"Max Epochs pro Trial/finalem Training: {epochs}\n"
        f"Tatsaechlich trainierte Epochen final: {len(history.history['loss'])}\n"
        f"Batch Size: {best_params['batch_size']}\n"
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
        f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n"
        f"Test MSE: {test_loss:.4f}\n"
        f"Test MAE: {test_mae:.4f}\n"
    )
    hyperparameter_path.write_text(hyperparameter_text, encoding="utf-8")

    # Die Rueckgabe enthaelt sowohl eine kompakte Zusammenfassung als auch die
    # Pfade zu allen erzeugten Artefakten.
    message = (
        f"Model trained on {len(csv_df)} rows "
        f"with feature array shape {feature_array_scaled.shape}"
        f" and diameter array shape {diameter_array_scaled.shape}"
        f", train shape {X_train.shape}/{y_train.shape}"
        f", validation shape {X_val.shape}/{y_val.shape}"
        f", test shape {X_test.shape}/{y_test.shape}"
        f", optuna trials {n_trials}"
        f", best trial {study.best_trial.number}"
        f", best optuna validation loss {study.best_value:.4f}"
        f", trained epochs {len(history.history['loss'])}"
        f", final validation loss {history.history['val_loss'][-1]:.4f}"
        f", test MSE {test_loss:.4f}"
        f", test MAE {test_mae:.4f}"
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
