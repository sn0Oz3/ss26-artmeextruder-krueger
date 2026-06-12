import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_mlp_model(
    input_dim=4,
    hidden_layer_1_units=32,
    hidden_layer_2_units=16,
    dropout_rate=0.1,
    hidden_activation="relu",
    output_activation="linear",
    optimizer="adam",
    loss="mse",
    metrics=None,
):
    try:
        from keras import Sequential
        from keras.layers import Dense, Dropout, Input
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras wird fuer das Multi-Layer-Perceptron benoetigt. "
            "Installiere es z.B. mit: pip install tensorflow"
        ) from exc

    if metrics is None:
        metrics = ["mae"]

    # Ein einfaches Multi-Layer-Perceptron mit zwei versteckten Schichten und Dropout zur Regularisierung. ----> Dropout-Rate von 0.1 bedeutet, dass 10% der Neuronen in den versteckten Schichten zufaellig deaktiviert werden, um Overfitting zu reduzieren.
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(hidden_layer_1_units, activation=hidden_activation),
            Dropout(dropout_rate),
            Dense(hidden_layer_2_units, activation=hidden_activation),
            Dense(1, activation=output_activation),
        ]
    )

    model.compile(
        optimizer=optimizer,  # Lernrate = 0.001 by default (Adams)
        loss=loss,
        metrics=metrics,
    )

    return model


def train_model(data):
    csv_path = Path(data)

    feature_columns = ['Temp Act (C)', 'Extruder RPM', 'Puller RPM', 'Sensor Pos']
    target_column = 'Diameter (mm)'

    # Parameter Multi-Layer-Perceptron
    hidden_layer_1_units = 32
    hidden_layer_2_units = 16
    dropout_rate = 0.1
    hidden_activation = "relu"
    output_activation = "linear"
    optimizer = "adam"
    loss = "mse"
    metrics = ["mae"]
    test_size = 0.2
    random_state = 42
    shuffle = True
    epochs = 100
    batch_size = 32
    early_stopping_monitor = "val_loss"
    early_stopping_mode = "min"
    early_stopping_patience = 10
    early_stopping_min_delta = 0.0001
    early_stopping_restore_best_weights = True

    # Read the CSV passed from main.py directly.
    csv_df = pd.read_csv(csv_path)

    # Extract the measurement columns and map them into a NumPy array as requested:
    feature_array = csv_df[feature_columns].to_numpy()

    # Store all values from the Diameter (mm) column in a second NumPy array.
    diameter_array = csv_df[[target_column]].to_numpy()

    # Scaler -> StandardScaler, da Temp gaaaanz groß und Durchmesser gaaaanz klein
    scaler_feature = StandardScaler()
    scaler_diameter = StandardScaler()
    feature_array_scaled = scaler_feature.fit_transform(feature_array)
    diameter_array_scaled = scaler_diameter.fit_transform(diameter_array)

    # Train/Test Split fuer das spaetere Training eines Multi-Layer-Perceptrons.
    X_train, X_test, y_train, y_test = train_test_split(
        feature_array_scaled,
        diameter_array_scaled,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    #MLP wird gebaut
    mlp_model = build_mlp_model(
        input_dim=X_train.shape[1],
        hidden_layer_1_units=hidden_layer_1_units,
        hidden_layer_2_units=hidden_layer_2_units,
        dropout_rate=dropout_rate,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    try:
        from keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras wird fuer das Multi-Layer-Perceptron benoetigt. "
            "Installiere es z.B. mit: pip install tensorflow"
        ) from exc

    #Early-Stopping-Callback, um das Training zu beenden, wenn die Validierungsleistung nicht mehr verbessert wird. ->Auswendig lernen vermeiden, überparametrisierung
    early_stopping = EarlyStopping(
        monitor=early_stopping_monitor,
        mode=early_stopping_mode,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        restore_best_weights=early_stopping_restore_best_weights,
    )

    #UX-Improvement
    print("Training des Models gestartet...")

    #model.fit
    history = mlp_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0,
    )

    #Speichern der Trainingshistorie, um sie spaeter in der Evaluation anzuzeigen.
    test_loss, test_mae = mlp_model.evaluate(X_test, y_test, verbose=0)

    models_dir = Path(__file__).resolve().parent.parent / "Models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "mlp_model.pkl"
    scaler_path = models_dir / "scalers.pkl"
    history_path = models_dir / "training_history.pkl"
    hyperparameter_path = models_dir / "hyperparameters.txt"

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

    # Ausgabe der Hyperparameter und finalen Ergebnisse in einer Textdatei fuer die Dokumentation.
    hyperparameter_text = (
        "Hyperparameter des MLP-Modells\n"
        "==============================\n\n"
        f"Datenquelle: {csv_path}\n"
        f"Feature-Spalten: {', '.join(feature_columns)}\n"
        f"Ziel-Spalte: {target_column}\n\n"
        "Modellarchitektur\n"
        "-----------------\n"
        f"Input Dimension: {X_train.shape[1]}\n"
        f"Hidden Layer 1 Neuronen: {hidden_layer_1_units}\n"
        f"Hidden Layer 2 Neuronen: {hidden_layer_2_units}\n"
        f"Hidden Activation: {hidden_activation}\n"
        f"Output Activation: {output_activation}\n"
        f"Dropout Rate: {dropout_rate}\n\n"
        "Training\n"
        "--------\n"
        f"Optimizer: {optimizer}\n"
        f"Loss: {loss}\n"
        f"Metrics: {', '.join(metrics)}\n"
        f"Epochs: {epochs}\n"
        f"Tatsaechlich trainierte Epochen: {len(history.history['loss'])}\n"
        f"Batch Size: {batch_size}\n"
        f"Test Size: {test_size}\n"
        f"Random State: {random_state}\n"
        f"Shuffle: {shuffle}\n\n"
        "Early Stopping\n"
        "--------------\n"
        f"Monitor: {early_stopping_monitor}\n"
        f"Mode: {early_stopping_mode}\n"
        f"Patience: {early_stopping_patience}\n"
        f"Min Delta: {early_stopping_min_delta}\n"
        f"Restore Best Weights: {early_stopping_restore_best_weights}\n\n"
        "Finale Ergebnisse\n"
        "-----------------\n"
        f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n"
        f"Test MSE: {test_loss:.4f}\n"
        f"Test MAE: {test_mae:.4f}\n"
    )
    hyperparameter_path.write_text(hyperparameter_text, encoding="utf-8")

    # Ausgabe im Terminal
    message = (
        # print(feature_array_scaled),
        f"Model trained on {len(csv_df)} rows "
        f"with feature array shape {feature_array_scaled.shape}"
        f"and diameter array shape {diameter_array_scaled.shape}"
        f", train shape {X_train.shape}/{y_train.shape}"
        f", test shape {X_test.shape}/{y_test.shape}"
        f", trained epochs {len(history.history['loss'])}"
        f", final validation loss {history.history['val_loss'][-1]:.4f}"
        f", test MSE {test_loss:.4f}"
        f", test MAE {test_mae:.4f}"
        f", model saved to {model_path}"
        f", scalers saved to {scaler_path}"
        f", training history saved to {history_path}"
        f", hyperparameters saved to {hyperparameter_path}"
    )

    return {
        "message": message,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "history_path": history_path,
        "hyperparameter_path": hyperparameter_path,
        "data_path": csv_path,
    }
