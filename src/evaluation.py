import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def evaluate_model(model_path, scaler_path, history_path, data_path=None):
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    history_path = Path(history_path)

    required_files = [model_path, scaler_path, history_path]
    if data_path is not None:
        required_files.append(Path(data_path))

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Folgende Dateien fehlen fuer die Evaluation: {missing_list}")

    with history_path.open("rb") as history_file:
        history = pickle.load(history_file)

    train_loss = history.get("loss", [])
    validation_loss = history.get("val_loss", [])

    if not train_loss or not validation_loss:
        raise ValueError("Training Loss oder Validation Loss fehlen in der Trainingshistorie.")

    print("Evaluation gestartet...")
    print(f"Finaler Training Loss: {train_loss[-1]:.4f}")
    print(f"Finaler Validation Loss: {validation_loss[-1]:.4f}")

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Training Loss", linewidth=1.8)
    ax.plot(epochs, validation_loss, label="Validation Loss", linewidth=1.8)
    ax.set_title("Training Loss und Validation Loss")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    plt.show()

    return {
        "final_training_loss": train_loss[-1],
        "final_validation_loss": validation_loss[-1],
        "epochs": len(train_loss),
    }
