from pathlib import Path

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

plt.close("all")

from data_preprocessing import plot_preprocessed, preprocess_data, write_preprocessed_csv
from evaluation import evaluate_model
from model import train_model


def main():
    #Pre-Processing
    plt.close("all")

    csv_filename = "data/recording_2026-06-10_13-53-40.csv" # "data/recording_2026-06-09_13-55-05.csv"
    csv_path = Path(__file__).resolve().parent.parent / csv_filename
    cleaned = preprocess_data(csv_path)

    output_path = write_preprocessed_csv(cleaned, csv_path)
    plot_preprocessed(cleaned)

    print(f"Preprocessing abgeschlossen für: {csv_path}")
    print(f"Ausgabe gespeichert unter: {output_path}")
    print(cleaned.head())

    # Model Training 
    # 40 Trials reichen fuer den bewusst kleinen Optuna-Suchraum aus.
    model_result = train_model(output_path, n_trials=40)
    print(model_result["message"])

    # Model Evaluation
    evaluation_result = evaluate_model(
        model_path=model_result["model_path"],
        scaler_path=model_result["scaler_path"],
        history_path=model_result["history_path"],
        data_path=model_result["data_path"],
    )
    print(evaluation_result)


if __name__ == "__main__":
    main()
