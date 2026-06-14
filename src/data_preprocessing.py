import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt


def write_preprocessed_csv(df: pd.DataFrame, csv_path: str | Path) -> Path:
    """Write the cleaned table to a *_preprocessed.csv file without a trailing blank line."""
    path = Path(csv_path)
    output_path = path.with_name(path.stem + "_preprocessed.csv")
    df.to_csv(output_path, index=False, lineterminator="\n")
    output_path.write_text(output_path.read_text(encoding="utf-8").rstrip("\r\n"), encoding="utf-8")
    return output_path


def plot_preprocessed(df: pd.DataFrame) -> None:       #Plot zum Vergleich gefiltert vs ungefiltert
    """Display the preprocessed Diameter series in a simple plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(df)), df["Diameter (mm)"], linewidth=1.5, color="#ff7f0e", label="Filtered Diameter (mm)")    

    raw_diameter = df.attrs.get("raw_diameter")
    if raw_diameter is not None:
        ax.plot(range(len(raw_diameter)), raw_diameter, linewidth=1.2, color="#1f77b4", label="Raw Diameter (mm)")  

    ax.set_title("Preprocessed Diameter over Time[s]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Diameter (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    plt.show()


def preprocess_data(csv_path: str | Path) -> pd.DataFrame:
    """Read a recording CSV, skip metadata and footer summary lines, remove the
    Timestamp column, and return the cleaned table.
    """
    path = Path(csv_path)

    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    # The first four lines are descriptive text; the real table starts at line 5.
    data_lines = [line for line in lines[4:] if line.strip()]

    # Ignore footer summary lines that are not numeric measurement rows.
    data_lines = [
        line
        for line in data_lines
        if not line.startswith("---")
        and not line.startswith("Ended:")
        and not line.startswith("Total samples:")
        and not line.startswith("Within tolerance:")
        and not line.startswith("Exceeded tolerance:")
        and not line.startswith("STATUS:")
    ]

    df = pd.read_csv(StringIO("".join(data_lines)), sep=",")

    # Remove the first column (Timestamp) from the table so it shifts left.
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # # Remove the third column from the left (0-based index 2) and shift the data left.
    # if len(df.columns) >= 3:
    #     df = df.drop(columns=[df.columns[2]])

    # Remove Temp Tgt (C) column if it exists, as it is not a measurement but a setpoint.
    if "Temp Tgt (C)" in df.columns:
        df = df.drop(columns=["Temp Tgt (C)"])
    
    # Remove Fan, weil const.
    if "Fan %" in df.columns:
        df = df.drop(columns=["Fan %"])

    raw_diameter = None
    if "Diameter (mm)" in df.columns:
        raw_diameter = df["Diameter (mm)"].copy()

    # Remove rows where Diameter (mm) is below 1. -> Anomalie, Filament würde reißen 
    if "Diameter (mm)" in df.columns:
        df = df.loc[pd.to_numeric(df["Diameter (mm)"], errors="coerce") >= 1]

    # Drop rows where at least two values are exactly 0. ----------------------> Spannungsversorgung bricht ein !!!!!!!!
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    zero_counts = (numeric_df == 0).sum(axis=1)
    df = df.loc[zero_counts < 2]

    # Zero-phase Butterworth low-pass filter on the first column (Diameter (mm)).  
    if "Diameter (mm)" in df.columns:
        fs = 1.0 / 1.0  # one sample per second approximation for the recording cadence
        cutoff = 0.05  # cutoff frequency in Hz (adjust as needed)
        b, a = butter(N=2, Wn=cutoff / (fs / 2.0), btype="low", analog=False)
        df["Diameter (mm)"] = filtfilt(b, a, df["Diameter (mm)"].astype(float))

    # Round all numeric measurement values to two decimal places.
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Remove any completely empty rows and compact the table.
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)

    if raw_diameter is not None:
        filtered_index = df.index.tolist()
        df.attrs["raw_diameter"] = raw_diameter.iloc[filtered_index].reset_index(drop=True)

    return df


if __name__ == "__main__":
    input_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("data/recording_2026-06-10_13-55-05.csv")
    cleaned = preprocess_data(input_path)
    print(cleaned.head())

    output_path = write_preprocessed_csv(cleaned, input_path)
    print(f"Ausgabe gespeichert unter: {output_path}")

    plot_preprocessed(cleaned)