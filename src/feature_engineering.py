DIAMETER_COLUMN = "Diameter (mm)"

RAW_FEATURE_COLUMNS = [
    DIAMETER_COLUMN,
    "Temp Act (C)",
    "Extruder RPM",
    "Puller RPM",
    "Sensor Pos",
]

DIAMETER_TREND_COLUMNS = [
    "Diameter Delta 5",
    "Diameter Slope 10",
]

MODEL_FEATURE_COLUMNS = [
    DIAMETER_COLUMN,
    *DIAMETER_TREND_COLUMNS,
    "Temp Act (C)",
    "Extruder RPM",
    "Puller RPM",
    "Sensor Pos",
]


def add_diameter_trend_features(df, diameter_column=DIAMETER_COLUMN):
    # Die Trend-Features nutzen nur aktuelle und vergangene Durchmesserwerte.
    # Dadurch sind sie auch live in der GUI verfuegbar und leaken keinen Zukunftswert.
    enriched_df = df.copy()
    diameter = enriched_df[diameter_column].astype(float)

    enriched_df["Diameter Delta 5"] = diameter.diff(5).fillna(0.0)
    enriched_df["Diameter Slope 10"] = (diameter - diameter.shift(10)).div(10).fillna(0.0)

    return enriched_df
