import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    df = df[
        ["price", "year", "odometer", "manufacturer",
         "fuel", "transmission", "drive", "type", "condition"]
    ]

    # Convert numerics safely
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

    df = df.dropna(subset=["price", "year", "odometer"])
    df = df[(df["price"] > 500) & (df["odometer"] >= 0)]

    # Vehicle age
    df["vehicle_age"] = 2025 - df["year"]
    df["age_squared"] = df["vehicle_age"] ** 2
    df["log_odometer"] = np.log1p(df["odometer"])

    # Log target
    df["log_price"] = np.log1p(df["price"])

    # Keep common vehicle types
    df = df[df["type"].isin(["sedan", "suv", "truck", "pickup"])]

    # REMOVE RARE MANUFACTURERS (KEY BOOST)
    counts = df["manufacturer"].value_counts()
    common = counts[counts >= 300].index
    df = df[df["manufacturer"].isin(common)]

    return df
