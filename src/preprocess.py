"""
preprocess.py
-------------
Data loading and preprocessing for the Car Recommendation System.
Handles cleaning, feature selection, encoding, and scaling.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


# ── Column aliases ──────────────────────────────────────────────────────────
RENAME_MAP = {
    "Engine CC":    "Engine(CC)",
    "Mileage Km/L": "Mileage(Km/L)",
}

FEATURE_COLS = [
    "Manufacturer", "Fuel_Type", "Transmission",
    "Engine(CC)", "Mileage(Km/L)", "Seats", "Price",
]

CATEGORICAL_COLS = ["Manufacturer", "Fuel_Type", "Transmission"]
NUMERIC_COLS     = ["Engine(CC)", "Mileage(Km/L)", "Seats", "Price"]


# ── Public API ───────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV, drop unneeded columns, rename, and clean.

    Returns a DataFrame with columns:
        Name, Manufacturer, Fuel_Type, Transmission,
        Engine(CC), Mileage(Km/L), Seats, Price
    """
    df = pd.read_csv(csv_path)

    # Rename to unified column names
    df = df.rename(columns=RENAME_MAP)

    # Keep only relevant columns
    keep = ["Name"] + FEATURE_COLS
    df = df[[c for c in keep if c in df.columns]]

    # Drop LPG (too few records to be useful)
    df = df[df["Fuel_Type"] != "LPG"]

    # Normalise manufacturer casing
    df["Manufacturer"] = df["Manufacturer"].str.title()

    # Drop rows with missing values in key columns
    df = df.dropna(subset=FEATURE_COLS)

    # Sanity-check seat count
    df = df[df["Seats"] <= 7]

    df = df.reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    One-hot encode categorical columns and return (X, scaler, columns).

    Returns
    -------
    X_scaled : np.ndarray  – scaled feature matrix
    scaler   : MinMaxScaler
    columns  : pd.Index    – column names after get_dummies
    """
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, X.columns


def encode_user_input(user_input: dict, columns, scaler) -> "np.ndarray":
    """
    Transform a single user-input dict into a scaled feature vector
    that matches the training column layout.

    Parameters
    ----------
    user_input : dict  with keys matching FEATURE_COLS
    columns    : pd.Index returned by build_feature_matrix
    scaler     : fitted MinMaxScaler

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    import pandas as pd

    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=columns, fill_value=0)
    return scaler.transform(user_df)


def save_artifacts(model, scaler, columns,
                   model_path="models/knn_model.pkl",
                   scaler_path="models/scaler.pkl",
                   columns_path="models/columns.pkl"):
    """Persist the trained model, scaler, and column list to disk."""
    with open(model_path, "wb")   as f: pickle.dump(model,   f)
    with open(scaler_path, "wb")  as f: pickle.dump(scaler,  f)
    with open(columns_path, "wb") as f: pickle.dump(columns, f)
    print(f"Saved → {model_path}, {scaler_path}, {columns_path}")


def load_artifacts(model_path="models/knn_model.pkl",
                   scaler_path="models/scaler.pkl",
                   columns_path="models/columns.pkl"):
    """Load and return (model, scaler, columns) from disk."""
    with open(model_path, "rb")   as f: model   = pickle.load(f)
    with open(scaler_path, "rb")  as f: scaler  = pickle.load(f)
    with open(columns_path, "rb") as f: columns = pickle.load(f)
    return model, scaler, columns
