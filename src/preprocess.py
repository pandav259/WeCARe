"""
preprocess.py
-------------
Data loading and preprocessing for the Car Recommendation System.
Handles cleaning, feature engineering, encoding, and scaling.

New in v2:
  - Car_Age, Usage_Intensity, Owner_Type_Enc, Power, Location_Enc features
  - Richer FEATURE_COLS used by both KNN and Random Forest price predictor
  - load_rf_artifacts / save_rf_artifacts for the price prediction model
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


# ── Column aliases ──────────────────────────────────────────────────────────
RENAME_MAP = {
    "Engine CC":    "Engine(CC)",
    "Mileage Km/L": "Mileage(Km/L)",
}

# Columns used by the KNN recommender (user-facing filters)
FEATURE_COLS = [
    "Manufacturer", "Fuel_Type", "Transmission",
    "Engine(CC)", "Mileage(Km/L)", "Seats", "Price",
]

# Extended feature set used by the Random Forest price predictor
RF_FEATURE_COLS = [
    "Engine(CC)", "Mileage(Km/L)", "Seats", "Power",
    "Car_Age", "Usage_Intensity", "Owner_Type_Enc", "Location_Enc",
    "Fuel_Type_Enc", "Transmission_Enc",
]

CATEGORICAL_COLS = ["Manufacturer", "Fuel_Type", "Transmission"]
NUMERIC_COLS     = ["Engine(CC)", "Mileage(Km/L)", "Seats", "Price"]

# Encoding maps (deterministic — same at training and inference time)
OWNER_TYPE_MAP = {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4}
FUEL_TYPE_MAP  = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}


# ── Public API ───────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV, engineer new features, drop unneeded columns, and clean.

    Returns a DataFrame with columns:
        Name, Manufacturer, Fuel_Type, Transmission,
        Engine(CC), Mileage(Km/L), Seats, Price,
        Power, Car_Age, Usage_Intensity,
        Owner_Type_Enc, Fuel_Type_Enc, Transmission_Enc, Location_Enc
    """
    df = pd.read_csv(csv_path)

    # Rename to unified column names
    df = df.rename(columns=RENAME_MAP)

    # ── Feature engineering ──────────────────────────────────────────────────
    REFERENCE_YEAR = 2024

    # Car age (years since manufacture)
    if "Year" in df.columns:
        df["Car_Age"] = REFERENCE_YEAR - df["Year"]
        df["Car_Age"] = df["Car_Age"].clip(lower=1)  # avoid division by 0

    # Usage intensity (km per year of car's life)
    if "Kilometers_Driven" in df.columns and "Car_Age" in df.columns:
        df["Usage_Intensity"] = df["Kilometers_Driven"] / df["Car_Age"]

    # Owner type ordinal encoding
    if "Owner_Type" in df.columns:
        df["Owner_Type_Enc"] = df["Owner_Type"].map(OWNER_TYPE_MAP).fillna(3)

    # Fuel, transmission, location numeric encodings
    df["Fuel_Type_Enc"]     = df["Fuel_Type"].map(FUEL_TYPE_MAP).fillna(0)
    df["Transmission_Enc"]  = df["Transmission"].map(TRANSMISSION_MAP).fillna(0)

    if "Location" in df.columns:
        locs = sorted(df["Location"].dropna().unique())
        loc_map = {loc: i for i, loc in enumerate(locs)}
        df["Location_Enc"] = df["Location"].map(loc_map).fillna(0)
    else:
        df["Location_Enc"] = 0

    # ── Standard cleaning ────────────────────────────────────────────────────
    # Drop LPG (too few records to be useful for recommendations)
    df = df[df["Fuel_Type"] != "LPG"]

    # Normalise manufacturer casing
    df["Manufacturer"] = df["Manufacturer"].str.title()

    # Drop rows with missing values in key columns
    core_cols = ["Name", "Manufacturer", "Fuel_Type", "Transmission",
                 "Engine(CC)", "Mileage(Km/L)", "Seats", "Price"]
    df = df.dropna(subset=core_cols)

    # Sanity-check seat count
    df = df[df["Seats"] <= 7]

    # Fill Power NaN with median
    if "Power" in df.columns:
        df["Power"] = pd.to_numeric(df["Power"], errors="coerce")
        df["Power"].fillna(df["Power"].median(), inplace=True)
    else:
        df["Power"] = 80.0  # fallback default

    df = df.reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    One-hot encode categorical columns and return (X_scaled, scaler, columns).
    Used by the KNN recommender.

    Returns
    -------
    X_scaled : np.ndarray  – scaled feature matrix
    scaler   : MinMaxScaler
    columns  : pd.Index    – column names after get_dummies
    """
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, X.columns


def build_rf_feature_matrix(df: pd.DataFrame):
    """
    Build the numeric feature matrix for the Random Forest price predictor.

    Returns
    -------
    X : pd.DataFrame  – raw (unscaled) feature matrix
    y : pd.Series     – target (Price)
    """
    available = [c for c in RF_FEATURE_COLS if c in df.columns]
    X = df[available].copy().fillna(0)
    y = df["Price"]
    return X, y


def encode_user_input(user_input: dict, columns, scaler) -> "np.ndarray":
    """
    Transform a single user-input dict into a scaled feature vector
    that matches the KNN training column layout.

    Parameters
    ----------
    user_input : dict  with keys matching FEATURE_COLS
    columns    : pd.Index returned by build_feature_matrix
    scaler     : fitted MinMaxScaler

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=columns, fill_value=0)
    return scaler.transform(user_df)


def encode_user_input_rf(user_input_rf: dict, rf_columns) -> "np.ndarray":
    """
    Encode a user input dict for the RF price predictor.

    Parameters
    ----------
    user_input_rf : dict with keys matching RF_FEATURE_COLS
    rf_columns    : list of column names the RF was trained on

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    row = {col: user_input_rf.get(col, 0) for col in rf_columns}
    return pd.DataFrame([row]).values


# ── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, columns,
                   model_path="models/knn_model.pkl",
                   scaler_path="models/scaler.pkl",
                   columns_path="models/columns.pkl"):
    """Persist the KNN model, scaler, and column list to disk."""
    with open(model_path,   "wb") as f: pickle.dump(model,   f)
    with open(scaler_path,  "wb") as f: pickle.dump(scaler,  f)
    with open(columns_path, "wb") as f: pickle.dump(columns, f)
    print(f"Saved → {model_path}, {scaler_path}, {columns_path}")


def load_artifacts(model_path="models/knn_model.pkl",
                   scaler_path="models/scaler.pkl",
                   columns_path="models/columns.pkl"):
    """Load and return (knn_model, scaler, columns) from disk."""
    with open(model_path,   "rb") as f: model   = pickle.load(f)
    with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    with open(columns_path, "rb") as f: columns = pickle.load(f)
    return model, scaler, columns


def save_rf_artifacts(rf_model, rf_columns,
                      rf_path="models/rf_model.pkl",
                      rf_cols_path="models/rf_columns.pkl"):
    """Persist the Random Forest model and its feature columns."""
    with open(rf_path,      "wb") as f: pickle.dump(rf_model,   f)
    with open(rf_cols_path, "wb") as f: pickle.dump(rf_columns, f)
    print(f"Saved → {rf_path}, {rf_cols_path}")


def load_rf_artifacts(rf_path="models/rf_model.pkl",
                      rf_cols_path="models/rf_columns.pkl"):
    """Load and return (rf_model, rf_columns) from disk."""
    with open(rf_path,      "rb") as f: rf_model   = pickle.load(f)
    with open(rf_cols_path, "rb") as f: rf_columns = pickle.load(f)
    return rf_model, rf_columns