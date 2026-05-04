import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


RENAME_MAP = {
    "Engine CC":    "Engine(CC)",
    "Mileage Km/L": "Mileage(Km/L)",
}

FEATURE_COLS = [
    "Manufacturer", "Fuel_Type", "Transmission",
    "Engine(CC)", "Mileage(Km/L)", "Seats", "Price",
]

RF_FEATURE_COLS = [
    "Engine(CC)", "Mileage(Km/L)", "Seats", "Power",
    "Car_Age", "Usage_Intensity", "Owner_Type_Enc", "Location_Enc",
    "Fuel_Type_Enc", "Transmission_Enc",
]

CATEGORICAL_COLS = ["Manufacturer", "Fuel_Type", "Transmission"]
NUMERIC_COLS     = ["Engine(CC)", "Mileage(Km/L)", "Seats", "Price"]

OWNER_TYPE_MAP = {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4}
FUEL_TYPE_MAP  = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df.rename(columns=RENAME_MAP)

    REFERENCE_YEAR = 2024

    if "Year" in df.columns:
        df["Car_Age"] = REFERENCE_YEAR - df["Year"]
        df["Car_Age"] = df["Car_Age"].clip(lower=1)  

    if "Kilometers_Driven" in df.columns and "Car_Age" in df.columns:
        df["Usage_Intensity"] = df["Kilometers_Driven"] / df["Car_Age"]

    if "Owner_Type" in df.columns:
        df["Owner_Type_Enc"] = df["Owner_Type"].map(OWNER_TYPE_MAP).fillna(3)

    df["Fuel_Type_Enc"]     = df["Fuel_Type"].map(FUEL_TYPE_MAP).fillna(0)
    df["Transmission_Enc"]  = df["Transmission"].map(TRANSMISSION_MAP).fillna(0)

    if "Location" in df.columns:
        locs = sorted(df["Location"].dropna().unique())
        loc_map = {loc: i for i, loc in enumerate(locs)}
        df["Location_Enc"] = df["Location"].map(loc_map).fillna(0)
    else:
        df["Location_Enc"] = 0

    df = df[df["Fuel_Type"] != "LPG"]

    df["Manufacturer"] = df["Manufacturer"].str.title()

    core_cols = ["Name", "Manufacturer", "Fuel_Type", "Transmission",
                 "Engine(CC)", "Mileage(Km/L)", "Seats", "Price"]
    df = df.dropna(subset=core_cols)

    df = df[df["Seats"] <= 7]

    if "Power" in df.columns:
        df["Power"] = pd.to_numeric(df["Power"], errors="coerce")
        df["Power"].fillna(df["Power"].median(), inplace=True)
    else:
        df["Power"] = 80.0 

    df = df.reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, X.columns


def build_rf_feature_matrix(df: pd.DataFrame):
    available = [c for c in RF_FEATURE_COLS if c in df.columns]
    X = df[available].copy().fillna(0)
    y = df["Price"]
    return X, y


def encode_user_input(user_input: dict, columns, scaler) -> "np.ndarray":
    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=columns, fill_value=0)
    return scaler.transform(user_df)


def encode_user_input_rf(user_input_rf: dict, rf_columns) -> "np.ndarray":
    row = {col: user_input_rf.get(col, 0) for col in rf_columns}
    return pd.DataFrame([row]).values


def save_artifacts(model, scaler, columns,
                   model_path="models/knn_model.pkl",
                   scaler_path="models/scaler.pkl",
                   columns_path="models/columns.pkl"):
    with open(model_path,   "wb") as f: pickle.dump(model,   f)
    with open(scaler_path,  "wb") as f: pickle.dump(scaler,  f)
    with open(columns_path, "wb") as f: pickle.dump(columns, f)
    print(f"Saved → {model_path}, {scaler_path}, {columns_path}")


def load_artifacts(model_path="models/knn_model.pkl",
                   scaler_path="models/scaler.pkl",
                   columns_path="models/columns.pkl"):
    with open(model_path,   "rb") as f: model   = pickle.load(f)
    with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    with open(columns_path, "rb") as f: columns = pickle.load(f)
    return model, scaler, columns


def save_rf_artifacts(rf_model, rf_columns,
                      rf_path="models/rf_model.pkl",
                      rf_cols_path="models/rf_columns.pkl"):
    with open(rf_path,      "wb") as f: pickle.dump(rf_model,   f)
    with open(rf_cols_path, "wb") as f: pickle.dump(rf_columns, f)
    print(f"Saved → {rf_path}, {rf_cols_path}")


def load_rf_artifacts(rf_path="models/rf_model.pkl",
                      rf_cols_path="models/rf_columns.pkl"):
    with open(rf_path,      "rb") as f: rf_model   = pickle.load(f)
    with open(rf_cols_path, "rb") as f: rf_columns = pickle.load(f)
    return rf_model, rf_columns