import numpy as np
import pandas as pd

from preprocess import CATEGORICAL_COLS, NUMERIC_COLS, encode_user_input



def _apply_hard_filters(df: pd.DataFrame, user_input: dict) -> pd.DataFrame:
    filtered = df.copy()

    for col, key in [
        ("Manufacturer", "Manufacturer"),
        ("Fuel_Type",    "Fuel_Type"),
        ("Transmission", "Transmission"),
    ]:
        value = user_input.get(key)
        if value and value not in ("Select Brand", "Select Fuel Type",
                                   "Select Transmission", "Any", None):
            candidate = filtered[filtered[col] == value]
            if not candidate.empty:
                filtered = candidate

    return filtered.reset_index(drop=True)


def _confidence_from_distance(distance: float,
                               max_dist: float = 3.0) -> float:
    confidence = max(0.0, 1.0 - distance / max_dist) * 100
    return round(confidence, 1)


def _feature_match_breakdown(row: pd.Series, user_input: dict) -> dict:
    checks = {
        "Price":    ("Price",         user_input.get("Price",         5.0),   30.0),
        "Engine":   ("Engine(CC)",    user_input.get("Engine(CC)",    1200),   3000),
        "Mileage":  ("Mileage(Km/L)", user_input.get("Mileage(Km/L)", 18.0),  40.0),
        "Seats":    ("Seats",         user_input.get("Seats",         5),      5.0),
    }
    breakdown = {}
    for label, (col, target, scale) in checks.items():
        diff = abs(float(row[col]) - float(target))
        score = max(0.0, 100.0 - (diff / scale) * 100)
        breakdown[label] = round(min(score, 100.0), 1)
    return breakdown


def recommend_knn(user_input: dict,
                  df: pd.DataFrame,
                  model,
                  scaler,
                  columns,
                  n: int = 5) -> pd.DataFrame:

    user_vec = encode_user_input(user_input, columns, scaler)

    pool_size = min(len(df), max(50, n * 10))
    distances, indices = model.kneighbors(user_vec, n_neighbors=pool_size)
    neighbour_indices  = indices[0]
    neighbour_dists    = distances[0]

    filtered_df  = _apply_hard_filters(df, user_input)
    filtered_idx = set(filtered_df.index)

    valid_pairs = [(i, d) for i, d in zip(neighbour_indices, neighbour_dists)
                   if i in filtered_idx]

    if valid_pairs:
        valid_idx   = [p[0] for p in valid_pairs[:n]]
        valid_dists = [p[1] for p in valid_pairs[:n]]
    else:
        valid_idx   = list(neighbour_indices[:n])
        valid_dists = list(neighbour_dists[:n])

    results = df.iloc[valid_idx].copy()
    results["knn_distance"] = valid_dists
    results["confidence"]   = results["knn_distance"].apply(
        _confidence_from_distance
    )

    breakdowns = results.apply(
        lambda row: _feature_match_breakdown(row, user_input), axis=1
    )
    breakdown_df = pd.DataFrame(list(breakdowns))
    breakdown_df.columns = [f"match_{c}" for c in breakdown_df.columns]
    results = pd.concat([results.reset_index(drop=True), breakdown_df], axis=1)

    results = results.drop_duplicates(subset=["Name"])
    return results.head(n).reset_index(drop=True)


def predict_price(user_specs: dict,
                  rf_model,
                  rf_columns) -> float:
    row = {col: user_specs.get(col, 0) for col in rf_columns}
    X   = pd.DataFrame([row])
    return float(rf_model.predict(X)[0])