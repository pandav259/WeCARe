"""
recommend.py
------------
Core recommendation logic for the Car Recommendation System.

v2 changes:
  - Removed Cosine Similarity (cleaner UX, as per plan)
  - Removed K-Means (removed as requested)
  - Added KNN confidence score based on normalised neighbour distance
  - Added feature match breakdown for "Why this car?" explainer
  - Kept hard categorical filter logic intact
"""

import numpy as np
import pandas as pd

from preprocess import CATEGORICAL_COLS, NUMERIC_COLS, encode_user_input


# ── Helper ───────────────────────────────────────────────────────────────────

def _apply_hard_filters(df: pd.DataFrame, user_input: dict) -> pd.DataFrame:
    """
    Return only the rows that match the user's categorical selections.
    If a filter would produce an empty result, it is relaxed gracefully.
    """
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
    """
    Convert a KNN Euclidean distance into a 0–100% confidence score.
    Closer neighbours → higher confidence.
    """
    confidence = max(0.0, 1.0 - distance / max_dist) * 100
    return round(confidence, 1)


def _feature_match_breakdown(row: pd.Series,
                              user_input: dict) -> dict:
    """
    Compute per-feature relative closeness (0–100%) for the "Why this car?"
    explainer shown in the Streamlit UI.

    Returns a dict: {feature_label: match_pct}
    """
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


# ── KNN Recommender ───────────────────────────────────────────────────────────

def recommend_knn(user_input: dict,
                  df: pd.DataFrame,
                  model,
                  scaler,
                  columns,
                  n: int = 5) -> pd.DataFrame:
    """
    Recommend cars using a pre-trained KNN model.

    Steps
    -----
    1. Hard-filter the dataset by categorical preferences.
    2. Encode the user vector and query the KNN index built on the
       full dataset to get neighbour indices + distances.
    3. Intersect neighbour indices with the filtered subset so that
       only matching cars survive.
    4. Attach confidence score and feature-match breakdown to each result.
    5. Fall back to top-n nearest neighbours from the full dataset if
       the intersection is empty (edge case).

    Parameters
    ----------
    user_input : dict  – keys: Manufacturer, Fuel_Type, Transmission,
                         Engine(CC), Mileage(Km/L), Seats, Price
    df         : cleaned DataFrame (same index used when model was fitted)
    model      : fitted NearestNeighbors
    scaler     : fitted MinMaxScaler
    columns    : pd.Index of training columns
    n          : number of recommendations

    Returns
    -------
    pd.DataFrame of top-n recommended cars, with added columns:
        knn_distance  – raw Euclidean distance
        confidence    – 0-100 confidence score
        match_Price / match_Engine / match_Mileage / match_Seats
    """
    user_vec = encode_user_input(user_input, columns, scaler)

    # Get a generous pool of neighbours from the full model
    pool_size = min(len(df), max(50, n * 10))
    distances, indices = model.kneighbors(user_vec, n_neighbors=pool_size)
    neighbour_indices  = indices[0]
    neighbour_dists    = distances[0]

    # Hard-filter the dataset
    filtered_df  = _apply_hard_filters(df, user_input)
    filtered_idx = set(filtered_df.index)

    # Keep only neighbours that pass the hard filters
    valid_pairs = [(i, d) for i, d in zip(neighbour_indices, neighbour_dists)
                   if i in filtered_idx]

    if valid_pairs:
        valid_idx   = [p[0] for p in valid_pairs[:n]]
        valid_dists = [p[1] for p in valid_pairs[:n]]
    else:
        # Graceful fallback – relax filters
        valid_idx   = list(neighbour_indices[:n])
        valid_dists = list(neighbour_dists[:n])

    results = df.iloc[valid_idx].copy()
    results["knn_distance"] = valid_dists
    results["confidence"]   = results["knn_distance"].apply(
        _confidence_from_distance
    )

    # Per-row feature match breakdown
    breakdowns = results.apply(
        lambda row: _feature_match_breakdown(row, user_input), axis=1
    )
    breakdown_df = pd.DataFrame(list(breakdowns))
    breakdown_df.columns = [f"match_{c}" for c in breakdown_df.columns]
    results = pd.concat([results.reset_index(drop=True), breakdown_df], axis=1)

    results = results.drop_duplicates(subset=["Name"])
    return results.head(n).reset_index(drop=True)


# ── Price Prediction Helper ───────────────────────────────────────────────────

def predict_price(user_specs: dict,
                  rf_model,
                  rf_columns) -> float:
    """
    Use the trained Random Forest to estimate a fair market price
    for a car described by user_specs.

    Parameters
    ----------
    user_specs : dict with keys matching RF_FEATURE_COLS
    rf_model   : trained RandomForestRegressor
    rf_columns : list of column names the RF was trained on

    Returns
    -------
    Predicted price in Lakhs (float)
    """
    row = {col: user_specs.get(col, 0) for col in rf_columns}
    X   = pd.DataFrame([row])
    return float(rf_model.predict(X)[0])