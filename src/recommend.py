"""
recommend.py
------------
Core recommendation logic for the Car Recommendation System.

Implements two strategies as required by the project synopsis:
  1. KNN (K-Nearest Neighbours) — primary recommender
  2. Cosine Similarity          — secondary / comparison recommender

Both strategies apply hard categorical filters FIRST (brand, fuel,
transmission), then rank the remaining candidates numerically.
This fixes the bug where filter selections were being ignored.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
        if value and value not in ("Select Brand", "Select Fuel Type", "Select Transmission"):
            candidate = filtered[filtered[col] == value]
            if not candidate.empty:          # only apply if it leaves rows
                filtered = candidate

    return filtered.reset_index(drop=True)


# ── Strategy 1 : KNN ─────────────────────────────────────────────────────────

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
       *full* dataset to get neighbour indices.
    3. Intersect neighbour indices with the filtered subset so that
       only matching cars survive.
    4. Fall back to the top-n nearest neighbours from the full dataset
       if the intersection is empty (edge case).

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
    pd.DataFrame of top-n recommended cars
    """
    user_vec = encode_user_input(user_input, columns, scaler)

    # Get a generous pool of neighbours from the full model
    pool_size = min(len(df), max(50, n * 10))
    distances, indices = model.kneighbors(user_vec, n_neighbors=pool_size)
    neighbour_indices = indices[0]

    # Hard-filter the dataset
    filtered_df = _apply_hard_filters(df, user_input)
    filtered_idx = set(filtered_df.index)

    # Keep only neighbours that also pass the hard filters
    valid = [i for i in neighbour_indices if i in filtered_idx]

    if valid:
        results = df.iloc[valid[:n]]
    else:
        # Graceful fallback – relax filters, return closest matches
        results = df.iloc[neighbour_indices[:n]]

    results = results.drop_duplicates(subset=["Name"])
    return results.head(n).reset_index(drop=True)


# ── Strategy 2 : Cosine Similarity ───────────────────────────────────────────

def recommend_cosine(user_input: dict,
                     df: pd.DataFrame,
                     scaler,
                     columns,
                     n: int = 5) -> pd.DataFrame:
    """
    Recommend cars using Cosine Similarity on scaled numeric features.

    Steps
    -----
    1. Hard-filter the dataset by categorical preferences.
    2. Scale numeric features of the filtered subset.
    3. Compute cosine similarity between the user vector and each car.
    4. Return the top-n highest-similarity cars.

    Parameters
    ----------
    Same as recommend_knn (no `model` needed).

    Returns
    -------
    pd.DataFrame with an added `similarity_score` column (0–1).
    """
    filtered_df = _apply_hard_filters(df, user_input).copy()

    if filtered_df.empty:
        return filtered_df   # nothing to recommend

    # Build numeric matrix for the filtered subset
    num_matrix = filtered_df[NUMERIC_COLS].values.astype(float)

    # User numeric vector (same order as NUMERIC_COLS)
    user_vec = np.array([[
        user_input.get("Engine(CC)",    1000),
        user_input.get("Mileage(Km/L)", 20.0),
        user_input.get("Seats",         5),
        user_input.get("Price",         5.0),
    ]], dtype=float)

    # Scale together so the user point lives in the same space
    from sklearn.preprocessing import MinMaxScaler as _MMS
    _scaler = _MMS()
    combined = np.vstack([num_matrix, user_vec])
    combined_scaled = _scaler.fit_transform(combined)

    data_scaled = combined_scaled[:-1]
    user_scaled  = combined_scaled[-1:]

    sims = cosine_similarity(user_scaled, data_scaled)[0]   # shape (n_rows,)

    filtered_df = filtered_df.copy()
    filtered_df["similarity_score"] = (sims * 100).round(2)

    result = (filtered_df
              .drop_duplicates(subset=["Name"])
              .sort_values("similarity_score", ascending=False)
              .head(n)
              .reset_index(drop=True))

    return result
