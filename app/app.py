"""
app.py — Car Recommendation System (Streamlit)
----------------------------------------------
Run with:  streamlit run app/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from preprocess  import load_and_clean, load_artifacts
from recommend   import recommend_knn, recommend_cosine
from utils       import (get_brand_logo, format_price,
                          fuel_icon, transmission_icon, dataset_summary)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Recommendation System",
    page_icon="🚗",
    layout="wide",
)

# ── Load data & artefacts (cached) ───────────────────────────────────────────
@st.cache_data
def load_data():
    return load_and_clean("data/cars.csv")

@st.cache_resource
def load_models():
    return load_artifacts(
        model_path="models/knn_model.pkl",
        scaler_path="models/scaler.pkl",
        columns_path="models/columns.pkl",
    )

df              = load_data()
model, scaler, columns = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Your Preferences")
    st.markdown("---")

    brand = st.selectbox(
        "🏷️ Brand",
        ["Any"] + sorted(df["Manufacturer"].unique()),
    )

    fuel = st.selectbox(
        "⛽ Fuel Type",
        ["Any"] + sorted(df["Fuel_Type"].unique()),
    )

    transmission = st.selectbox(
        "🕹️ Transmission",
        ["Any"] + sorted(df["Transmission"].unique()),
    )

    engine_cc = st.slider(
        "🔧 Engine (CC)",
        int(df["Engine(CC)"].min()),
        int(df["Engine(CC)"].max()),
        1200,
    )

    mileage = st.slider(
        "⛽ Min. Mileage (km/l)",
        float(df["Mileage(Km/L)"].min()),
        float(df["Mileage(Km/L)"].max()),
        15.0,
    )

    seats = st.slider("💺 Seats", 2, 7, 5)

    price = st.slider(
        "💰 Budget (Lakhs)",
        float(df["Price"].min()),
        float(df["Price"].max()),
        5.0,
    )

    method = st.radio(
        "🧠 Recommendation Method",
        ["KNN (K-Nearest Neighbours)", "Cosine Similarity"],
        help="KNN finds cars closest to your input in feature space. "
             "Cosine Similarity measures directional similarity on numeric specs.",
    )

    n_results = st.slider("📋 Number of Results", 3, 10, 5)

    st.markdown("---")
    recommend_btn = st.button("🚀 Find My Car", use_container_width=True)

    # Dataset stats
    st.markdown("---")
    st.markdown("**📊 Dataset Stats**")
    for k, v in dataset_summary(df).items():
        st.markdown(f"- **{k}:** {v}")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Smart Car Recommender")
st.markdown("#### Find your perfect car using Machine Learning")
st.markdown("---")

# ── Recommendation ────────────────────────────────────────────────────────────
if recommend_btn:

    # Build user-input dict (map "Any" → None so filters are relaxed)
    user_input = {
        "Manufacturer":  brand        if brand        != "Any" else None,
        "Fuel_Type":     fuel         if fuel         != "Any" else None,
        "Transmission":  transmission if transmission != "Any" else None,
        "Engine(CC)":    engine_cc,
        "Mileage(Km/L)": mileage,
        "Seats":         seats,
        "Price":         price,
    }

    with st.spinner("🔍 Finding the best cars for you…"):
        if method.startswith("KNN"):
            results = recommend_knn(
                user_input, df, model, scaler, columns, n=n_results
            )
            score_col = None
        else:
            results = recommend_cosine(
                user_input, df, scaler, columns, n=n_results
            )
            score_col = "similarity_score"

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown(f"### 🎯 Top {len(results)} Recommended Cars")

    method_label = "KNN" if method.startswith("KNN") else "Cosine Similarity"
    filter_parts = []
    if brand        != "Any": filter_parts.append(brand)
    if fuel         != "Any": filter_parts.append(fuel)
    if transmission != "Any": filter_parts.append(transmission)
    filter_str = " · ".join(filter_parts) if filter_parts else "All brands/fuels"

    st.info(
        f"**Method:** {method_label} &nbsp;|&nbsp; "
        f"**Filters:** {filter_str} &nbsp;|&nbsp; "
        f"**Budget:** ₹{price:.1f}L"
    )

    if results.empty:
        st.error("No cars found for this combination. Try relaxing your filters.")
        st.stop()

    # Cards
    for i, (_, row) in enumerate(results.iterrows()):
        logo_url = get_brand_logo(row["Manufacturer"])

        score_badge = ""
        if score_col and score_col in row:
            score_badge = (
                f'<span style="background:#1db954;color:#fff;'
                f'padding:2px 8px;border-radius:12px;font-size:0.8em;">'
                f'Match: {row[score_col]:.1f}%</span>'
            )

        price_delta = row["Price"] - price
        delta_str   = (f'+₹{price_delta:.2f}L over budget'
                       if price_delta > 0
                       else f'₹{abs(price_delta):.2f}L under budget')
        delta_color = "#e05252" if price_delta > 0 else "#1db954"

        st.markdown(
            f"""
            <div style="
                background:#1a1a2e;
                border:1px solid #2d2d44;
                border-radius:14px;
                padding:18px 22px;
                margin-bottom:16px;
                display:flex;
                gap:20px;
                align-items:flex-start;
            ">
              <div style="min-width:60px;text-align:center;padding-top:4px;">
                <span style="font-size:2em;">🚗</span><br/>
                <img src="{logo_url}" height="30"
                     style="object-fit:contain;margin-top:6px;"
                     onerror="this.style.display='none'"/>
              </div>
              <div style="flex:1;">
                <h4 style="margin:0 0 6px 0;color:#e0e0ff;">
                  #{i+1} &nbsp; {row['Name']} &nbsp; {score_badge}
                </h4>
                <p style="margin:2px 0;">
                  💰 <b>{format_price(row['Price'])}</b>
                  &nbsp; <span style="color:{delta_color};font-size:0.85em;">({delta_str})</span>
                </p>
                <p style="margin:2px 0;">
                  {fuel_icon(row['Fuel_Type'])} {row['Fuel_Type']} &nbsp;|&nbsp;
                  {transmission_icon(row['Transmission'])} {row['Transmission']}
                </p>
                <p style="margin:2px 0;">
                  🔧 {int(row['Engine(CC)'])} CC &nbsp;|&nbsp;
                  ⛽ {row['Mileage(Km/L)']:.1f} km/l &nbsp;|&nbsp;
                  💺 {int(row['Seats'])} seats
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    with st.expander("📊 Side-by-side Comparison Table"):
        display_cols = ["Name", "Manufacturer", "Fuel_Type", "Transmission",
                        "Engine(CC)", "Mileage(Km/L)", "Seats", "Price"]
        if score_col and score_col in results.columns:
            display_cols.append(score_col)
        st.dataframe(
            results[display_cols].reset_index(drop=True),
            use_container_width=True,
        )

else:
    # Landing state
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#666;">
            <div style="font-size:5em;">🚗</div>
            <h3>Set your preferences in the sidebar and hit <em>Find My Car</em></h3>
            <p>Our ML model will match you with the best options from 5,000+ Indian cars.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    