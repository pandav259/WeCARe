"""
app.py — Car Recommendation System (Streamlit) v2
--------------------------------------------------
Run with:  streamlit run app/app.py

Changes from v1:
  - Removed Cosine Similarity toggle (cleaner UX)
  - Added Price Estimator panel (Random Forest prediction)
  - Added "Why this car?" feature-match breakdown
  - Added confidence indicator (based on KNN distance)
  - Added buyer persona badge per result card
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from preprocess import (load_and_clean, load_artifacts, load_rf_artifacts,
                        encode_user_input_rf, OWNER_TYPE_MAP,
                        FUEL_TYPE_MAP, TRANSMISSION_MAP)
from recommend  import recommend_knn, predict_price
from utils      import (get_brand_logo, format_price, price_delta_str,
                        fuel_icon, transmission_icon, dataset_summary,
                        confidence_bar_html, feature_match_html, persona_label)

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
    knn, scaler, columns = load_artifacts(
        model_path="models/knn_model.pkl",
        scaler_path="models/scaler.pkl",
        columns_path="models/columns.pkl",
    )
    rf_model, rf_columns = load_rf_artifacts(
        rf_path="models/rf_model.pkl",
        rf_cols_path="models/rf_columns.pkl",
    )
    return knn, scaler, columns, rf_model, rf_columns

df                              = load_data()
knn, scaler, columns, rf, rf_cols = load_models()

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

    # Engine CC — step 100
    engine_cc = st.slider(
        "🔧 Engine (CC)",
        int(df["Engine(CC)"].min()),
        int(df["Engine(CC)"].max()),
        1200,
        step=100,
    )

    # Mileage — integers only
    mileage = st.slider(
        "⛽ Min. Mileage (km/l)",
        int(df["Mileage(Km/L)"].min()),
        int(df["Mileage(Km/L)"].max()),
        15,
    )

    seats = st.slider("💺 Seats", 2, 7, 5)

    # Budget — integers only
    price = st.slider(
        "💰 Budget (Lakhs)",
        int(df["Price"].min()),
        int(df["Price"].max()),
        5,
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rec, tab_price = st.tabs(["🎯 Recommendations", "💡 Price Estimator"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    if recommend_btn:

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
            results = recommend_knn(
                user_input, df, knn, scaler, columns, n=n_results
            )

        # ── Results header ────────────────────────────────────────────────────
        st.markdown(f"### 🎯 Top {len(results)} Recommended Cars")

        filter_parts = []
        if brand        != "Any": filter_parts.append(brand)
        if fuel         != "Any": filter_parts.append(fuel)
        if transmission != "Any": filter_parts.append(transmission)
        filter_str = " · ".join(filter_parts) if filter_parts else "All brands/fuels"

        st.info(
            f"**Method:** KNN &nbsp;|&nbsp; "
            f"**Filters:** {filter_str} &nbsp;|&nbsp; "
            f"**Budget:** ₹{price:.1f}L"
        )

        if results.empty:
            st.error("No cars found for this combination. Try relaxing your filters.")
            st.stop()

        # ── Result cards ──────────────────────────────────────────────────────
        for i, (_, row) in enumerate(results.iterrows()):
            logo_url   = get_brand_logo(row["Manufacturer"])
            delta_str, delta_color = price_delta_str(row["Price"], price)
            conf_html  = confidence_bar_html(row.get("confidence", 50))
            match_html = feature_match_html(row)
            p_emoji, p_label = persona_label(
                row["Price"], int(row["Seats"]), int(row["Engine(CC)"])
            )

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
                      #{i+1} &nbsp; {row['Name']}
                      &nbsp;
                      <span style="background:#2d2d60;color:#aaa;padding:2px 8px;
                            border-radius:10px;font-size:0.75em;font-weight:normal;">
                        {p_emoji} {p_label}
                      </span>
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
                    {conf_html}
                    {match_html}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Comparison table ──────────────────────────────────────────────────
        with st.expander("📊 Side-by-side Comparison Table"):
            display_cols = ["Name", "Manufacturer", "Fuel_Type", "Transmission",
                            "Engine(CC)", "Mileage(Km/L)", "Seats", "Price", "confidence"]
            st.dataframe(
                results[[c for c in display_cols if c in results.columns]]
                .reset_index(drop=True),
                use_container_width=True,
            )

    else:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;color:#666;">
                <div style="font-size:5em;">🚗</div>
                <h3>Set your preferences in the sidebar and hit <em>Find My Car</em></h3>
                <p>Our KNN model will match you with the best options from 5,000+ Indian cars.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Price Estimator
# ═══════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.markdown("### 💡 Fair Price Estimator")
    st.markdown(
        "Enter a car's specifications below and our **Random Forest** model will "
        "predict its fair market price."
    )

    col1, col2 = st.columns(2)

    with col1:
        pe_engine  = st.number_input("🔧 Engine (CC)", 600, 5000, 1200, step=100)
        pe_mileage = st.number_input("⛽ Mileage (km/l)", 5, 50, 18, step=1)
        pe_seats    = st.selectbox("💺 Seats", [2, 4, 5, 6, 7], index=2)
        pe_power    = st.number_input("⚡ Power (bhp)", 30.0, 600.0, 85.0, step=5.0)

    with col2:
        pe_age     = st.slider("📅 Car Age (years)", 0, 20, 3)
        pe_km      = st.number_input("🛣️ Kilometers Driven", 0, 500000, 50000, step=1000)
        pe_owner    = st.selectbox("👤 Owner Type", list(OWNER_TYPE_MAP.keys()))
        pe_fuel     = st.selectbox("⛽ Fuel Type", list(FUEL_TYPE_MAP.keys())[:4])
        pe_trans    = st.selectbox("🕹️ Transmission", list(TRANSMISSION_MAP.keys()))

    if st.button("🔮 Predict Price", use_container_width=True):
        usage_intensity = pe_km / max(pe_age, 1)

        user_rf = {
            "Engine(CC)":      pe_engine,
            "Mileage(Km/L)":   pe_mileage,
            "Seats":           pe_seats,
            "Power":           pe_power,
            "Car_Age":         pe_age,
            "Usage_Intensity": usage_intensity,
            "Owner_Type_Enc":  OWNER_TYPE_MAP.get(pe_owner, 1),
            "Fuel_Type_Enc":   FUEL_TYPE_MAP.get(pe_fuel, 0),
            "Transmission_Enc": TRANSMISSION_MAP.get(pe_trans, 0),
            "Location_Enc":    0,   # neutral location
        }

        predicted = predict_price(user_rf, rf, rf_cols)

        st.success(f"### 🏷️ Estimated Fair Price: **{format_price(predicted)}**")

        # Show feature importances
        if hasattr(rf, "feature_importances_"):
            import pandas as pd
            import matplotlib.pyplot as plt
            fi = pd.Series(rf.feature_importances_, index=rf_cols).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            fi.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title("Feature Importances — Random Forest Price Predictor")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown(
            f"""
            <div style="background:#1a1a2e;border:1px solid #2d2d44;
                        border-radius:12px;padding:16px 20px;margin-top:12px;">
              <p style="margin:4px 0;">🔧 <b>{pe_engine} CC</b> &nbsp;|&nbsp;
                ⛽ <b>{pe_mileage} km/l</b> &nbsp;|&nbsp;
                💺 <b>{pe_seats} seats</b></p>
              <p style="margin:4px 0;">📅 <b>{pe_age} year(s) old</b> &nbsp;|&nbsp;
                🛣️ <b>{pe_km:,} km driven</b> &nbsp;|&nbsp;
                👤 <b>{pe_owner} owner</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div style="text-align:center;padding:40px 20px;color:#555;">
                <div style="font-size:4em;">💡</div>
                <p>Fill in the car specs above and click <em>Predict Price</em>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
