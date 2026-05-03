"""
utils.py
--------
Shared utility functions for the Car Recommendation System.

v2 additions:
  - confidence_bar() — HTML progress-bar for confidence indicator
  - feature_match_html() — "Why this car?" explainer bar chart
  - persona_label() — maps KNN results to user persona
  - price_delta_str() — cleaner price delta formatting
"""

import pandas as pd


# ── Brand logo map ───────────────────────────────────────────────────────────

BRAND_LOGOS: dict[str, str] = {
    "Maruti":       "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Suzuki_logo_2.svg/120px-Suzuki_logo_2.svg.png",
    "Hyundai":      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Hyundai_Motor_Company_logo.svg/120px-Hyundai_Motor_Company_logo.svg.png",
    "Honda":        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Honda_Logo.svg/120px-Honda_Logo.svg.png",
    "Toyota":       "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Toyota_carlogo.svg/120px-Toyota_carlogo.svg.png",
    "Ford":         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Ford_logo_flat.svg/120px-Ford_logo_flat.svg.png",
    "Tata":         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Tata_logo.svg/120px-Tata_logo.svg.png",
    "Mahindra":     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Mahindra_Rise_Logo.svg/120px-Mahindra_Rise_Logo.svg.png",
    "Volkswagen":   "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Volkswagen_logo_2019.svg/120px-Volkswagen_logo_2019.svg.png",
    "Audi":         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Audi-Logo_2016.svg/120px-Audi-Logo_2016.svg.png",
    "Bmw":          "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BMW.svg/120px-BMW.svg.png",
    "Mercedesbenz": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Mercedes-Logo.svg/120px-Mercedes-Logo.svg.png",
    "Renault":      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Renault_2021_Text.svg/120px-Renault_2021_Text.svg.png",
    "Nissan":       "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Nissan_logo.svg/120px-Nissan_logo.svg.png",
    "Skoda":        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Skoda_Auto_2011_logo.svg/120px-Skoda_Auto_2011_logo.svg.png",
    "Chevrolet":    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Chevrolet_logo.svg/120px-Chevrolet_logo.svg.png",
    "Jeep":         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Jeep_wordmark.svg/120px-Jeep_wordmark.svg.png",
    "Fiat":         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Fiat_Logo.svg/120px-Fiat_Logo.svg.png",
    "Datsun":       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Datsun_logo.svg/120px-Datsun_logo.svg.png",
    "Volvo":        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Volvo_logo.svg/120px-Volvo_logo.svg.png",
    "Jaguar":       "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Jaguar_Cars_logo.svg/120px-Jaguar_Cars_logo.svg.png",
    "Land":         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Land_Rover_logo.svg/120px-Land_Rover_logo.svg.png",
    "Mitsubishi":   "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Mitsubishi_logo.svg/120px-Mitsubishi_logo.svg.png",
    "Porsche":      "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Porsche_logo.svg/120px-Porsche_logo.svg.png",
}

PLACEHOLDER_LOGO = "https://via.placeholder.com/120x60?text=Car"


def get_brand_logo(manufacturer: str) -> str:
    return BRAND_LOGOS.get(manufacturer, PLACEHOLDER_LOGO)


# ── Formatting helpers ───────────────────────────────────────────────────────

def format_price(price_lakhs: float) -> str:
    if price_lakhs >= 100:
        return f"₹{price_lakhs:.2f} Lakhs (1 Cr+)"
    return f"₹{price_lakhs:.2f} Lakhs"


def price_delta_str(actual: float, budget: float) -> tuple[str, str]:
    """Return (text, colour) for the price delta badge."""
    delta = actual - budget
    if delta > 0:
        return f"+₹{delta:.2f}L over budget", "#e05252"
    return f"₹{abs(delta):.2f}L under budget", "#1db954"


def fuel_icon(fuel_type: str) -> str:
    icons = {"Petrol": "⛽", "Diesel": "🛢️", "CNG": "🌿", "Electric": "⚡"}
    return icons.get(fuel_type, "🔋")


def transmission_icon(transmission: str) -> str:
    return "🔄" if transmission == "Automatic" else "🕹️"


# ── Confidence indicator ─────────────────────────────────────────────────────

def confidence_bar_html(confidence: float) -> str:
    """
    Return an HTML snippet with a colour-coded progress bar for confidence.
    confidence: 0–100 float.
    """
    color = "#1db954" if confidence >= 70 else ("#f0a500" if confidence >= 40 else "#e05252")
    label = "High" if confidence >= 70 else ("Medium" if confidence >= 40 else "Low")
    return (
        f'<div style="margin:4px 0;">'
        f'  <span style="font-size:0.8em;color:#aaa;">Confidence: <b style="color:{color};">'
        f'{label} ({confidence:.0f}%)</b></span>'
        f'  <div style="background:#2d2d44;border-radius:6px;height:6px;margin-top:3px;">'
        f'    <div style="background:{color};width:{confidence:.0f}%;height:6px;'
        f'border-radius:6px;"></div>'
        f'  </div>'
        f'</div>'
    )


# ── Feature match breakdown ──────────────────────────────────────────────────

def feature_match_html(row: "pd.Series") -> str:
    """
    Build a small HTML table showing the top match features from a result row.
    Expects columns: match_Price, match_Engine, match_Mileage, match_Seats
    """
    features = {
        "💰 Price":   row.get("match_Price",   0),
        "🔧 Engine":  row.get("match_Engine",  0),
        "⛽ Mileage": row.get("match_Mileage", 0),
        "💺 Seats":   row.get("match_Seats",   0),
    }
    bars = ""
    for label, pct in features.items():
        color = "#1db954" if pct >= 70 else ("#f0a500" if pct >= 40 else "#e05252")
        bars += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
            f'  <span style="font-size:0.75em;color:#ccc;min-width:80px;">{label}</span>'
            f'  <div style="flex:1;background:#2d2d44;border-radius:4px;height:5px;">'
            f'    <div style="background:{color};width:{pct:.0f}%;height:5px;border-radius:4px;"></div>'
            f'  </div>'
            f'  <span style="font-size:0.7em;color:{color};">{pct:.0f}%</span>'
            f'</div>'
        )
    return (
        f'<details style="margin-top:6px;">'
        f'  <summary style="font-size:0.8em;color:#888;cursor:pointer;">🔍 Why this car?</summary>'
        f'  <div style="padding:6px 0;">{bars}</div>'
        f'</details>'
    )


# ── Persona labelling ────────────────────────────────────────────────────────

def persona_label(price: float, seats: int, engine_cc: int) -> tuple[str, str]:
    """
    Assign a buyer persona label and emoji based on the car's specs.
    Returns (emoji, label).
    """
    if price <= 6 and engine_cc <= 1200:
        return "💸", "Budget Commuter"
    if seats >= 7 or engine_cc >= 2000:
        return "👨‍👩‍👧‍👦", "Family SUV Seeker"
    if engine_cc >= 1600 and price >= 15:
        return "🏎️", "Performance Enthusiast"
    if price <= 10:
        return "🏙️", "City Driver"
    return "🚗", "All-Rounder"


# ── Dataset summary ──────────────────────────────────────────────────────────

def dataset_summary(df: "pd.DataFrame") -> dict:
    return {
        "Total Cars":    len(df),
        "Brands":        df["Manufacturer"].nunique(),
        "Fuel Types":    df["Fuel_Type"].nunique(),
        "Price Range":   f"₹{df['Price'].min():.2f}L – ₹{df['Price'].max():.0f}L",
        "Mileage Range": f"{df['Mileage(Km/L)'].min():.1f} – {df['Mileage(Km/L)'].max():.1f} km/l",
        "Engine Range":  f"{df['Engine(CC)'].min()} – {df['Engine(CC)'].max()} CC",
    }