"""
utils.py
--------
Shared utility functions for the Car Recommendation System.
"""

import pandas as pd


# ── Car image helper ─────────────────────────────────────────────────────────

# Curated Wikimedia / public-domain logo URLs per brand.
# Falls back to a generic placeholder if the brand isn't listed.
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
    """Return a logo URL for the given manufacturer name."""
    return BRAND_LOGOS.get(manufacturer, PLACEHOLDER_LOGO)


# ── Formatting helpers ───────────────────────────────────────────────────────

def format_price(price_lakhs: float) -> str:
    """Return a human-readable price string."""
    if price_lakhs >= 100:
        return f"₹{price_lakhs:.2f} Lakhs (1 Cr+)"
    return f"₹{price_lakhs:.2f} Lakhs"


def fuel_icon(fuel_type: str) -> str:
    icons = {"Petrol": "⛽", "Diesel": "🛢️", "CNG": "🌿", "Electric": "⚡"}
    return icons.get(fuel_type, "🔋")


def transmission_icon(transmission: str) -> str:
    return "🔄" if transmission == "Automatic" else "🕹️"


# ── Dataset summary ──────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for display in the sidebar or About page."""
    return {
        "Total Cars":     len(df),
        "Brands":         df["Manufacturer"].nunique(),
        "Fuel Types":     df["Fuel_Type"].nunique(),
        "Price Range":    f"₹{df['Price'].min():.2f}L – ₹{df['Price'].max():.0f}L",
        "Mileage Range":  f"{df['Mileage(Km/L)'].min():.1f} – {df['Mileage(Km/L)'].max():.1f} km/l",
        "Engine Range":   f"{df['Engine(CC)'].min()} – {df['Engine(CC)'].max()} CC",
    }
