"""
ADVANCED DATA LOADER WITH HYBRID FEATURE ENGINEERING
====================================================
- Distance calculation
- Temperature normalization
- Climate categorization
- TF-IDF vectorization (content-based layer)
"""

import pandas as pd
import numpy as np
import math
import ast
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = "data/worldwide_travel_cities.csv"

START_CITY_COORDS = {
    "belgrade": (44.7866, 20.4489)
}


# -------------------------------------------------
# Distance calculation
# -------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2 +
        math.cos(phi1) * math.cos(phi2) *
        math.sin(d_lambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -------------------------------------------------
# Main loader
# -------------------------------------------------
def load_destinations(start_city: str = "belgrade"):
    df = pd.read_csv(DATA_PATH)

    # Normalize columns
    df.columns = df.columns.str.lower()
    df["city"] = df["city"].str.lower()
    df["region"] = df["region"].str.lower().str.strip()

    # ----------------------------
    # DISTANCE
    # ----------------------------
    start_lat, start_lon = START_CITY_COORDS[start_city]

    df["distance_km"] = df.apply(
        lambda r: haversine_distance(
            start_lat, start_lon,
            r["latitude"], r["longitude"]
        ),
        axis=1
    )

    # Normalize distance
    min_d, max_d = df["distance_km"].min(), df["distance_km"].max()
    df["distance_norm"] = (df["distance_km"] - min_d) / (max_d - min_d)

    # ----------------------------
    # TEMPERATURE
    # ----------------------------
    def avg_summer_temp(temp_dict_str):
        temps = ast.literal_eval(temp_dict_str)
        summer = [6, 7, 8]
        return sum(temps[str(m)]["avg"] for m in summer) / 3

    df["avg_temp"] = df["avg_temp_monthly"].apply(avg_summer_temp)

    min_t, max_t = df["avg_temp"].min(), df["avg_temp"].max()
    df["avg_temp_norm"] = (df["avg_temp"] - min_t) / (max_t - min_t)

    # Climate category
    def climate_label(t):
        if t < 0.33:
            return "cold"
        elif t < 0.66:
            return "moderate"
        return "warm"

    df["climate_category"] = df["avg_temp_norm"].apply(climate_label)

    # ----------------------------
    # TEXT PROFILE CREATION
    # ----------------------------
    def create_text_profile(row):
        parts = [
            str(row["short_description"]),
            row["city"],
            row["country"],
            row["region"]
        ]

        # Attribute boosting
        attribute_map = {
            "beaches": "beach seaside coastal",
            "culture": "culture history heritage museums",
            "nightlife": "nightlife clubs party entertainment",
            "nature": "nature outdoor hiking mountains",
            "cuisine": "food gastronomy culinary restaurants",
            "wellness": "spa wellness relaxation",
            "adventure": "adventure sports activities"
        }

        for attr, keywords in attribute_map.items():
            if attr in row and row[attr] >= 4:
                parts.append(keywords)

        return " ".join(parts)

    df["text_profile"] = df.apply(create_text_profile, axis=1)

    # ----------------------------
    # TF-IDF
    # ----------------------------
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(df["text_profile"])

    return df, vectorizer, tfidf_matrix
