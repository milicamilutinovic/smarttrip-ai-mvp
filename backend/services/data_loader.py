"""
ADVANCED DATA LOADER WITH HYBRID FEATURE ENGINEERING
====================================================
- Distance calculation
- Temperature extraction for specific month
- Climate categorization
- TF-IDF vectorization (content-based layer)
"""

import pandas as pd
import numpy as np
import math
import ast
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = "data/worldwide_travel_cities.csv"


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth using Haversine formula.
    
    Returns: distance in kilometers
    """
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


def load_destinations(start_city_coords: tuple, month: int):
    """
    Load and prepare destination dataset.
    
    Args:
        start_city_coords: (latitude, longitude) tuple of starting city
        month: Month number (1-12) for temperature extraction
    
    Returns:
        df: DataFrame with processed destinations
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: TF-IDF matrix of text profiles
    """
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.lower()
    df["city"] = df["city"].str.lower()
    df["region"] = df["region"].str.lower().str.strip()

    # DISTANCE
    start_lat, start_lon = start_city_coords

    df["distance_km"] = df.apply(
        lambda r: haversine_distance(
            start_lat, start_lon,
            r["latitude"], r["longitude"]
        ),
        axis=1
    )

    min_d, max_d = df["distance_km"].min(), df["distance_km"].max()
    df["distance_norm"] = (df["distance_km"] - min_d) / (max_d - min_d)

    # TEMPERATURE - extract for specific month
    def get_temp_for_month(temp_dict_str, target_month):
        temps = ast.literal_eval(temp_dict_str)
        return temps[str(target_month)]["avg"]

    df["avg_temp"] = df["avg_temp_monthly"].apply(
        lambda x: get_temp_for_month(x, month)
    )

    min_t, max_t = df["avg_temp"].min(), df["avg_temp"].max()
    df["avg_temp_norm"] = (df["avg_temp"] - min_t) / (max_t - min_t)

    def climate_label(temp_celsius):
        if temp_celsius < 10:
            return "cold"
        elif temp_celsius < 23:
            return "moderate"
        return "warm"

    df["climate_category"] = df["avg_temp"].apply(climate_label)

    # TEXT PROFILE CREATION
    def create_text_profile(row):
        parts = [
            str(row["short_description"]),
            row["city"],
            row["country"],
            row["region"]
        ]

        attribute_map = {
            "beaches": "beach seaside coastal tropical sand ocean",
            "culture": "culture history heritage museums architecture art gallery",
            "nightlife": "nightlife clubs bars party entertainment dancing music",
            "nature": "nature outdoor hiking mountains scenery forests lakes",
            "cuisine": "food gastronomy culinary restaurants dining local cuisine",
            "wellness": "spa wellness relaxation retreat massage sauna jacuzzi",
            "adventure": "adventure sports activities adrenaline extreme climbing",
            "urban": "city shopping urban metropolitan downtown modern",
            "seclusion": "quiet peaceful remote secluded isolated tranquil"
        }

        for attr, keywords in attribute_map.items():
            if attr in row and row[attr] >= 4:
                parts.append(keywords)

        return " ".join(parts)

    df["text_profile"] = df.apply(create_text_profile, axis=1)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(df["text_profile"])

    return df, vectorizer, tfidf_matrix