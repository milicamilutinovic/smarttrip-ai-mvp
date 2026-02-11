"""
CONTENT-BASED + HYBRID RECOMMENDER (TF-IDF)
============================================
Semantic matching using TF-IDF + cosine similarity
combined with structured attribute scoring.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------------------------
# Mapping user activities to dataset attributes
# -------------------------------------------------

ACTIVITY_TO_DATASET = {
    # nightlife & social
    "nightlife": ["nightlife", "urban"],
    "clubs": ["nightlife"],
    "party": ["nightlife"],

    # food
    "food": ["cuisine"],
    "cuisine": ["cuisine"],
    "restaurants": ["cuisine"],

    # beach & swimming
    "beach": ["beaches"],
    "swimming": ["beaches"],

    # nature & walking
    "nature": ["nature"],
    "walking": ["urban", "nature"],
    "hiking": ["nature", "adventure"],

    # relax & quiet
    "relax": ["wellness"],
    "quiet": ["seclusion"],

    # culture
    "culture": ["culture"],
    "history": ["culture"],
    "museums": ["culture"],

    # adventure
    "adventure": ["adventure"],
    "sports": ["adventure"],
}


DATASET_COLUMNS = [
    "culture",
    "adventure",
    "nature",
    "beaches",
    "nightlife",
    "cuisine",
    "wellness",
    "urban",
    "seclusion"
]


# -------------------------------------------------
# Create user semantic query text
# -------------------------------------------------
def create_user_query_text(group_profile: dict) -> str:
    """
    Creates text query from group preferences
    for TF-IDF semantic matching.
    """

    parts = []

    # Weighted repetition of activities
    for activity, weight in group_profile["interests"].items():
        repetitions = max(1, int(weight * 10))
        parts.extend([activity] * repetitions)

    # Region
    parts.append(group_profile["region"])

    # Temperature keywords
    temp_pref = group_profile["preferred_temperature"]

    if temp_pref < 0.33:
        parts.extend(["cold", "winter", "snow"] * 2)
    elif temp_pref > 0.66:
        parts.extend(["warm", "sunny", "summer", "hot"] * 2)
    else:
        parts.extend(["mild", "moderate", "spring"] * 2)

    return " ".join(parts)


# -------------------------------------------------
# TF-IDF Content Similarity
# -------------------------------------------------
def compute_content_based_scores(
    destinations: pd.DataFrame,
    group_profile: dict,
    vectorizer: TfidfVectorizer,
    tfidf_matrix
) -> pd.DataFrame:
    """
    Compute cosine similarity between user query and destinations.
    """

    df = destinations.copy()

    user_query = create_user_query_text(group_profile)

    query_vector = vectorizer.transform([user_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

    df["content_similarity"] = similarities

    return df


# -------------------------------------------------
# Attribute-based Interest Score
# -------------------------------------------------
def compute_interest_scores(
    destinations: pd.DataFrame,
    group_profile: dict
) -> pd.DataFrame:

    df = destinations.copy()
    group_interests = group_profile["interests"]

    def interest_score(row):
        score = 0.0

        for activity, weight in group_interests.items():
            mapped_features = ACTIVITY_TO_DATASET.get(activity, [])

            for feature in mapped_features:
                if feature in DATASET_COLUMNS and feature in row:
                    score += weight * (row[feature] / 5.0)

        return score

    df["interest_score"] = df.apply(interest_score, axis=1)

    return df


# -------------------------------------------------
# Hybrid Combination
# -------------------------------------------------
def compute_hybrid_interest_score(
    destinations: pd.DataFrame,
    alpha: float = 0.6
) -> pd.DataFrame:
    """
    Combine attribute-based and content-based scores.

    alpha → weight for structured attributes
    (1 - alpha) → weight for semantic similarity
    """

    df = destinations.copy()

    # Safe normalization
    if df["interest_score"].max() > 0:
        norm_interest = df["interest_score"] / df["interest_score"].max()
    else:
        norm_interest = df["interest_score"]

    if df["content_similarity"].max() > 0:
        norm_content = df["content_similarity"] / df["content_similarity"].max()
    else:
        norm_content = df["content_similarity"]

    df["hybrid_interest_score"] = (
        alpha * norm_interest +
        (1 - alpha) * norm_content
    )

    return df
