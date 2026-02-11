"""
DIVERSIFICATION WITH MMR (Maximal Marginal Relevance)
=====================================================
Ensures top recommendations are not overly similar.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# Destination Similarity Matrix
# -------------------------------------------------
def calculate_destination_similarity_matrix(
    tfidf_matrix: np.ndarray
) -> np.ndarray:
    """
    Computes cosine similarity matrix between all destinations.
    """

    return cosine_similarity(tfidf_matrix, tfidf_matrix)


# -------------------------------------------------
# MMR Algorithm
# -------------------------------------------------
def maximal_marginal_relevance(
    destinations: pd.DataFrame,
    similarity_matrix: np.ndarray,
    k: int = 10,
    lambda_param: float = 0.5
) -> pd.DataFrame:
    """
    Greedy MMR selection.

    lambda_param:
        1.0 → only relevance
        0.0 → only diversity
    """

    if len(destinations) <= k:
        return destinations.head(k).copy()

    # IMPORTANT: Reset index to align with similarity matrix
    df = destinations.reset_index(drop=True)

    scores = df["final_score"].values

    # Normalize scores
    if scores.max() > 0:
        norm_scores = scores / scores.max()
    else:
        norm_scores = scores

    selected = [0]  # highest ranked first
    candidates = list(range(1, len(df)))

    for _ in range(k - 1):

        if not candidates:
            break

        mmr_values = []

        for idx in candidates:

            relevance = norm_scores[idx]

            max_sim = max(
                similarity_matrix[idx, sel_idx]
                for sel_idx in selected
            )

            mmr_score = (
                lambda_param * relevance -
                (1 - lambda_param) * max_sim
            )

            mmr_values.append((idx, mmr_score))

        best_idx = max(mmr_values, key=lambda x: x[1])[0]

        selected.append(best_idx)
        candidates.remove(best_idx)

    return df.iloc[selected].copy()


# -------------------------------------------------
# Diversification Wrapper
# -------------------------------------------------
def diversify_recommendations(
    destinations: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int = 10,
    diversity_level: str = "balanced"
) -> pd.DataFrame:

    lambda_map = {
        "high": 0.3,      # more diversity
        "balanced": 0.5,  # balanced
        "low": 0.7        # more relevance
    }

    lambda_param = lambda_map.get(diversity_level, 0.5)

    return maximal_marginal_relevance(
        destinations,
        similarity_matrix,
        k=top_k,
        lambda_param=lambda_param
    )


# -------------------------------------------------
# Explanation Layer
# -------------------------------------------------
def add_diversity_explanation(
    diversified_df: pd.DataFrame,
    original_df: pd.DataFrame
) -> pd.DataFrame:

    df = diversified_df.copy()

    # Map original rank positions
    original_df = original_df.reset_index(drop=True)

    original_rank_map = {
        city: idx + 1
        for idx, city in enumerate(original_df["city"])
    }

    df["original_rank"] = df["city"].map(original_rank_map)
    df["rank_shift"] = df["original_rank"] - (df.index + 1)

    def categorize(row):
        if row["rank_shift"] <= 2:
            return "High relevance"
        elif row["rank_shift"] <= 5:
            return "Balanced pick"
        return "Diversity pick"

    df["selection_reason"] = df.apply(categorize, axis=1)

    return df
