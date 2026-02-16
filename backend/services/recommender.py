"""
HYBRID RECOMMENDER
==================
Combines three scoring methods:
1. Intent extraction → Attribute matching (structured)
2. TF-IDF semantic similarity (unstructured)
3. Weighted hybrid combination
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict

try:
    from intent_extraction import extract_intent_vector, intent_vector_to_array, DATASET_COLUMNS
except ImportError:
    DATASET_COLUMNS = [
        "culture", "adventure", "nature", "beaches",
        "nightlife", "cuisine", "wellness", "urban", "seclusion"
    ]
    def extract_intent_vector(text, attributes=None, alpha=0.7):
        if attributes is None:
            attributes = DATASET_COLUMNS
        return {attr: 0.0 for attr in attributes}
    def intent_vector_to_array(intent_dict, attributes):
        return np.array([intent_dict.get(attr, 0.0) for attr in attributes])


def compute_user_intent_vector(user_text: str, alpha: float = 0.7) -> Dict[str, float]:
    """
    Extract intent vector from user text using hybrid approach.
    
    Args:
        user_text: User description
        alpha: Weight for lexicon vs embeddings (default: 0.7)
    
    Returns:
        Intent vector dict
    """
    return extract_intent_vector(user_text, DATASET_COLUMNS, alpha=alpha)


def compute_attribute_scores(
    destinations: pd.DataFrame,
    user_intent: Dict[str, float]
) -> pd.DataFrame:
    """
    Score destinations based on attribute matching.
    
    Formula: score = Σ (user_intent[attr] * city[attr] / 5.0)
    """
    df = destinations.copy()
    
    def calculate_score(row):
        score = 0.0
        for attr in DATASET_COLUMNS:
            if attr in row and attr in user_intent:
                city_value = row[attr] / 5.0
                score += user_intent[attr] * city_value
        return float(score)
    
    df["attribute_score"] = df.apply(calculate_score, axis=1)
    
    return df


def compute_tfidf_similarity(
    destinations: pd.DataFrame,
    user_text: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix
) -> pd.DataFrame:
    """
    Compute semantic similarity using TF-IDF.
    """
    df = destinations.copy()
    
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]
    
    df["tfidf_similarity"] = similarities
    
    return df


def compute_hybrid_text_score(
    destinations: pd.DataFrame,
    user_text: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    beta: float = 0.6,
    alpha: float = 0.7
) -> pd.DataFrame:
    """
    Hybrid text scoring combining intent extraction and TF-IDF.
    
    Args:
        destinations: DataFrame with city data
        user_text: User description
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: TF-IDF matrix
        beta: Weight for attribute vs TF-IDF
              beta=1.0 → only attributes
              beta=0.0 → only TF-IDF
        alpha: Weight for lexicon vs embeddings in intent extraction
    
    Returns:
        DataFrame with hybrid_text_score column
    """
    df = destinations.copy()
    
    user_intent = compute_user_intent_vector(user_text, alpha=alpha)
    
    df = compute_attribute_scores(df, user_intent)
    df = compute_tfidf_similarity(df, user_text, vectorizer, tfidf_matrix)
    
    norm_attr = df["attribute_score"] / df["attribute_score"].max() if df["attribute_score"].max() > 0 else df["attribute_score"]
    norm_tfidf = df["tfidf_similarity"] / df["tfidf_similarity"].max() if df["tfidf_similarity"].max() > 0 else df["tfidf_similarity"]
    
    df["hybrid_text_score"] = beta * norm_attr + (1 - beta) * norm_tfidf
    
    df["user_intent"] = str(user_intent)
    
    return df


def compute_content_based_scores(
    destinations: pd.DataFrame,
    group_profile: dict,
    vectorizer: TfidfVectorizer,
    tfidf_matrix
) -> pd.DataFrame:
    """
    Backward compatibility function.
    """
    if "group_description" in group_profile:
        return compute_tfidf_similarity(
            destinations,
            group_profile["group_description"],
            vectorizer,
            tfidf_matrix
        )
    return destinations


def compute_interest_scores(
    destinations: pd.DataFrame,
    group_profile: dict
) -> pd.DataFrame:
    """
    Backward compatibility function.
    """
    if "group_description" in group_profile:
        user_intent = compute_user_intent_vector(group_profile["group_description"])
        return compute_attribute_scores(destinations, user_intent)
    return destinations


def compute_hybrid_interest_score(
    destinations: pd.DataFrame,
    alpha: float = 0.6
) -> pd.DataFrame:
    """
    Backward compatibility function.
    """
    df = destinations.copy()
    
    if "interest_score" in df.columns:
        norm_interest = df["interest_score"] / df["interest_score"].max() if df["interest_score"].max() > 0 else df["interest_score"]
    else:
        norm_interest = pd.Series(0.0, index=df.index)
    
    if "content_similarity" in df.columns:
        norm_content = df["content_similarity"] / df["content_similarity"].max() if df["content_similarity"].max() > 0 else df["content_similarity"]
    else:
        norm_content = pd.Series(0.0, index=df.index)
    
    df["hybrid_interest_score"] = alpha * norm_interest + (1 - alpha) * norm_content
    
    return df