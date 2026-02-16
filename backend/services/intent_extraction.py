"""
HYBRID INTENT EXTRACTION
=========================
Combines lexicon-based keyword matching with sentence embeddings
for robust intent extraction from user descriptions.

Approach:
1. Lexicon matching (fast, accurate for known words)
2. Embedding similarity (handles unknown words and synonyms)
3. Weighted combination (70% lexicon, 30% embeddings)
"""

import re
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer


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


INTENT_LEXICON = {
    "culture": [
        "museum", "museums", "gallery", "galleries",
        "history", "historic", "historical", "heritage",
        "art", "arts", "artwork", "paintings",
        "culture", "cultural", "architecture", "architectural",
        "cathedral", "cathedrals", "church", "churches",
        "monuments", "monument", "landmarks",
        "tradition", "traditional", "castle", "palace",
        "opera", "theatre", "theater", "library"
    ],
    
    "adventure": [
        "adventure", "adventurous", "adrenaline",
        "rafting", "kayaking", "climbing", "climb",
        "extreme sports", "extreme", "thrill",
        "skiing", "ski", "snowboarding",
        "surfing", "surf", "diving", "scuba",
        "trekking", "trek", "paragliding",
        "zip line", "zipline", "bungee"
    ],
    
    "nature": [
        "nature", "natural", "hiking", "hike",
        "mountain", "mountains", "trails", "trail",
        "lake", "lakes", "rivers", "waterfalls",
        "scenery", "scenic", "views", "landscapes",
        "forest", "forests", "woods",
        "park", "parks", "national park",
        "wildlife", "countryside", "outdoor"
    ],
    
    "beaches": [
        "beach", "beaches", "sea", "ocean",
        "swimming", "swim", "coast", "coastal",
        "tropical", "island", "islands",
        "sand", "sandy", "seaside",
        "sunbathing", "snorkeling",
        "bay", "bays", "cove"
    ],
    
    "nightlife": [
        "party", "partying", "parties",
        "club", "clubs", "clubbing", "nightclub",
        "bar", "bars", "pub", "pubs",
        "nightlife", "night life",
        "dance", "dancing",
        "entertainment", "live music",
        "cocktails", "drinks"
    ],
    
    "cuisine": [
        "food", "foods", "eat", "eating",
        "cuisine", "culinary", "gastronomy",
        "restaurant", "restaurants", "dining",
        "wine", "wines", "local food",
        "street food", "cooking",
        "taste", "tasting", "delicious",
        "pizza", "pasta", "coffee", "cafe"
    ],
    
    "wellness": [
        "spa", "spas", "wellness",
        "relax", "relaxation", "relaxing",
        "retreat", "massage", "massages",
        "yoga", "meditation",
        "peaceful", "tranquility",
        "sauna", "jacuzzi", "hot tub",
        "thermal", "hot springs"
    ],
    
    "urban": [
        "city", "cities", "city life",
        "shopping", "shop", "shops",
        "metropolitan", "downtown",
        "urban", "modern",
        "cosmopolitan", "vibrant",
        "streets", "walking", "strolling"
    ],
    
    "seclusion": [
        "quiet", "silence",
        "remote", "remoteness",
        "secluded", "seclusion",
        "peaceful", "peace", "serene",
        "isolated", "isolation",
        "off the beaten path",
        "tranquil", "calm",
        "hidden", "escape"
    ]
}


SEED_WORDS = {
    "culture": ["museum", "history", "art", "architecture"],
    "adventure": ["climbing", "rafting", "skiing", "extreme"],
    "nature": ["hiking", "mountains", "forest", "lake"],
    "beaches": ["beach", "sea", "swimming", "coast"],
    "nightlife": ["party", "club", "bar", "dance"],
    "cuisine": ["food", "restaurant", "dining", "cuisine"],
    "wellness": ["spa", "massage", "relax", "yoga"],
    "urban": ["city", "shopping", "downtown", "urban"],
    "seclusion": ["quiet", "peaceful", "remote", "isolated"]
}


_model = None
_seed_embeddings = None


def get_model():
    """Lazy load sentence transformer model"""
    global _model, _seed_embeddings
    
    if _model is None:
        print("Loading sentence embedding model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        
        _seed_embeddings = {}
        for cat, seeds in SEED_WORDS.items():
            embeddings = _model.encode(seeds)
            _seed_embeddings[cat] = np.mean(embeddings, axis=0)
        
        print("Model loaded.")
    
    return _model, _seed_embeddings


def preprocess_text(text: str) -> str:
    """Clean and normalize user input"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\',.-]', '', text)
    return text.strip()


def lexicon_based_matching(user_text: str) -> Dict[str, float]:
    """Extract intent using lexicon keyword matching"""
    text = preprocess_text(user_text)
    tokens = text.split()
    
    scores = {attr: 0.0 for attr in DATASET_COLUMNS}
    
    for attr in DATASET_COLUMNS:
        keywords = INTENT_LEXICON[attr]
        
        for keyword in keywords:
            if keyword in text:
                scores[attr] += 1.0
    
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    
    return scores


def embedding_based_matching(user_text: str) -> Dict[str, float]:
    """Extract intent using sentence embeddings"""
    if not user_text or not user_text.strip():
        return {attr: 0.0 for attr in DATASET_COLUMNS}
    
    model, seed_embeddings = get_model()
    
    text_embedding = model.encode(user_text)
    
    scores = {}
    for cat, seed_emb in seed_embeddings.items():
        similarity = np.dot(text_embedding, seed_emb) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(seed_emb)
        )
        scores[cat] = max(0, float(similarity))
    
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    
    return scores


def extract_intent_vector(
    user_text: str,
    attributes: List[str] = None,
    alpha: float = 0.7
) -> Dict[str, float]:
    """
    Hybrid intent extraction: lexicon + embeddings
    
    Args:
        user_text: User description
        attributes: List of attributes (default: DATASET_COLUMNS)
        alpha: Weight for lexicon (default: 0.7)
               alpha=1.0 → only lexicon
               alpha=0.0 → only embeddings
    
    Returns:
        Dict mapping attribute → normalized score (0-1)
    """
    if attributes is None:
        attributes = DATASET_COLUMNS
    
    if not user_text or not user_text.strip():
        return {attr: 0.0 for attr in attributes}
    
    lexicon_scores = lexicon_based_matching(user_text)
    embedding_scores = embedding_based_matching(user_text)
    
    final_scores = {}
    for attr in attributes:
        final_scores[attr] = (
            alpha * lexicon_scores.get(attr, 0) +
            (1 - alpha) * embedding_scores.get(attr, 0)
        )
    
    total = sum(final_scores.values())
    if total > 0:
        final_scores = {k: v / total for k, v in final_scores.items()}
    
    return final_scores


def intent_vector_to_array(intent_dict: Dict[str, float], attributes: List[str]) -> np.ndarray:
    """Convert intent dictionary to numpy array"""
    return np.array([intent_dict.get(attr, 0.0) for attr in attributes])


if __name__ == "__main__":
    test_cases = [
        "I want museums and good food",
        "Looking for beaches and relaxation with spa",
        "Adventure hiking and climbing in mountains",
        "Quiet peaceful retreat with nature",
        "City shopping and nightlife with bars",
        "I want partying, skiing, museums, good pizza and coffee"
    ]
    
    print("=" * 70)
    print("HYBRID INTENT EXTRACTION TEST")
    print("=" * 70)
    
    for text in test_cases:
        print(f"\nUser: '{text}'")
        print("-" * 70)
        
        intent = extract_intent_vector(text, alpha=0.7)
        
        sorted_intent = sorted(intent.items(), key=lambda x: x[1], reverse=True)
        
        print("Top intents:")
        for attr, score in sorted_intent[:3]:
            if score > 0:
                print(f"  {attr:12s}: {score:.3f}")