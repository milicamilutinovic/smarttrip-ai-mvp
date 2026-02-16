import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# 1️ UČITAVANJE PODATAKA
# =============================

df = pd.read_csv("/Users/milicamilutinovic/mvpsmarttrip/data/worldwide_travel_cities.csv")

attributes = [
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

# =============================
# 2️ EMBEDDING MODEL
# =============================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 Jači konceptualni opisi atributa
attribute_concepts = [
    "culture, museums, art, history, architecture, heritage",
    "adventure, hiking, trekking, extreme sports, outdoor activities",
    "nature, mountains, forests, lakes, scenic landscapes",
    "beaches, sea, sand, tropical coast",
    "nightlife, clubs, bars, party, music",
    "cuisine, local food, restaurants, gastronomy",
    "wellness, spa, relaxation, retreat",
    "urban, city life, shopping, metropolitan atmosphere",
    "seclusion, remote, peaceful, isolated, quiet retreat"
]

# Embedding atributa
attribute_embeddings = model.encode(attribute_concepts)

# =============================
# 3️ TF-IDF MODEL
# =============================

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df["short_description"])

# =============================
# 4️ MAPIRANJE TEKSTA → 9 ATRIBUTA
# =============================

def text_to_attribute_vector(user_text):
    
    user_embedding = model.encode([user_text])
    
    similarities = cosine_similarity(
        user_embedding,
        attribute_embeddings
    )[0]
    
    # uklanjamo negativne vrednosti
    similarities = np.clip(similarities, 0, None)
    
    # skaliramo na 0–5
    if similarities.max() != 0:
        similarities = similarities / similarities.max() * 5
        
    return similarities

# =============================
# 5️ HIBRIDNI RECOMMENDER
# =============================

def recommend(user_text, top_k=5, alpha=0.6):
    
    # ---- ATTRIBUTE SCORE ----
    user_vector = text_to_attribute_vector(user_text)
    
    city_matrix = df[attributes].values
    
    attribute_scores = cosine_similarity(
        user_vector.reshape(1, -1),
        city_matrix
    )[0]
    
    # ---- TEXT SCORE ----
    user_tfidf = tfidf.transform([user_text])
    
    text_scores = cosine_similarity(
        user_tfidf,
        tfidf_matrix
    )[0]
    
    # ---- FINAL SCORE ----
    final_scores = alpha * attribute_scores + (1 - alpha) * text_scores
    
    df["final_score"] = final_scores
    
    results = df.sort_values("final_score", ascending=False).head(top_k)
    
    return results[["city", "country", "region", "final_score", "short_description"]]

# =============================
# 6️x DEMO
# =============================

if __name__ == "__main__":
    
    print("\n=== SmartTrip Full Hybrid AI Recommender ===\n")
    
    user_input = input("Describe your ideal trip (in English):\n\n")
    
    recommendations = recommend(user_input)
    
    print("\nTop Recommended Cities:\n")
    
    for _, row in recommendations.iterrows():
        print(f"🌍 {row['city']}, {row['country']} ({row['region']})")
        print(f"Score: {round(row['final_score'], 3)}")
        print(f"Description: {row['short_description']}")
        print("-" * 70)
