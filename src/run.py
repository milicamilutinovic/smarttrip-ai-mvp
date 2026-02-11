import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. LOAD DATASETS
# --------------------------------------------------

destinations = pd.read_csv(
    "/Users/milicamilutinovic/mvpsmarttrip/data/travel_destinations.csv"
)
fuzzy = pd.read_csv(
    "/Users/milicamilutinovic/mvpsmarttrip/data/tourism_fuzzy_dataset.csv"
)
cities = pd.read_csv(
    "/Users/milicamilutinovic/mvpsmarttrip/data/worldwide_travel_cities.csv"
)

# --------------------------------------------------
# 2. CLEAN & NORMALIZE COLUMNS
# --------------------------------------------------

# ---- travel_destinations.csv ----
destinations = destinations.rename(columns={
    "City": "city",
    "Category": "text",
    "Best_Time_to_Travel": "best_season"
})

destinations["city"] = destinations["city"].str.lower()
destinations["text"] = destinations["text"].str.lower()
destinations["best_season"] = destinations["best_season"].fillna("all")

# ---- fuzzy dataset ----
fuzzy.columns = fuzzy.columns.str.lower()

# if these columns don't exist, create them (MVP-safe)
for col in ["budget_level", "family_friendly", "group_activities"]:
    if col not in fuzzy.columns:
        fuzzy[col] = np.random.choice(
            ["low", "medium", "high"] if col == "budget_level" else [0, 1],
            size=len(fuzzy)
        )

# ---- cities dataset ----
cities = cities.rename(columns={
    "City": "city"
})
cities["city"] = cities["city"].str.lower()

# simulate rating if missing
if "avg_rating" not in cities.columns:
    cities["avg_rating"] = np.random.uniform(3.5, 4.8, size=len(cities))

# --------------------------------------------------
# 3. FEATURE ENRICHMENT
# --------------------------------------------------

destinations = destinations.merge(
    cities[["city", "avg_rating"]],
    on="city",
    how="left"
)

destinations["avg_rating"] = destinations["avg_rating"].fillna(
    destinations["avg_rating"].mean()
)

# --------------------------------------------------
# 4. TEXT VECTORIZATION (AI CORE)
# --------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=300
)

tfidf_matrix = tfidf.fit_transform(destinations["text"])

# --------------------------------------------------
# 5. SCORING FUNCTIONS
# --------------------------------------------------

def budget_score(dest_budget, user_budget):
    if dest_budget == user_budget:
        return 1.0
    if user_budget == "low" and dest_budget != "low":
        return 0.3
    return 0.7


def group_score(row, group_type):
    if group_type == "students":
        return 1.0
    if group_type == "family":
        return 0.8
    if group_type == "team":
        return 0.9
    return 0.7


def season_score(best_season, user_season):
    if best_season == "all":
        return 1.0
    return 1.0 if user_season.lower() in best_season.lower() else 0.5

# --------------------------------------------------
# 6. LOCAL SCENARIO-BASED TESTING
# --------------------------------------------------

test_users = [
    {"interests": "budget city culture", "budget": "low", "group_type": "students", "season": "summer"},
    {"interests": "kids nature safety", "budget": "medium", "group_type": "family", "season": "spring"},
    {"interests": "team activities conference", "budget": "high", "group_type": "team", "season": "autumn"}
]

print("\nSMARTTRIP – LOCAL SCENARIO TESTING")
print("=" * 50)

for i, user in enumerate(test_users, 1):

    print(f"\nTEST USER {i}")
    print(f"Interests: {user['interests']} | Budget: {user['budget']} | Group: {user['group_type']}")

    user_vec = tfidf.transform([user["interests"]])
    destinations["text_similarity"] = cosine_similarity(
        user_vec, tfidf_matrix
    )[0]

    destinations["budget_score"] = destinations.apply(
        lambda r: budget_score("medium", user["budget"]),
        axis=1
    )

    destinations["group_score"] = destinations.apply(
        lambda r: group_score(r, user["group_type"]),
        axis=1
    )

    destinations["season_score"] = destinations["best_season"].apply(
        lambda s: season_score(s, user["season"])
    )

    destinations["final_score"] = (
        0.5 * destinations["text_similarity"] +
        0.2 * destinations["budget_score"] +
        0.2 * destinations["group_score"] +
        0.1 * destinations["season_score"]
    )

    top = destinations.sort_values(
        by="final_score",
        ascending=False
    ).head(5)

    print("\nTop 5 recommendations:")
    print(top[["city", "final_score", "best_season"]].to_string(index=False))

