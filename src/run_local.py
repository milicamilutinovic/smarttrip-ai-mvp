"""
SMARTTRIP FINAL LOCAL DEMO
============================
Complete AI pipeline:

1. TF-IDF vectorization
2. Content-based cosine similarity
3. Attribute-based scoring
4. Hybrid scoring
5. Budget + region + climate scoring
6. Fairness metrics
7. MMR diversification
"""

from backend.services.data_loader import load_destinations
from backend.services.aggregation import aggregate_group_preferences
from backend.services.recommender import (
    compute_content_based_scores,
    compute_interest_scores,
    compute_hybrid_interest_score
)
from backend.services.scoring import apply_scoring
from backend.services.diversification import (
    calculate_destination_similarity_matrix,
    diversify_recommendations,
    add_diversity_explanation
)


# -------------------------------------------------
# Pretty print helpers
# -------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)


def print_destination(row, rank):
    print(f"\n#{rank}. {row['city'].title()}, {row['country'].title()}")
    print(f"   Region: {row['region'].title()} | Budget: {row['budget_level']}")
    print(f"   Distance: {row['distance_km']:.0f} km")

    est_cost = row.get("estimated_cost", 0)
    print(f"   Estimated cost: €{est_cost:.0f}")
    print(f"   Avg temp: {row['avg_temp']:.1f}°C")

    print("\n   SCORES:")
    print(f"      Final: {row['final_score']:.3f}")
    print(f"      Hybrid: {row.get('hybrid_interest_score', 0):.3f}")
    print(f"      Interest: {row.get('interest_score', 0):.3f}")
    print(f"      Content: {row.get('content_similarity', 0):.3f}")
    print(f"      Budget: {row.get('budget_score', 0):.3f}")
    print(f"      Region: {row.get('region_score', 0):.3f}")
    print(f"      Climate: {row.get('climate_score', 0):.3f}")

    if "fairness_score" in row:
        print("\n   FAIRNESS:")
        print(f"      Fairness score: {row['fairness_score']:.3f}")
        print(f"      Avg satisfaction: {row['avg_satisfaction']:.3f}")
        print(f"      Least misery: {row['least_misery']:.3f}")
        print(f"      Variance: {row['satisfaction_variance']:.3f}")

    if "selection_reason" in row:
        print(f"\n   Selection reason: {row['selection_reason']}")

    print(f"\n   {row['short_description'][:120]}...\n")


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

def main():

    print_section("SMARTTRIP FINAL AI MVP DEMO")

    # -------------------------------------------------
    # 1. Load data (TF-IDF version)
    # -------------------------------------------------
    destinations, vectorizer, tfidf_matrix = load_destinations(
        start_city="belgrade"
    )

    print(f"\nLoaded {len(destinations)} destinations.")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # -------------------------------------------------
    # 2. Simulated group input
    # -------------------------------------------------
    group_input = {
        "start_city": "belgrade",
        "month": "june",
        "region": "europe",
        "group": [
            {
                "max_budget_eur": 1000,
                "max_days": 5,
                "activities": ["nightlife", "food", "walking"],
                "temperature": "warm"
            },
            {
                "max_budget_eur": 700,
                "max_days": 4,
                "activities": ["relax", "quiet", "nature"],
                "temperature": "neutral"
            },
            {
                "max_budget_eur": 1200,
                "max_days": 7,
                "activities": ["beach", "swimming", "food"],
                "temperature": "warm"
            }
        ]
    }

    # -------------------------------------------------
    # 3. Aggregate group preferences
    # -------------------------------------------------
    group_profile = aggregate_group_preferences(
        group_input,
        strategy="average_with_fairness"
    )

    print_section("GROUP PROFILE")
    for k, v in group_profile.items():
        if isinstance(v, dict):
            continue
        print(f"{k}: {v}")

    # -------------------------------------------------
    # 4. Content-based scoring
    # -------------------------------------------------
    destinations = compute_content_based_scores(
        destinations,
        group_profile,
        vectorizer,
        tfidf_matrix
    )

    # -------------------------------------------------
    # 5. Attribute scoring
    # -------------------------------------------------
    destinations = compute_interest_scores(
        destinations,
        group_profile
    )

    # -------------------------------------------------
    # 6. Hybrid scoring
    # -------------------------------------------------
    destinations = compute_hybrid_interest_score(
        destinations,
        alpha=0.6
    )

    # -------------------------------------------------
    # 7. Multi-criteria scoring + fairness
    # -------------------------------------------------
    destinations = apply_scoring(
        destinations,
        group_profile,
        group_input=group_input,
        use_fairness=True
    )

    # -------------------------------------------------
    # 8. Sort by final score
    # -------------------------------------------------
    sorted_destinations = destinations.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    # -------------------------------------------------
    # 9. MMR Diversification
    # -------------------------------------------------
    similarity_matrix = calculate_destination_similarity_matrix(
        tfidf_matrix
    )

    diversified = diversify_recommendations(
        sorted_destinations,
        similarity_matrix,
        top_k=10,
        diversity_level="balanced"
    )

    diversified = add_diversity_explanation(
        diversified,
        sorted_destinations
    )

    # -------------------------------------------------
    # 10. Print Results
    # -------------------------------------------------
    print_section("TOP 10 DIVERSIFIED RECOMMENDATIONS")

    for rank, (_, row) in enumerate(diversified.iterrows(), 1):
        print_destination(row, rank)

    print_section("TOP 5 WITHOUT DIVERSIFICATION")

    for rank, (_, row) in enumerate(sorted_destinations.head(5).iterrows(), 1):
        print(f"{rank}. {row['city'].title()} Score: {row['final_score']:.3f}")

    print_section("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
