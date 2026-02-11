# backend/api/main.py
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI
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
    diversify_recommendations
)
from backend.api.schemas import GroupRequest


app = FastAPI(title="SmartTrip AI API")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(Path("frontend/index.html"))

# -------------------------------------------------
# Load data once at startup (IMPORTANT)
# -------------------------------------------------
destinations, vectorizer, tfidf_matrix = load_destinations(
    start_city="belgrade"
)


@app.get("/")
def root():
    return {"message": "SmartTrip AI API is running"}


@app.post("/recommend")
def recommend(group_input: GroupRequest):

    group_dict = group_input.dict()

    # 1️ Aggregate
    group_profile = aggregate_group_preferences(
        group_dict,
        strategy="average_with_fairness"
    )

    df = destinations.copy()

    # 2️ Content scoring
    df = compute_content_based_scores(
        df,
        group_profile,
        vectorizer,
        tfidf_matrix
    )

    # 3️ Attribute scoring
    df = compute_interest_scores(df, group_profile)

    # 4️ Hybrid
    df = compute_hybrid_interest_score(df, alpha=0.6)

    # 5️ Final scoring
    df = apply_scoring(
        df,
        group_profile,
        group_input=group_dict,
        use_fairness=True
    )

    # 6️ Sort
    sorted_df = df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    # 7️ Diversify
    similarity_matrix = calculate_destination_similarity_matrix(
        tfidf_matrix
    )

    top_results = diversify_recommendations(
        sorted_df,
        similarity_matrix,
        top_k=10
    )

    # 8️ Prepare response
    response = []

    for _, row in top_results.iterrows():
        response.append({
            "city": row["city"],
            "country": row["country"],
            "final_score": float(row["final_score"]),
            "estimated_cost": float(row["estimated_cost"]),
            "avg_temp": float(row["avg_temp"]),
            "description": row["short_description"]
        })

    return {"recommendations": response}
