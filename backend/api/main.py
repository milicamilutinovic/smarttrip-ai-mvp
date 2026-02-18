from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.api.schemas import GroupRequest
from backend.services.data_loader import load_destinations
from backend.services.aggregation import aggregate_group_preferences
from backend.services.recommender import compute_hybrid_text_score
from backend.services.scoring import apply_scoring
from backend.services.diversification import (
    calculate_destination_similarity_matrix,
    diversify_recommendations
)
from backend.services.geocoding import get_coordinates


app = FastAPI(title="SmartTrip AI API")

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(Path("frontend/index.html"))


@app.post("/recommend")
def recommend(group_input: GroupRequest):

    group_dict = group_input.dict()

    # 1️ Get coordinates
    coords = get_coordinates(group_dict["start_city"])

    if coords is None:
        return {"recommendations": [], "error": "Invalid or unknown start city"}

    # 2️ Load destinations
    df, vectorizer, tfidf_matrix = load_destinations(
        start_city_coords=coords,
        month=group_dict["month"],
        region=group_dict["region"]
    )

    if df is None:
        return {"recommendations": [], "error": "No destinations found in this region"}

    # 3️ Aggregate
    group_profile = aggregate_group_preferences(
        group_dict,
        strategy="average_with_fairness"
    )

    # 4️ Text similarity
    df = compute_hybrid_text_score(
        df,
        group_profile["group_description"],
        vectorizer,
        tfidf_matrix[df.index],
        beta=0.6,
        alpha=0.7
    )

    # 5️ Multi scoring
    df = apply_scoring(
        df,
        group_profile,
        group_input=group_dict,
        use_fairness=True
    )

    print("\n===== DEBUG TOP 5 BEFORE SORT =====")

    cols_to_show = [
        "city",
        "hybrid_text_score",
        "budget_score",
        "temp_score",
        "duration_score",
        "estimated_cost",
        "distance_km"
    ]

    if "fairness_score" in df.columns:
        cols_to_show.append("fairness_score")

    print(df[cols_to_show].head(5))

    # 6️ Sort
    sorted_df = df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    print("\n===== DEBUG TOP 5 AFTER SORT =====")
    print(sorted_df[cols_to_show].head(5))

    # 7️ Diversify
    similarity_matrix = calculate_destination_similarity_matrix(
        tfidf_matrix[df.index]
    )

    top_results = diversify_recommendations(
        sorted_df,
        similarity_matrix,
        top_k=10,
        diversity_level="balanced"
    )

    # 8️ Response
    response = []

    for _, row in top_results.iterrows():
        response.append({
            "city": row["city"].title(),
            "country": row["country"],
            "final_score": float(row["final_score"]),
            "estimated_cost": float(row["estimated_cost"]),
            "avg_temp": float(row["avg_temp"]),
            "description": row["short_description"]
        })
        print("\n--- DESTINATION ANALYSIS ---")
        print("City:", row["city"])
        print("Final score:", row["final_score"])
        print("Text score:", row.get("hybrid_text_score"))
        print("Budget score:", row.get("budget_score"))
        print("Temp score:", row.get("temp_score"))
        print("Duration score:", row.get("duration_score"))
        print("Fairness score:", row.get("fairness_score"))
        print("Estimated cost:", row["estimated_cost"])
        print("Distance:", row["distance_km"])


    return {"recommendations": response}


@app.get("/health")
def health():
    return {"status": "ok"}
