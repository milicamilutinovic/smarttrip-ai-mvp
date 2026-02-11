"""
ADVANCED SCORING WITH FAIRNESS
==============================
Multi-criteria scoring with:
- Smoothed budget feasibility
- Region preference
- Climate match
- Individual satisfaction tracking
- Fairness metrics (least misery + variance penalty)
"""

import pandas as pd
import numpy as np
from backend.services.cost_model import estimate_total_trip_cost


# -------------------------------------------------
# Temperature mapping
# -------------------------------------------------
TEMP_MAP = {
    "cold": 0.0,
    "neutral": 0.5,
    "warm": 1.0
}


# -------------------------------------------------
# Smoothed Budget Score (uses precomputed cost)
# -------------------------------------------------
def budget_score_smooth(row, group_budget: float) -> float:

    estimated_cost = row["estimated_cost"]

    ratio = estimated_cost / group_budget if group_budget > 0 else 10

    if ratio <= 1.0:
        return 1.0
    else:
        k = 3.0
        return float(np.exp(-k * (ratio - 1.0)))


# -------------------------------------------------
# Region Score
# -------------------------------------------------
def region_score(destination_region: str, group_region: str) -> float:
    return 1.0 if destination_region == group_region else 0.4


# -------------------------------------------------
# Climate Score
# -------------------------------------------------
def climate_score(avg_temp: float, preferred_temp: float) -> float:
    return 1.0 - abs(avg_temp - preferred_temp)


# -------------------------------------------------
# Individual Satisfaction (uses precomputed cost)
# -------------------------------------------------
def calculate_individual_scores(
    df: pd.DataFrame,
    group_input: dict,
    group_profile: dict
) -> pd.DataFrame:

    df = df.copy()
    group = group_input["group"]

    for person_idx, person in enumerate(group):

        def person_score(row):
            score = 0.0

            # Temperature match
            person_temp = TEMP_MAP[person["temperature"]]
            temp_match = 1.0 - abs(row["avg_temp_norm"] - person_temp)
            score += 0.3 * temp_match

            # Activity match
            base_interest = row.get(
                "hybrid_interest_score",
                row.get("interest_score", 0)
            )
            score += 0.5 * base_interest

            # Budget constraint (uses already computed cost)
            estimated_cost = row["estimated_cost"]
            person_budget = person["max_budget_eur"]

            if estimated_cost <= person_budget:
                score += 0.2
            elif person_budget <= 0:
                penalty = 0.0
            else:
                penalty = max(
                    0,
                    1 - (estimated_cost - person_budget) / person_budget
                )
                score += 0.2 * penalty

            return float(score)

        df[f"person_{person_idx}_score"] = df.apply(person_score, axis=1)

    return df


# -------------------------------------------------
# Fairness Metrics
# -------------------------------------------------
def calculate_fairness_metrics(
    df: pd.DataFrame,
    group_size: int
) -> pd.DataFrame:

    df = df.copy()

    person_cols = [f"person_{i}_score" for i in range(group_size)]

    df["least_misery"] = df[person_cols].min(axis=1)
    df["avg_satisfaction"] = df[person_cols].mean(axis=1)
    df["satisfaction_variance"] = df[person_cols].var(axis=1, ddof=0)

    df["fairness_score"] = (
        0.5 * df["avg_satisfaction"] +
        0.4 * df["least_misery"] -
        0.1 * df["satisfaction_variance"]
    )

    return df


# -------------------------------------------------
# MAIN SCORING FUNCTION
# -------------------------------------------------
def apply_scoring(
    destinations: pd.DataFrame,
    group_profile: dict,
    group_input: dict = None,
    use_fairness: bool = True
) -> pd.DataFrame:

    df = destinations.copy()

    # -------------------------------------------------
    # Compute cost ONCE
    # -------------------------------------------------
    df["estimated_cost"] = df.apply(
        lambda row: estimate_total_trip_cost(
            distance_km=row["distance_km"],
            budget_level=row["budget_level"],
            days=group_profile["trip_days"]
        ),
        axis=1
    )

    # -------------------------------------------------
    # Budget
    # -------------------------------------------------
    df["budget_score"] = df.apply(
        lambda row: budget_score_smooth(
            row,
            group_profile["group_budget"]
        ),
        axis=1
    )

    # -------------------------------------------------
    # Region
    # -------------------------------------------------
    df["region_score"] = df["region"].apply(
        lambda r: region_score(r, group_profile["region"])
    )

    # -------------------------------------------------
    # Climate
    # -------------------------------------------------
    df["climate_score"] = df["avg_temp_norm"].apply(
        lambda t: climate_score(
            t,
            group_profile["preferred_temperature"]
        )
    )

    # -------------------------------------------------
    # WITH FAIRNESS
    # -------------------------------------------------
    if use_fairness and group_input is not None:

        df = calculate_individual_scores(df, group_input, group_profile)
        df = calculate_fairness_metrics(df, group_profile["group_size"])

        df["final_score"] = (
            0.35 * df.get("hybrid_interest_score", df.get("interest_score", 0)) +
            0.25 * df["budget_score"] +
            0.10 * df["region_score"] +
            0.10 * df["climate_score"] +
            0.20 * df["fairness_score"]
        )

    # -------------------------------------------------
    # WITHOUT FAIRNESS
    # -------------------------------------------------
    else:

        df["final_score"] = (
            0.45 * df.get("hybrid_interest_score", df.get("interest_score", 0)) +
            0.30 * df["budget_score"] +
            0.15 * df["region_score"] +
            0.10 * df["climate_score"]
        )
    
    #elimination of edge-case problems
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df
