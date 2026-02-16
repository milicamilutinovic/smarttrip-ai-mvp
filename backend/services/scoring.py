"""
ADVANCED SCORING WITH FAIRNESS
==============================
Multi-criteria scoring with:
- Hybrid text scoring (intent + TF-IDF)
- Duration matching
- Temperature matching (absolute Celsius)
- Budget feasibility (exponential penalty)
- Region preference
- Fairness metrics (least misery + variance)
"""

import pandas as pd
import numpy as np
import ast
from backend.services.cost_model import estimate_total_trip_cost as cost_func


TEMP_MAP = {
    "cold": 0.0,
    "neutral": 0.5,
    "warm": 1.0
}

TEMP_RANGES = {
    "cold": (0, 15),
    "neutral": (15, 25),
    "warm": (25, 40)
}

DURATION_DAYS = {
    "one day": 1,
    "weekend": 2.5,
    "short trip": 4,
    "one week": 7,
    "long trip": 12
}


#def estimate_total_trip_cost(distance_km: float, budget_level: str, days: int) -> float:
#    """Import from cost_model or define locally"""
#    from cost_model import estimate_total_trip_cost as cost_func
#    return cost_func(distance_km, budget_level, days)


def budget_score_smooth(row, group_budget: float) -> float:
    """Exponential penalty for budget excess"""
    estimated_cost = row["estimated_cost"]
    ratio = estimated_cost / group_budget if group_budget > 0 else 10
    
    if ratio <= 1.0:
        return 1.0
    else:
        k = 3.0
        return float(np.exp(-k * (ratio - 1.0)))

def budget_utilization_score(row, group_budget: float) -> float:
    """
    AI-style budget satisfaction:
    Ideal spending ~ 75% of available budget.
    
    Too cheap → low experience utilization
    Too expensive → penalty
    """

    estimated_cost = row["estimated_cost"]

    if group_budget <= 0:
        return 0.0

    utilization = estimated_cost / group_budget

    # Ideal utilization around 0.75
    ideal = 0.75
    sigma = 0.25  # controls tolerance

    score = np.exp(-((utilization - ideal) ** 2) / (2 * sigma ** 2))

    return float(score)

def region_score(destination_region: str, group_region: str) -> float:
    """Region match score"""
    return 1.0 if destination_region == group_region else 0.4


def temperature_score_absolute(city_temp_celsius: float, user_temp_pref: str) -> float:
    """
    Temperature scoring based on absolute Celsius values.
    
    Args:
        city_temp_celsius: Actual temperature in Celsius
        user_temp_pref: "cold", "neutral", or "warm"
    
    Returns:
        Score 0-1
    """
    min_t, max_t = TEMP_RANGES[user_temp_pref]
    
    if min_t <= city_temp_celsius <= max_t:
        return 1.0
    else:
        center = (min_t + max_t) / 2
        distance = abs(city_temp_celsius - center)
        sigma = 10
        return float(np.exp(-(distance**2) / (2 * sigma**2)))


def climate_score(avg_temp_norm: float, preferred_temp: float) -> float:
    """Legacy normalized temperature score"""
    return 1.0 - abs(avg_temp_norm - preferred_temp)


def duration_score(city_ideal_durations_str: str, user_days: int) -> float:
    """
    Duration matching score.
    
    Args:
        city_ideal_durations_str: JSON array like '["Short trip", "One week"]'
        user_days: Number of days user plans to stay
    
    Returns:
        Score 0-1
    """
    try:
        city_durations = ast.literal_eval(city_ideal_durations_str)
    except:
        return 0.5
    
    if not city_durations:
        return 0.5
    
    city_durations_lower = [d.lower() for d in city_durations]
    
    scores = []
    for duration_name in city_durations_lower:
        if duration_name not in DURATION_DAYS:
            continue
        
        ideal_days = DURATION_DAYS[duration_name]
        distance = abs(user_days - ideal_days)
        
        if distance == 0:
            scores.append(1.0)
        elif distance <= 2:
            scores.append(0.8)
        elif distance <= 4:
            scores.append(0.6)
        else:
            scores.append(max(0, 1 - distance / 10))
    
    return float(max(scores)) if scores else 0.3


def calculate_individual_scores(
    df: pd.DataFrame,
    group_input: dict,
    group_profile: dict
) -> pd.DataFrame:
    """Calculate individual satisfaction scores for fairness"""
    df = df.copy()
    group = group_input["group"]

    for person_idx, person in enumerate(group):
        def person_score(row):
            score = 0.0

            person_temp = TEMP_MAP[person["temperature"]]
            temp_match = 1.0 - abs(row["avg_temp_norm"] - person_temp)
            score += 0.3 * temp_match

            base_interest = row.get(
                "hybrid_text_score",
                row.get("interest_score", 0)
            )
            score += 0.5 * base_interest

            estimated_cost = row["estimated_cost"]
            person_budget = person["max_budget_eur"]

            if estimated_cost <= person_budget:
                score += 0.2
            elif person_budget <= 0:
                penalty = 0.0
            else:
                penalty = max(0, 1 - (estimated_cost - person_budget) / person_budget)
                score += 0.2 * penalty

            return float(score)

        df[f"person_{person_idx}_score"] = df.apply(person_score, axis=1)

    return df


def calculate_fairness_metrics(df: pd.DataFrame, group_size: int) -> pd.DataFrame:
    """Calculate fairness metrics"""
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


def apply_scoring(
    destinations: pd.DataFrame,
    group_profile: dict,
    group_input: dict = None,
    use_fairness: bool = True
) -> pd.DataFrame:
    """
    Main scoring function combining all criteria.
    
    Final score weights:
    - 35% text matching (hybrid_text_score)
    - 20% budget
    - 15% temperature
    - 10% duration
    - 10% region
    - 10% fairness (if group)
    """
    df = destinations.copy()

    # Compute cost
    df["estimated_cost"] = df.apply(
        lambda row: cost_func(
            distance_km=row["distance_km"],
            budget_level=row["budget_level"],
            days=group_profile["trip_days"]
        ),
        axis=1
    )

    # Budget score
    df["budget_score"] = df.apply(
        lambda row: budget_utilization_score(row, group_profile["group_budget"]),
        axis=1
    )


    # Region score
    #df["region_score"] = df["region"].apply(
    #    lambda r: region_score(r, group_profile["region"])
    #)

    # Temperature score (use absolute if available)
    if "avg_temp" in df.columns:
        user_temp_pref = "warm"
        if group_input and len(group_input.get("group", [])) > 0:
            user_temp_pref = group_input["group"][0]["temperature"]
        
        df["temp_score"] = df["avg_temp"].apply(
            lambda t: temperature_score_absolute(t, user_temp_pref)
        )
    else:
        df["temp_score"] = df["avg_temp_norm"].apply(
            lambda t: climate_score(t, group_profile["preferred_temperature"])
        )

    # Duration score
    if "ideal_durations" in df.columns:
        df["duration_score"] = df["ideal_durations"].apply(
            lambda x: duration_score(x, group_profile["trip_days"])
        )
    else:
        df["duration_score"] = 0.5

    # WITH FAIRNESS
    if use_fairness and group_input is not None:
        df = calculate_individual_scores(df, group_input, group_profile)
        df = calculate_fairness_metrics(df, group_profile["group_size"])

        df["final_score"] = (
            0.40 * df.get("hybrid_text_score", df.get("interest_score", 0)) +
            0.25 * df["budget_score"] +
            0.15 * df["temp_score"] +
            0.10 * df["duration_score"] +
            0.10 * df["fairness_score"]
        )               



    # WITHOUT FAIRNESS
    else:
        df["final_score"] = (
            0.45 * df.get("hybrid_text_score", df.get("interest_score", 0)) +
            0.25 * df["budget_score"] +
            0.15 * df["temp_score"] +
            0.15 * df["duration_score"]
        )

    
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df