"""
GROUP AGGREGATION MODULE
=========================
Advanced group-aware aggregation with fairness metrics.

Features:
- Hard constraints (budget, duration)
- Temperature aggregation
- Activity interest aggregation
- Temperature variance (fairness signal)
- Activity diversity (Jaccard-based)
- Individual preference tracking
"""

from collections import defaultdict
from typing import Dict, List
import numpy as np


# -------------------------------------------------
# Temperature mapping to normalized space
# -------------------------------------------------
TEMP_MAP = {
    "cold": 0.0,
    "neutral": 0.5,
    "warm": 1.0
}


# -------------------------------------------------
# Main aggregation function
# -------------------------------------------------
def aggregate_group_preferences(
    group_input: Dict,
    strategy: str = "average_with_fairness"
) -> Dict:
    """
    Aggregate group preferences with fairness awareness.

    Args:
        group_input: {
            "start_city": str,
            "month": int,
            "region": str,
            "group": [
                {
                    "max_budget_eur": int,
                    "max_days": int,
                    "temperature": str,
                    "activities": List[str]
                },
                ...
            ]
        }

        strategy:
            - "average"
            - "least_misery"
            - "average_with_fairness"

    Returns:
        group_profile dict
    """

    group = group_input["group"]
    size = len(group)

    if size == 0:
        raise ValueError("Group cannot be empty.")

    # -------------------------------------------------
    # HARD CONSTRAINTS
    # -------------------------------------------------
    group_budget = min(p["max_budget_eur"] for p in group)
    group_days = min(p["max_days"] for p in group)

    # -------------------------------------------------
    # TEMPERATURE AGGREGATION
    # -------------------------------------------------
    temps = [TEMP_MAP[p["temperature"]] for p in group]

    if strategy == "least_misery":
        # Conservative – middle ground
        preferred_temperature = np.median(temps)
    else:
        preferred_temperature = sum(temps) / size

    temp_variance = float(np.var(temps))

    # -------------------------------------------------
    # ACTIVITY AGGREGATION
    # -------------------------------------------------
    interests = defaultdict(int)
    individual_activities = []

    for p in group:
        person_interests = {}
        for act in p.get("activities", []):
            interests[act] += 1
            person_interests[act] = 1.0
        individual_activities.append(person_interests)

    # Normalized interest weights (0-1)
    interest_profile = {
        k: v / size for k, v in interests.items()
    }

    activity_diversity = calculate_activity_diversity(individual_activities)

    # -------------------------------------------------
    # BUILD GROUP PROFILE
    # -------------------------------------------------
    group_profile = {
        "start_city": group_input["start_city"],
        "month": group_input["month"],
        "region": group_input["region"],

        # Hard constraints
        "group_budget": group_budget,
        "trip_days": group_days,

        # Soft preferences
        "preferred_temperature": preferred_temperature,
        "interests": interest_profile,

        # Fairness tracking
        "group_size": size,
        "individual_temps": temps,
        "temp_variance": temp_variance,
        "individual_activities": individual_activities,
        "activity_diversity": activity_diversity,

        "aggregation_strategy": strategy
    }

    return group_profile


# -------------------------------------------------
# Activity diversity (Jaccard-based)
# -------------------------------------------------
def calculate_activity_diversity(
    individual_activities: List[Dict]
) -> float:
    """
    Measures how different group members are in their activities.

    Returns:
        0 → identical interests
        1 → completely different interests
    """

    if len(individual_activities) < 2:
        return 0.0

    diversities = []

    for i in range(len(individual_activities)):
        for j in range(i + 1, len(individual_activities)):

            set_i = set(individual_activities[i].keys())
            set_j = set(individual_activities[j].keys())

            intersection = len(set_i & set_j)
            union = len(set_i | set_j)

            if union > 0:
                jaccard_similarity = intersection / union
                diversities.append(1 - jaccard_similarity)

    return float(np.mean(diversities)) if diversities else 0.0


# -------------------------------------------------
# Individual satisfaction (for fairness scoring)
# -------------------------------------------------
def calculate_individual_satisfaction(
    destination_row,
    person_preferences: Dict
) -> float:
    """
    Computes individual satisfaction score (0-1)
    for fairness penalization layer.

    Used inside scoring module.
    """

    satisfaction = 0.0

    # Temperature satisfaction
    person_temp = TEMP_MAP[person_preferences["temperature"]]
    dest_temp = destination_row["avg_temp_norm"]

    temp_satisfaction = 1.0 - abs(person_temp - dest_temp)
    satisfaction += 0.4 * temp_satisfaction

    # Activity satisfaction (simple heuristic)
    activity_match = 0.0
    person_activities = person_preferences.get("activities", [])

    if person_activities:
        for act in person_activities:
            if act in destination_row:
                activity_match += destination_row[act] / 5.0

        activity_match = min(1.0, activity_match / len(person_activities))

    satisfaction += 0.6 * activity_match

    return float(satisfaction)
