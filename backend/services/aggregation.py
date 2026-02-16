"""
GROUP AGGREGATION MODULE
========================
Features:
- Hard constraints (budget, duration)
- Temperature aggregation
- Activity interest aggregation
- Text description aggregation (NEW)
- Temperature variance (fairness signal)
- Activity diversity (Jaccard-based)
- Individual preference tracking
"""

from collections import defaultdict
from typing import Dict, List
import numpy as np
from backend.services.scoring import DURATION_DAYS


TEMP_MAP = {
    "cold": 0.0,
    "neutral": 0.5,
    "warm": 1.0
}


def aggregate_group_preferences(
    group_input: Dict,
    strategy: str = "average_with_fairness"
) -> Dict:

    group = group_input["group"]
    size = len(group)

    if size == 0:
        raise ValueError("Group cannot be empty.")

    # HARD CONSTRAINTS
    group_budget = min(p["max_budget_eur"] for p in group)

    group_days = min(
        DURATION_DAYS[p["max_days"]]
        for p in group
    )


    # TEMPERATURE AGGREGATION
    temps = [TEMP_MAP[p["temperature"]] for p in group]

    if strategy == "least_misery":
        preferred_temperature = np.median(temps)
    else:
        preferred_temperature = sum(temps) / size

    temp_variance = float(np.var(temps))

    # ACTIVITY AGGREGATION
    interests = defaultdict(int)
    individual_activities = []

    for p in group:
        person_interests = {}
        for act in p.get("activities", []):
            interests[act] += 1
            person_interests[act] = 1.0
        individual_activities.append(person_interests)

    interest_profile = {
        k: v / size for k, v in interests.items()
    }

    activity_diversity = calculate_activity_diversity(individual_activities)

    # TEXT DESCRIPTION AGGREGATION
    descriptions = []
    individual_descriptions = []

    for p in group:
        desc = p.get("description", "")
        if desc:
            descriptions.append(desc)
            individual_descriptions.append(desc)

    group_description = " ".join(descriptions)

    # BUILD GROUP PROFILE (NO REGION)
    group_profile = {
        "start_city": group_input["start_city"],
        "month": group_input["month"],

        "group_budget": group_budget,
        "trip_days": group_days,

        "preferred_temperature": preferred_temperature,
        "interests": interest_profile,

        "group_description": group_description,
        "individual_descriptions": individual_descriptions,

        "group_size": size,
        "individual_temps": temps,
        "temp_variance": temp_variance,
        "individual_activities": individual_activities,
        "activity_diversity": activity_diversity,

        "aggregation_strategy": strategy
    }

    return group_profile



def calculate_activity_diversity(individual_activities: List[Dict]) -> float:
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


def calculate_individual_satisfaction(destination_row, person_preferences: Dict) -> float:
    """
    Computes individual satisfaction score (0-1) for fairness penalization layer.
    Used inside scoring module.
    """
    satisfaction = 0.0

    person_temp = TEMP_MAP[person_preferences["temperature"]]
    dest_temp = destination_row["avg_temp_norm"]

    temp_satisfaction = 1.0 - abs(person_temp - dest_temp)
    satisfaction += 0.4 * temp_satisfaction

    activity_match = 0.0
    person_activities = person_preferences.get("activities", [])

    if person_activities:
        for act in person_activities:
            if act in destination_row:
                activity_match += destination_row[act] / 5.0

        activity_match = min(1.0, activity_match / len(person_activities))

    satisfaction += 0.6 * activity_match

    return float(satisfaction)