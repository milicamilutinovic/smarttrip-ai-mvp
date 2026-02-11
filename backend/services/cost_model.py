# backend/services/cost_model.py

import math


# ----------------------------
# TRANSPORT COST MODEL
# ----------------------------

def estimate_transport_cost(distance_km: float) -> float:
    """
    Heuristic estimation of transport cost per person (EUR)
    based on distance from start city.
    """

    if distance_km < 600:
        # bus or car
        return distance_km * 0.06

    elif distance_km < 1500:
        # short haul flight
        return distance_km * 0.10

    else:
        # long haul flight
        return distance_km * 0.13


# ----------------------------
# ACCOMMODATION COST MODEL
# ----------------------------

ACCOMMODATION_COST = {
    "budget": 30,
    "mid-range": 70,
    "luxury": 120
}


def estimate_accommodation_cost(
    budget_level: str,
    days: int
) -> float:
    """
    Estimates accommodation cost per person.
    """

    per_night = ACCOMMODATION_COST.get(
        budget_level,
        60  # default fallback
    )

    return per_night * days


# ----------------------------
# TOTAL TRIP COST
# ----------------------------

def estimate_total_trip_cost(
    distance_km: float,
    budget_level: str,
    days: int
) -> float:
    """
    Total estimated cost per person.
    """

    transport = estimate_transport_cost(distance_km)
    accommodation = estimate_accommodation_cost(budget_level, days)

    return transport + accommodation
