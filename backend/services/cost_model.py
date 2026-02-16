"""
COST ESTIMATION MODEL
=====================
Estimates total trip cost per person including:
- Transport (based on distance)
- Accommodation (based on budget level)
- Daily spending (food, activities, local transport)
"""

import math


def estimate_transport_cost(distance_km: float) -> float:
    """
    Heuristic estimation of transport cost per person (EUR)
    based on distance from start city.
    """
    if distance_km < 600:
        return distance_km * 0.06
    elif distance_km < 1500:
        return distance_km * 0.10
    else:
        return distance_km * 0.13


ACCOMMODATION_COST = {
    "budget": 30,
    "mid-range": 70,
    "luxury": 120
}


def estimate_accommodation_cost(budget_level: str, days: int) -> float:
    """
    Estimates accommodation cost per person.
    """
    per_night = ACCOMMODATION_COST.get(budget_level, 60)
    return per_night * days


DAILY_SPENDING = {
    "budget": 40,
    "mid-range": 80,
    "luxury": 150
}


def estimate_daily_spending(budget_level: str, days: int) -> float:
    """
    Estimates daily spending (food, activities, local transport) per person.
    """
    per_day = DAILY_SPENDING.get(budget_level, 70)
    return per_day * days


def estimate_total_trip_cost(distance_km: float, budget_level: str, days: int) -> float:
    """
    Total estimated cost per person (transport + accommodation + daily spending).
    """
    transport = estimate_transport_cost(distance_km)
    accommodation = estimate_accommodation_cost(budget_level, days)
    daily = estimate_daily_spending(budget_level, days)
    
    return transport + accommodation + daily