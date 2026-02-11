# backend/api/schemas.py

from pydantic import BaseModel
from typing import List


class Person(BaseModel):
    max_budget_eur: float
    max_days: int
    activities: List[str]
    temperature: str


class GroupRequest(BaseModel):
    start_city: str
    month: str
    region: str
    group: List[Person]
