from pydantic import BaseModel
from typing import List


class Person(BaseModel):
    max_budget_eur: float
    max_days: str   # STRING
    temperature: str
    description: str


class GroupRequest(BaseModel):
    start_city: str
    month: int
    region: str
    group: List[Person]
