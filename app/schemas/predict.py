from pydantic import BaseModel, Field
from typing import List

class TopPrediction(BaseModel):
    rank: int
    name: str
    confidence: float

class Response(BaseModel):
    predicted_name_image: str = Field(...)
    confidence: float = Field(...)
    probability: List[TopPrediction]
