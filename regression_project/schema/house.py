from pydantic import BaseModel, Field
from typing import List


class HouseInput(BaseModel):
    neighborhood: str = Field(description="Neighborhood location")
    house_type: str = Field(description="Type of house")
    bedrooms: int = Field(ge=1, le=10, description="Number of bedrooms")
    bathrooms: int = Field(ge=1, le=6, description="Number of bathrooms")
    sqft: int = Field(ge=200, le=10000, description="Square footage")
    lot_size: int = Field(ge=100, le=50000, description="Lot size in sqft")
    year_built: int = Field(ge=1950, le=2025, description="Year the house was built")
    condition: str = Field(description="House condition")
    garage: int = Field(ge=0, le=1, description="Has garage (0 or 1)")
    pool: int = Field(ge=0, le=1, description="Has pool (0 or 1)")
    distance_to_city_km: float = Field(ge=0, le=100, description="Distance to city in km")


class PredictionResponse(BaseModel):
    predicted_price: float


class BatchPredictionResponse(BaseModel):
    predicted_prices: List[float]
