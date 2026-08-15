from pydantic import BaseModel, Field
from typing import List, Dict


class CustomerInput(BaseModel):
    age: int = Field(ge=18, le=100, description="Customer age")
    annual_income: float = Field(ge=0, description="Annual income")
    total_spent: float = Field(ge=0, description="Total spending")
    number_of_orders: int = Field(ge=0, le=500, description="Number of orders")
    avg_order_value: float = Field(ge=0, description="Average order value")
    days_since_last_purchase: int = Field(ge=0, le=1000, description="Days since last purchase")
    discount_usage_pct: float = Field(ge=0, le=100, description="Discount usage percentage")
    website_visits: int = Field(ge=0, le=5000, description="Website visits")
    channel: str = Field(description="Online, In-store, or Mobile app")
    region: str = Field(description="North, South, East, West")
    loyalty_tier: str = Field(description="Bronze, Silver, Gold, Platinum")


class ClusterResponse(BaseModel):
    cluster: int
    cluster_label: str


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float


class MetricsResponse(BaseModel):
    n_clusters: int
    silhouette_score: float
    inertia: float
    cluster_counts: Dict[int, int]
