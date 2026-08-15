from pydantic import BaseModel, Field
from typing import List


class ChurnInput(BaseModel):
    age: int = Field(ge=18, le=100, description="Customer age")
    gender: str = Field(description="Male or Female")
    tenure_months: int = Field(ge=0, le=200, description="Months as customer")
    monthly_charges: float = Field(ge=0, le=500, description="Monthly bill amount")
    total_spend: float = Field(ge=0, description="Total amount spent")
    contract_type: str = Field(description="Month-to-month, One year, Two year")
    payment_method: str = Field(description="Payment method used")
    internet_service: str = Field(description="DSL, Fiber optic, or No")
    support_tickets: int = Field(ge=0, le=50, description="Number of support tickets")
    has_partner: int = Field(ge=0, le=1, description="Has partner (0 or 1)")
    has_dependents: int = Field(ge=0, le=1, description="Has dependents (0 or 1)")
    paperless_billing: int = Field(ge=0, le=1, description="Paperless billing (0 or 1)")


class ChurnResponse(BaseModel):
    churn: bool
    probability: float


class BatchChurnResponse(BaseModel):
    results: List[ChurnResponse]
