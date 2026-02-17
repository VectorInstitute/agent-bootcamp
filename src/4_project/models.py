from datetime import date
from pydantic import BaseModel


class UserAccountContext(BaseModel):
    customer_id: str
    name: str
    nra: int = "65"  # 60 or 65
    status: str = "active"  # active, inactive, deferred, retired
    # dateOfBirth: date
    # enrolmentDate: date
    # totalContribution: float
    # contributionPerPayPeriod: float
    # numberOfBeneficiaries: int


class InputGuardRailOutput(BaseModel):
    is_off_topic: bool
    reason: str
