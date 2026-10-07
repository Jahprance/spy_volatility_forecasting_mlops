from pydantic import BaseModel, Field

FEATURE_ORDER = [
    "har_d",
    "har_w",
    "har_m",
    "rv_lag1",
    "abs_return",
    "hl_range",
    "log_volume",
]

class PredictionInput(BaseModel):
    har_d: float = Field(ge=0)
    har_w: float = Field(ge=0)
    har_m: float = Field(ge=0)
    rv_lag1: float = Field(ge=0)
    abs_return: float = Field(ge=0)
    hl_range: float = Field(ge=0)
    log_volume: float
