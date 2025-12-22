from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field
from pydantic.config import ConfigDict


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime


class AnalyzeRequest(BaseModel):
    text: str


class Segment(BaseModel):
    start: int
    end: int
    prob_ai: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "label": 1,
                    "prob_ai": 0.76,
                    "segments": [{"start": 0, "end": 120, "prob_ai": 0.81}],
                    "threshold": 0.10,
                }
            ]
        }
    )

    label: int = Field(
        description="0 = Human, 1 = AI-generated",
        examples=[1],
    )
    prob_ai: float = Field(
        description="Probability that the text is AI-generated",
        examples=[0.76],
    )
    segments: List[Segment]
    threshold: Optional[float] = Field(
        default=None,
        description="Decision threshold used for label selection",
        examples=[0.10],
    )


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    label_pred: int
    prob_ai: float
    text: str
    segments: List[Segment]


class AnalysisPreviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    label_pred: int
    prob_ai: float
    text_preview: str
    segments: List[Segment]


class StatsOut(BaseModel):
    total_count: int
    ai_count: int
    human_count: int
    avg_prob_ai: float
