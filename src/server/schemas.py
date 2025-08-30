from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    review: Dict[str, Any] = Field(..., description="Single Google review JSON object")

class ReviewResponse(BaseModel):
    model: str
    final_decision: Literal["relevant", "not_relevant"]
    explanation: str
    confidence: float
    features: Dict[str, Any]
    llm_vote: Optional[Dict[str, Any]] = None
