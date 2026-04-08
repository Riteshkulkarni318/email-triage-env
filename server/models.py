from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum

class EmailCategory(str, Enum):
    SPAM = "spam"
    URGENT = "urgent"
    NORMAL = "normal"
    PROMOTIONAL = "promotional"

class Department(str, Enum):
    SUPPORT = "support"
    SALES = "sales"
    BILLING = "billing"
    TECHNICAL = "technical"
    HR = "hr"

class EmailObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    task: str
    step: int
    max_steps: int
    context: Optional[Dict[str, Any]] = None

class ClassifyAction(BaseModel):
    category: EmailCategory
    confidence: float  # 0.0-1.0
    reason: Optional[str] = None

class TriageAction(BaseModel):
    category: EmailCategory
    department: Department
    priority: int  # 1-5, 5=highest
    reason: Optional[str] = None

class RespondAction(BaseModel):
    subject: str
    body: str
    tone: str  # professional/empathetic/firm
    resolved: bool

class StepResult(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResult(BaseModel):
    observation: EmailObservation

class StateResult(BaseModel):
    task: str
    step: int
    max_steps: int
    total_reward: float
    emails_processed: int
    done: bool