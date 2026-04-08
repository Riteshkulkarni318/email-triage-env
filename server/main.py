from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

from env import EmailTriageEnv
from models import StepResult, ResetResult, StateResult

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv environment for email triage and response tasks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instances per session (simple in-memory for HF Space)
_envs: Dict[str, EmailTriageEnv] = {}
_default_task = os.getenv("EMAIL_TASK", "classify")


def get_or_create_env(task: Optional[str] = None) -> EmailTriageEnv:
    t = task or _default_task
    if t not in _envs:
        _envs[t] = EmailTriageEnv(task=t)
    return _envs[t]


class ResetRequest(BaseModel):
    task: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]
    task: Optional[str] = None


@app.get("/")
def root():
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "tasks": ["classify", "triage", "respond"],
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = None) -> dict:
    task = (req.task if req else None) or _default_task
    env = EmailTriageEnv(task=task)
    _envs[task] = env
    result = env.reset()
    return result.model_dump()

@app.post("/step")
def step(req: StepRequest) -> dict:
    task = req.task or _default_task
    env = get_or_create_env(task)
    result = env.step(req.action)
    return result.model_dump()

@app.get("/state")
def state(task: Optional[str] = None) -> dict:
    t = task or _default_task
    env = get_or_create_env(t)
    return env.state().model_dump()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "classify",
                "difficulty": "easy",
                "description": "Classify emails into categories: spam/urgent/normal/promotional",
                "max_steps": 5,
                "score_range": [0.0, 1.0]
            },
            {
                "name": "triage",
                "difficulty": "medium",
                "description": "Triage emails to correct department with priority ranking",
                "max_steps": 5,
                "score_range": [0.0, 1.0]
            },
            {
                "name": "respond",
                "difficulty": "hard",
                "description": "Draft appropriate email responses satisfying content requirements",
                "max_steps": 5,
                "score_range": [0.0, 1.0]
            }
        ]
    }