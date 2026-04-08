import uuid
from typing import Optional, Dict, Any
from models import (
    EmailObservation, StepResult, ResetResult, StateResult,
    ClassifyAction, TriageAction, RespondAction
)
from data import get_emails_for_task
from graders import grade_classify, grade_triage, grade_respond

TASK_CONFIG = {
    "classify": {"max_steps": 5, "grader": grade_classify},
    "triage":   {"max_steps": 5, "grader": grade_triage},
    "respond":  {"max_steps": 5, "grader": grade_respond},
}

class EmailTriageEnv:
    def __init__(self, task: str = "classify"):
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task}. Choose from {list(TASK_CONFIG.keys())}")
        self.task = task
        self.config = TASK_CONFIG[task]
        self.session_id = str(uuid.uuid4())
        self._reset_state()

    def _reset_state(self):
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.emails = get_emails_for_task(self.task)
        self.email_index = 0
        self.rewards_history = []

    def _current_email(self) -> Optional[Dict[str, Any]]:
        if self.email_index < len(self.emails):
            return self.emails[self.email_index]
        return None

    def _make_observation(self) -> EmailObservation:
        email = self._current_email()
        if not email:
            # Fallback observation when done
            return EmailObservation(
                email_id="done",
                subject="",
                body="",
                sender="",
                timestamp="",
                task=self.task,
                step=self.step_count,
                max_steps=self.config["max_steps"],
            )
        return EmailObservation(
            email_id=email["id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            timestamp=email["timestamp"],
            task=self.task,
            step=self.step_count,
            max_steps=self.config["max_steps"],
            context={"email_index": self.email_index, "total_emails": len(self.emails)},
        )

    def reset(self) -> ResetResult:
        self._reset_state()
        return ResetResult(observation=self._make_observation())

    def step(self, action_data: Dict[str, Any]) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._make_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already done"},
            )

        self.step_count += 1
        email = self._current_email()

        if not email:
            self.done = True
            return StepResult(
                observation=self._make_observation(),
                reward=0.0,
                done=True,
                info={"error": "No more emails"},
            )

        ground_truth = email["ground_truth"]
        grader = self.config["grader"]
        reward, info = grader(action_data, ground_truth)

        self.total_reward += reward
        self.rewards_history.append(reward)
        self.email_index += 1

        # Check done
        max_steps = self.config["max_steps"]
        if self.email_index >= len(self.emails) or self.step_count >= max_steps:
            self.done = True

        info["email_id"] = email["id"]
        info["step"] = self.step_count
        info["cumulative_reward"] = self.total_reward

        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=self.done,
            info=info,
        )

    def state(self) -> StateResult:
        return StateResult(
            task=self.task,
            step=self.step_count,
            max_steps=self.config["max_steps"],
            total_reward=self.total_reward,
            emails_processed=self.email_index,
            done=self.done,
        )