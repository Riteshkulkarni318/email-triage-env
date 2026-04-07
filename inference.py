"""
Email Triage OpenEnv — Baseline Inference Script
Follows the required [START] / [STEP] / [END] stdout format exactly.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional, Dict, Any
import httpx
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME = os.getenv("EMAIL_TASK", "classify")
BENCHMARK = "email-triage-env"
MAX_STEPS = 5
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPTS = {
    "classify": textwrap.dedent("""
        You are an expert email classifier. Given an email, classify it into exactly one category.
        Categories: spam, urgent, normal, promotional
        Respond ONLY with a JSON object with these exact fields:
        {"category": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}
        No other text, no markdown, just the JSON object.
    """).strip(),

    "triage": textwrap.dedent("""
        You are an expert email triage specialist. Given an email, determine its category, 
        the correct department to handle it, and its priority.
        Categories: spam, urgent, normal, promotional
        Departments: support, sales, billing, technical, hr
        Priority: 1 (lowest) to 5 (highest)
        Respond ONLY with a JSON object:
        {"category": "<category>", "department": "<dept>", "priority": <1-5>, "reason": "<brief reason>"}
        No other text, just the JSON.
    """).strip(),

    "respond": textwrap.dedent("""
        You are a professional customer support agent. Given an email, draft an appropriate response.
        Your response must:
        - Be professional and empathetic
        - Address the customer's issue directly
        - Include relevant action items or next steps
        - Be at least 100 characters long
        Respond ONLY with a JSON object:
        {"subject": "<reply subject>", "body": "<full response body>", "tone": "<professional|empathetic|firm>", "resolved": <true|false>}
        No other text, just the JSON.
    """).strip(),
}


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_agent_action(client: OpenAI, task: str, observation: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    history_block = "\n".join(history[-3:]) if history else "None"
    user_prompt = f"""
Email to process:
Subject: {observation.get('subject', '')}
From: {observation.get('sender', '')}
Body: {observation.get('body', '')}

Step: {observation.get('step', 1)} of {observation.get('max_steps', 5)}
Previous steps: {history_block}
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        # Fallback actions
        fallbacks = {
            "classify": {"category": "normal", "confidence": 0.5, "reason": "fallback"},
            "triage": {"category": "normal", "department": "support", "priority": 2, "reason": "fallback"},
            "respond": {"subject": "Re: Your Email", "body": "Thank you for reaching out. We will look into this shortly.", "tone": "professional", "resolved": False},
        }
        return fallbacks[task]


async def run_task(task: str) -> float:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(base_url=ENV_URL, timeout=30.0) as http:
            # Reset
            resp = await http.post("/reset", json={"task": task})
            resp.raise_for_status()
            data = resp.json()
            observation = data["observation"]
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action = get_agent_action(client, task, observation, history)
                action_str = json.dumps(action, separators=(',', ':'))

                # Step
                resp = await http.post("/step", json={"action": action, "task": task})
                resp.raise_for_status()
                result = resp.json()

                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                error = result.get("info", {}).get("error")

                rewards.append(reward)
                steps_taken = step
                observation = result.get("observation", observation)

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                history.append(f"Step {step}: reward={reward:.2f}")

                if done:
                    break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task} error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    tasks = ["classify", "triage", "respond"]
    all_scores = []
    for task in tasks:
        score = await run_task(task)
        all_scores.append(score)
        print(f"[INFO] Task {task} completed with score {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"[INFO] Average score across all tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())