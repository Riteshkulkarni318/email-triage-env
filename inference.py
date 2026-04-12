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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL   = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "email-triage-env"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.5

# Scores must be strictly between 0 and 1 (exclusive)
SCORE_MIN = 0.001
SCORE_MAX = 0.999

SYSTEM_PROMPTS = {
    "classify": textwrap.dedent("""
        You are an expert email classifier. Classify the email into exactly one category:
        - spam: phishing, scam, unsolicited marketing
        - urgent: needs immediate action (outages, legal threats, security, locked accounts)
        - normal: standard business requests, questions, feedback
        - promotional: legitimate marketing from known companies

        Respond ONLY with valid JSON (no markdown):
        {"category": "<category>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}
    """).strip(),

    "triage": textwrap.dedent("""
        You are an email triage specialist. Determine category, department, and priority.
        Categories: spam, urgent, normal, promotional
        Departments: technical, billing, support, sales, hr
        Priority: 1 (lowest) to 5 (highest/critical)

        Respond ONLY with valid JSON (no markdown):
        {"category": "<cat>", "department": "<dept>", "priority": <1-5>, "reason": "<one sentence>"}
    """).strip(),

    "respond": textwrap.dedent("""
        You are a senior customer support agent. Write a professional email response.
        - Show empathy and acknowledge the issue
        - Provide concrete next steps
        - Be at least 150 characters long
        - End professionally

        Respond ONLY with valid JSON (no markdown):
        {"subject": "Re: <subject>", "body": "<full response>", "tone": "<professional|empathetic|firm>", "resolved": <true|false>}
    """).strip(),

    "summarize": textwrap.dedent("""
        You are a business email analyst. Write a concise summary of the email.
        Include: main issue, urgency, and recommended action.

        Respond ONLY with valid JSON (no markdown):
        {"summary": "<2-3 sentence summary>", "urgency": "<low|medium|high|critical>", "action_required": "<what needs to happen>"}
    """).strip(),
}

FALLBACKS = {
    "classify":  {"category": "normal", "confidence": 0.5, "reason": "fallback"},
    "triage":    {"category": "normal", "department": "support", "priority": 2, "reason": "fallback"},
    "respond":   {"subject": "Re: Your Email", "body": "Thank you for reaching out. We sincerely apologize for any inconvenience. Our team will review your request immediately and contact you within 24 hours to resolve this matter.", "tone": "professional", "resolved": False},
    "summarize": {"summary": "Customer sent an email requiring team attention and follow-up.", "urgency": "medium", "action_required": "Review and respond to customer inquiry promptly."},
}


def clamp_score(value: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by the grader."""
    return min(max(float(value), SCORE_MIN), SCORE_MAX)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    action_clean = str(action).replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def get_action(client, task, obs, history):
    history_str = "\n".join(history[-3:]) if history else "None"
    prompt = f"""Email:
Subject: {obs.get('subject', 'N/A')}
From: {obs.get('sender', 'N/A')}
Body: {obs.get('body', 'N/A')}

Step {obs.get('step', 1)} of {obs.get('max_steps', 8)}
History: {history_str}

Reply with ONLY the JSON object."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        text = (resp.choices[0].message.content or "").strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"[DEBUG] Model error task={task}: {e}", flush=True)
        return FALLBACKS[task]


async def run_task(task: str) -> float:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(base_url=ENV_URL, timeout=60.0) as http:
            r = await http.post("/reset", json={"task": task})
            r.raise_for_status()
            obs = r.json().get("observation", {})
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action = get_action(client, task, obs, history)
                action_str = json.dumps(action, separators=(',', ':'))

                r = await http.post("/step", json={"action": action, "task": task})
                r.raise_for_status()
                result = r.json()

                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                error  = result.get("info", {}).get("error")

                rewards.append(reward)
                steps_taken = step
                obs = result.get("observation", obs)

                log_step(step, action_str, reward, done, error)
                history.append(f"step={step} reward={reward:.2f}")

                if done:
                    break

        raw_score = sum(rewards) / len(rewards) if rewards else 0.0
        # FIX: clamp to open interval (0, 1) — grader rejects exactly 0.0 or 1.0
        score = clamp_score(raw_score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error task={task}: {e}", flush=True)
        score = SCORE_MIN  # safe fallback, still strictly > 0
    finally:
        log_end(success, steps_taken, score, rewards)

    return score


async def main():
    tasks = ["classify", "triage", "respond", "summarize"]
    scores = []
    for task in tasks:
        score = await run_task(task)
        scores.append(score)
        print(f"[INFO] task={task} score={score:.3f}", flush=True)
    avg = sum(scores) / len(scores)
    print(f"[INFO] average={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
