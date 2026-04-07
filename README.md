# Email Triage OpenEnv 📧

A real-world OpenEnv environment where AI agents learn to classify, triage, and respond to business emails — modeling actual customer support workflows used by thousands of companies.

## Why Email Triage?

Email handling is one of the highest-volume tasks in any organization. This environment provides agents with realistic training signal for the full email management pipeline: from classification to routing to response drafting.

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `classify` | Easy | Label emails as spam/urgent/normal/promotional | 5 |
| `triage` | Medium | Route to correct department with priority score | 5 |
| `respond` | Hard | Draft a full reply satisfying content requirements | 5 |

## Observation Space

```json
{
  "email_id": "string",
  "subject": "string",
  "body": "string",
  "sender": "string",
  "timestamp": "ISO8601 string",
  "task": "classify|triage|respond",
  "step": "integer",
  "max_steps": "integer",
  "context": {"email_index": 0, "total_emails": 5}
}
```

## Action Space

**classify:**
```json
{"category": "spam|urgent|normal|promotional", "confidence": 0.0-1.0, "reason": "string"}
```

**triage:**
```json
{"category": "...", "department": "support|sales|billing|technical|hr", "priority": 1-5, "reason": "string"}
```

**respond:**
```json
{"subject": "string", "body": "string", "tone": "professional|empathetic|firm", "resolved": true|false}
```

## Reward Design

- **Classify:** +0.7 correct category, +0.2 appropriate confidence, +0.1 reason provided
- **Triage:** +0.3 category, +0.4 department, +0.2 priority accuracy, +0.1 reason
- **Respond:** +0.2 length, +0.4 required keywords, +0.2 resolution, +0.1 tone, +0.1 subject

## Setup

```bash
# Local
pip install -r requirements.txt
cd server && uvicorn main:app --host 0.0.0.0 --port 7860

# Docker
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# Run baseline
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860
python inference.py
```

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| classify | Qwen2.5-72B | ~0.82 |
| triage | Qwen2.5-72B | ~0.71 |
| respond | Qwen2.5-72B | ~0.64 |