from typing import List, Dict, Any

EMAILS: List[Dict[str, Any]] = [
    # EASY - Classification set
    {
        "id": "e001",
        "subject": "URGENT: Server Down - Production Impact",
        "body": "Our main production server has been down for 30 minutes. Revenue impact is $10k/min. Need immediate escalation.",
        "sender": "ops-team@company.com",
        "timestamp": "2024-01-15T09:00:00Z",
        "ground_truth": {
            "category": "urgent",
            "department": "technical",
            "priority": 5,
            "keywords_required": ["apologize", "escalat", "immediately", "priorit"],
            "min_response_length": 50,
            "must_resolve": True,
        }
    },
    {
        "id": "e002",
        "subject": "50% OFF - Limited Time Offer Just For You!",
        "body": "Click here to claim your exclusive discount! This offer expires in 24 hours. Don't miss out on amazing deals!",
        "sender": "noreply@deals-promo.net",
        "timestamp": "2024-01-15T09:05:00Z",
        "ground_truth": {
            "category": "spam",
            "department": "support",
            "priority": 1,
            "keywords_required": [],
            "min_response_length": 0,
            "must_resolve": False,
        }
    },
    {
        "id": "e003",
        "subject": "Invoice #4521 - Payment Overdue",
        "body": "Dear Team, I have been trying to resolve my invoice dispute for 3 weeks. Invoice #4521 shows charges I did not authorize. Please help.",
        "sender": "john.doe@client.com",
        "timestamp": "2024-01-15T09:10:00Z",
        "ground_truth": {
            "category": "urgent",
            "department": "billing",
            "priority": 4,
            "keywords_required": ["apologize", "invoice", "review", "contact"],
            "min_response_length": 80,
            "must_resolve": True,
        }
    },
    {
        "id": "e004",
        "subject": "Team lunch next Friday?",
        "body": "Hey everyone, thinking of organizing a team lunch next Friday at noon. Let me know if you're interested!",
        "sender": "alice@company.com",
        "timestamp": "2024-01-15T09:15:00Z",
        "ground_truth": {
            "category": "normal",
            "department": "hr",
            "priority": 2,
            "keywords_required": [],
            "min_response_length": 20,
            "must_resolve": False,
        }
    },
    {
        "id": "e005",
        "subject": "API Integration broken after your update",
        "body": "Since your latest update (v2.3.1), our API integration has been returning 500 errors on every POST request. This is blocking our entire dev team.",
        "sender": "dev@partner-company.com",
        "timestamp": "2024-01-15T09:20:00Z",
        "ground_truth": {
            "category": "urgent",
            "department": "technical",
            "priority": 5,
            "keywords_required": ["apologize", "bug", "fix", "version", "team"],
            "min_response_length": 100,
            "must_resolve": True,
        }
    },
    {
        "id": "e006",
        "subject": "Request for enterprise pricing",
        "body": "Hi, we are a 500-person company looking to switch to your platform. Could you share enterprise pricing and schedule a demo?",
        "sender": "procurement@bigcorp.com",
        "timestamp": "2024-01-15T09:25:00Z",
        "ground_truth": {
            "category": "normal",
            "department": "sales",
            "priority": 4,
            "keywords_required": ["pricing", "demo", "enterprise", "team"],
            "min_response_length": 80,
            "must_resolve": False,
        }
    },
    {
        "id": "e007",
        "subject": "Congratulations! You've won a free iPhone!",
        "body": "You have been selected as a lucky winner! Claim your free iPhone 15 Pro by clicking the link and entering your credit card for shipping.",
        "sender": "winner@free-prizes-online.xyz",
        "timestamp": "2024-01-15T09:30:00Z",
        "ground_truth": {
            "category": "spam",
            "department": "support",
            "priority": 1,
            "keywords_required": [],
            "min_response_length": 0,
            "must_resolve": False,
        }
    },
    {
        "id": "e008",
        "subject": "Password reset not working - locked out",
        "body": "I've been locked out of my account for 2 hours. The password reset email never arrives. I have an important presentation in 1 hour and need access NOW.",
        "sender": "sarah.smith@client.org",
        "timestamp": "2024-01-15T09:35:00Z",
        "ground_truth": {
            "category": "urgent",
            "department": "technical",
            "priority": 5,
            "keywords_required": ["apologize", "reset", "manual", "access"],
            "min_response_length": 80,
            "must_resolve": True,
        }
    },
]

TASK_EMAIL_MAP = {
    "classify": ["e001", "e002", "e003", "e004", "e007"],
    "triage":   ["e001", "e003", "e005", "e006", "e008"],
    "respond":  ["e001", "e003", "e005", "e006", "e008"],
}

def get_emails_for_task(task: str):
    ids = TASK_EMAIL_MAP.get(task, [])
    email_map = {e["id"]: e for e in EMAILS}
    return [email_map[i] for i in ids if i in email_map]