from typing import Dict, Any, Tuple

def grade_classify(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    """Grade classification task. Returns (reward, info)"""
    reward = 0.0
    info = {}

    category_correct = action_data.get("category") == ground_truth["category"]
    confidence = float(action_data.get("confidence", 0.5))

    if category_correct:
        reward += 0.7
        # Bonus for appropriate confidence
        if confidence >= 0.7:
            reward += 0.2
        elif confidence >= 0.5:
            reward += 0.1
    else:
        # Partial credit for close categories
        predicted = action_data.get("category", "")
        actual = ground_truth["category"]
        # urgent vs normal is closer than spam vs urgent
        close_pairs = [("urgent", "normal"), ("normal", "promotional")]
        for a, b in close_pairs:
            if (predicted == a and actual == b) or (predicted == b and actual == a):
                reward += 0.2
                break

    # Reason provided
    if action_data.get("reason"):
        reward += 0.1

    info["category_correct"] = category_correct
    info["confidence"] = confidence
    return min(reward, 1.0), info


def grade_triage(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    """Grade triage task. Returns (reward, info)"""
    reward = 0.0
    info = {}

    category_correct = action_data.get("category") == ground_truth["category"]
    dept_correct = action_data.get("department") == ground_truth["department"]
    priority = int(action_data.get("priority", 1))
    expected_priority = ground_truth["priority"]
    priority_diff = abs(priority - expected_priority)

    if category_correct:
        reward += 0.3
    if dept_correct:
        reward += 0.4

    # Priority scoring (partial credit)
    if priority_diff == 0:
        reward += 0.2
    elif priority_diff == 1:
        reward += 0.1

    if action_data.get("reason"):
        reward += 0.1

    info["category_correct"] = category_correct
    info["dept_correct"] = dept_correct
    info["priority_diff"] = priority_diff
    return min(reward, 1.0), info


def grade_respond(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    """Grade response task. Returns (reward, info)"""
    reward = 0.0
    info = {}

    body = (action_data.get("body") or "").lower()
    subject = (action_data.get("subject") or "").lower()
    resolved = action_data.get("resolved", False)
    tone = action_data.get("tone", "")

    # Length check
    min_len = ground_truth.get("min_response_length", 50)
    if len(body) >= min_len:
        reward += 0.2

    # Required keywords
    required_keywords = ground_truth.get("keywords_required", [])
    if required_keywords:
        matched = sum(1 for kw in required_keywords if kw.lower() in body)
        keyword_score = matched / len(required_keywords)
        reward += 0.4 * keyword_score
        info["keywords_matched"] = matched
        info["keywords_total"] = len(required_keywords)
    else:
        reward += 0.4  # No keywords required = full marks for that section

    # Resolution check
    must_resolve = ground_truth.get("must_resolve", False)
    if must_resolve and resolved:
        reward += 0.2
    elif not must_resolve:
        reward += 0.2

    # Tone appropriateness
    valid_tones = ["professional", "empathetic", "firm"]
    if tone in valid_tones:
        reward += 0.1

    # Subject line present and relevant
    if subject and len(subject) > 5:
        reward += 0.1

    info["resolved"] = resolved
    info["tone"] = tone
    info["body_length"] = len(body)
    return min(reward, 1.0), info