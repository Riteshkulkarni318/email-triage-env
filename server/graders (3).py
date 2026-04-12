from typing import Dict, Any, Tuple

CATEGORY_PROXIMITY = {
    ("urgent", "normal"): 0.2, ("normal", "urgent"): 0.2,
    ("normal", "promotional"): 0.1, ("promotional", "normal"): 0.1,
    ("spam", "promotional"): 0.1, ("promotional", "spam"): 0.1,
}
DEPT_PROXIMITY = {
    ("support", "technical"): 0.15, ("technical", "support"): 0.15,
    ("billing", "support"): 0.10,   ("support", "billing"): 0.10,
    ("sales", "support"): 0.10,     ("support", "sales"): 0.10,
    ("hr", "support"): 0.10,        ("support", "hr"): 0.10,
}
POSITIVE_TONE  = ["apologize","sorry","understand","empathize","appreciate","thank","regret","concern"]
PROFESSIONAL   = ["please","kindly","sincerely","regards","team","assist","support","help","resolve","address"]
ACTION_WORDS   = ["will","immediately","escalat","priorit","investigat","review","contact","follow","update","fix","resolv"]
GENERIC_PHRASES= ["thank you for reaching out","we have received your email","your satisfaction is our priority"]


def grade_classify(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    reward, info = 0.0, {}
    predicted, actual = action_data.get("category",""), ground_truth["category"]
    confidence = float(action_data.get("confidence", 0.5))
    reason = action_data.get("reason","")
    correct = predicted == actual
    if correct:
        reward += 0.6
        reward += 0.2 if confidence >= 0.8 else (0.1 if confidence >= 0.6 else 0)
    else:
        reward += CATEGORY_PROXIMITY.get((predicted, actual), 0.0)
        if {predicted, actual} == {"spam","urgent"}: reward -= 0.1
    if reason and len(reason) > 10:
        reward += 0.15
        if sum(1 for w in ["urgent","spam","promotional","normal","scam","phish","server","invoice"] if w in reason.lower()) >= 2:
            reward += 0.05
    info.update({"category_correct": correct, "predicted": predicted, "actual": actual, "confidence": confidence})
    return round(min(max(reward, 0.0), 1.0), 4), info


def grade_triage(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    reward, info = 0.0, {}
    pred_cat, actual_cat   = action_data.get("category",""), ground_truth["category"]
    pred_dept, actual_dept = action_data.get("department",""), ground_truth["department"]
    priority  = int(action_data.get("priority", 1))
    exp_pri   = ground_truth["priority"]
    pri_diff  = abs(priority - exp_pri)
    reason    = action_data.get("reason","")
    if pred_cat == actual_cat: reward += 0.25
    else: reward += CATEGORY_PROXIMITY.get((pred_cat, actual_cat), 0.0) * 0.5
    if pred_dept == actual_dept: reward += 0.40
    else: reward += DEPT_PROXIMITY.get((pred_dept, actual_dept), 0.0)
    reward += 0.25 if pri_diff == 0 else (0.15 if pri_diff == 1 else (0.05 if pri_diff == 2 else 0))
    if reason and len(reason) > 15:
        reward += 0.07
        if any(w in reason.lower() for w in ["billing","technical","support","sales","hr","urgent","priority"]): reward += 0.03
    info.update({"category_correct": pred_cat==actual_cat, "dept_correct": pred_dept==actual_dept,
                 "priority_diff": pri_diff, "predicted_dept": pred_dept, "actual_dept": actual_dept})
    return round(min(max(reward, 0.0), 1.0), 4), info


def grade_respond(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    reward, info = 0.0, {}
    body     = (action_data.get("body") or "").strip()
    subject  = (action_data.get("subject") or "").strip()
    resolved = action_data.get("resolved", False)
    bl       = body.lower()
    min_len  = ground_truth.get("min_response_length", 50)
    blen     = len(body)
    reward  += 0.15 if blen >= min_len*2 else (0.10 if blen >= min_len else (0.05 if blen >= min_len*0.5 else 0))
    req_kw = ground_truth.get("keywords_required",[])
    if req_kw:
        matched = sum(1 for kw in req_kw if kw.lower() in bl)
        reward += 0.30 * (matched / len(req_kw))
        info["keywords_matched"] = matched
    else: reward += 0.30
    pos = sum(1 for w in POSITIVE_TONE if w in bl)
    pro = sum(1 for w in PROFESSIONAL   if w in bl)
    reward += (0.08 if pos>=2 else 0.04 if pos>=1 else 0) + (0.07 if pro>=3 else 0.03 if pro>=1 else 0)
    act = sum(1 for w in ACTION_WORDS if w in bl)
    reward += 0.20 if act>=3 else (0.13 if act>=2 else (0.07 if act>=1 else 0))
    must_resolve = ground_truth.get("must_resolve", False)
    if must_resolve and resolved: reward += 0.10
    elif not must_resolve:        reward += 0.10
    if subject and len(subject)>5:
        reward += 0.03
        if subject.lower().startswith("re:"): reward += 0.02
    if sum(1 for p in GENERIC_PHRASES if p in bl) >= 2: reward -= 0.10
    if ground_truth.get("priority",1) >= 4 and blen < 100: reward -= 0.10
    info.update({"body_length": blen, "resolved": resolved, "tone": action_data.get("tone","")})
    return round(min(max(reward, 0.0), 1.0), 4), info


def grade_summarize(action_data: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
    reward, info = 0.0, {}
    summary   = (action_data.get("summary") or "").strip()
    urgency   = (action_data.get("urgency") or "").lower()
    action_req= (action_data.get("action_required") or "").strip()
    sl        = len(summary)
    reward   += 0.40 if sl >= 100 else (0.25 if sl >= 60 else (0.10 if sl >= 30 else 0))
    urgency_map = {5:"critical", 4:"high", 3:"medium", 2:"low", 1:"low"}
    exp_urgency = urgency_map.get(ground_truth.get("priority", 1), "medium")
    levels = ["low","medium","high","critical"]
    if urgency == exp_urgency: reward += 0.30
    elif urgency in levels and abs(levels.index(urgency)-levels.index(exp_urgency)) == 1: reward += 0.15
    if action_req and len(action_req) > 15:
        reward += 0.15
        if any(w in action_req.lower() for w in ["contact","resolv","escalat","review","refund","fix","respond"]): reward += 0.05
    req_kw = ground_truth.get("keywords_required",[])
    if req_kw:
        matched = sum(1 for kw in req_kw if kw.lower() in summary.lower())
        reward += 0.10 * (matched / len(req_kw))
    info.update({"summary_length": sl, "urgency": urgency, "expected_urgency": exp_urgency})
    return round(min(max(reward, 0.0), 1.0), 4), info
