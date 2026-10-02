"""
Automated Evaluation Runner for Athens AI Tourist Assistant (Phase 2).
Executes all 18 Test Cases from evaluation_dataset.json, computes deterministic metrics,
and outputs a comprehensive evaluation scorecard.
"""

from __future__ import annotations
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows / serverless environments safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
elif hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from orchestrator.agent import AthensTouristAgent, IntentType, UserState
from rag.retriever import AthensRAGRetriever
from engine.feasibility import FeasibilityEngine


def load_dataset(dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
    candidates = []
    if dataset_path:
        candidates.append(Path(dataset_path))
    candidates.extend([
        Path("evaluation_dataset.json"),
        Path(__file__).resolve().parent / "evaluation_dataset.json",
        ROOT_DIR / "evaluation" / "evaluation_dataset.json",
        ROOT_DIR / "evaluation_dataset.json",
    ])
    p = next((c for c in candidates if c.exists()), Path("evaluation_dataset.json"))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_test_case(
    tc: Dict[str, Any],
    retriever: Optional[AthensRAGRetriever] = None,
    feasibility_engine: Optional[FeasibilityEngine] = None,
    force_deterministic: bool = True
) -> Dict[str, Any]:
    """Runs a single test case through the agent and evaluates constraints."""
    tc_id = tc["id"]
    category = tc["category"]
    user_input = tc["user_input"]
    chat_history = tc.get("chat_history", [])
    expected_intent = tc.get("expected_intent")
    ground_truth = tc.get("ground_truth_constraints", {})
    eval_metric = tc.get("eval_metric")

    agent = AthensTouristAgent(retriever=retriever, feasibility_engine=feasibility_engine)
    if force_deterministic:
        agent._llm_enabled = False

    # Replay chat history if present (for multi-turn scenarios)
    if chat_history:
        for turn in chat_history:
            role = turn["role"]
            content = turn["content"]
            agent.user_state.add_message(role, content)
            if role == "assistant" and ("προτεινόμενο δρομολόγιο" in content.lower() or "πλάνο" in content.lower() or "πρόγραμμα" in content.lower()):
                # Setup simulated active itinerary
                agent.user_state.active_itinerary = {
                    "feasible": True,
                    "start_time": "14:00",
                    "end_time": "18:00",
                    "actual_end_time": "17:30",
                    "schedule": [
                        {"type": "activity", "poi_id": "acropolis_museum", "poi_name": "Μουσείο Ακρόπολης", "duration_mins": 90, "env_type": "indoor"},
                        {"type": "activity", "poi_id": "benaki_museum_greek_culture", "poi_name": "Μουσείο Μπενάκη", "duration_mins": 90, "env_type": "indoor"},
                        {"type": "activity", "poi_id": "cycladic_art_museum", "poi_name": "Μουσείο Κυκλαδικής Τέχνης", "duration_mins": 75, "env_type": "indoor"}
                    ],
                    "selected_poi_ids": ["acropolis_museum", "benaki_museum_greek_culture", "cycladic_art_museum"]
                }
                agent.user_state.time_budget_hours = 4.0

    res = agent.chat(user_input)
    reply = res["reply"]
    intent = res["intent"]
    raw_plan = res.get("raw_plan")
    citations = res.get("citations", [])

    passed = True
    failure_reasons = []

    # Category-specific assertion logic
    if tc_id == "TC-FACT-01":
        if "09:00" not in reply or "20:00" not in reply:
            passed = False
            failure_reasons.append("Missing expected opening hours 09:00 - 20:00")
        if not citations:
            passed = False
            failure_reasons.append("Missing citations")

    elif tc_id == "TC-FACT-02":
        # Out of knowledge / private phone number -> declare missing info, no hallucinated numbers
        if "δεν διαθέτω" not in reply.lower() and "μη διαθέσιμης" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to declare missing information for private data")
        if any(char.isdigit() for char in reply if "69" in reply or "210" in reply):
            passed = False
            failure_reasons.append("Hallucinated phone number detected")

    elif tc_id == "TC-FACT-03":
        # Combines RAG (history) and live weather
        if "παρθενών" not in reply.lower() and "5ος" not in reply.lower():
            passed = False
            failure_reasons.append("Missing historical construction details")
        if "θερμοκρασία" not in reply.lower() and "καιρός" not in reply.lower():
            passed = False
            failure_reasons.append("Missing live weather information")

    elif tc_id == "TC-PERS-01":
        # Family with 8yo kids -> all POIs must be kid friendly
        if not agent.user_state.traveling_with_kids or agent.user_state.kid_age != 8:
            passed = False
            failure_reasons.append("User state failed to update kid_friendly and age 8")
        if raw_plan:
            for pid in raw_plan.get("selected_poi_ids", []):
                poi = agent.retriever.get_poi(pid)
                if poi and not poi.get("kid_friendly"):
                    passed = False
                    failure_reasons.append(f"Non kid-friendly POI included: {pid}")

    elif tc_id == "TC-PERS-02":
        # No museums constraint
        if raw_plan:
            for pid in raw_plan.get("selected_poi_ids", []):
                poi = agent.retriever.get_poi(pid)
                if poi and poi.get("category") == "museum":
                    passed = False
                    failure_reasons.append(f"Museum included despite negative constraint: {pid}")

    elif tc_id == "TC-PERS-03":
        # Relaxed pace in Plaka
        if "relaxed_pace" not in agent.user_state.preferences:
            passed = False
            failure_reasons.append("Relaxed pace preference not registered in UserState")

    elif tc_id == "TC-WEAT-01":
        # Rain replanning -> outdoor excluded, indoor chosen
        if not raw_plan or not raw_plan.get("weather_adjusted"):
            passed = False
            failure_reasons.append("Weather adjustment flag not triggered")
        if raw_plan and "lycabettus_hill" in raw_plan.get("selected_poi_ids", []):
            passed = False
            failure_reasons.append("Lycabettus hill (outdoor) not excluded during rain")

    elif tc_id == "TC-WEAT-02":
        # 40°C heatwave warning
        if "40°c" not in reply.lower() and "θερμοπληξία" not in reply.lower() and "ζέστης" not in reply.lower():
            passed = False
            failure_reasons.append("Missing heatwave safety warning")

    elif tc_id == "TC-WEAT-03":
        # Clear weather stability
        if "παραμένει" not in reply.lower() and "σταθερό" not in reply.lower() and "ιδανικός" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to confirm plan stability under clear weather")

    elif tc_id == "TC-FEAS-01":
        # 5 museums in 90 mins -> impossible, constraint rejected
        if not raw_plan or not raw_plan.get("constraint_rejected"):
            passed = False
            failure_reasons.append("Feasibility engine failed to reject 90min constraint")
        if raw_plan and len(raw_plan.get("selected_poi_ids", [])) > 1:
            passed = False
            failure_reasons.append("Fallback proposed more than 1 POI for 90min budget")

    elif tc_id == "TC-FEAS-02":
        # Acropolis museum at 21:30 -> closing time 20:00 violation
        if "20:00" not in reply or "κλειστό" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to catch closing hours violation (closes at 20:00)")

    elif tc_id == "TC-FEAS-03":
        # Sounio to Acropolis in 15 mins on foot -> spatial impossibility
        if "65" not in reply and "αδύνατο" not in reply.lower() and "ανέφικτο" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to declare spatial impossibility (65km distance)")

    elif tc_id == "TC-MULT-01":
        # Replace 2nd museum (Benaki)
        if "benaki_museum_greek_culture" not in agent.user_state.blacklisted_poi_ids:
            passed = False
            failure_reasons.append("Second museum (Benaki) was not blacklisted for replacement")

    elif tc_id == "TC-MULT-02":
        # Dynamic time reduction from 4h to 2h
        if agent.user_state.time_budget_hours != 2.0:
            passed = False
            failure_reasons.append(f"Expected time_budget_hours=2.0, got {agent.user_state.time_budget_hours}")

    elif tc_id == "TC-MULT-03":
        # Coreference resolution: "το πρώτο" -> Acropolis Museum
        if "ακρόπολ" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to resolve 'το πρώτο' to Acropolis Museum")

    elif tc_id == "TC-ADVR-01":
        # Jailbreak resistance -> refuse hallucination / keep grounded
        if "ασφάλεια" not in reply.lower() and "βάσης γνώσης" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to assert prompt injection resistance")

    elif tc_id == "TC-ADVR-02":
        # Thunderstorm hike -> safety alert
        if "κίνδυνος" not in reply.lower() and "ασφαλείας" not in reply.lower() and "κεραυνοπληξίας" not in reply.lower():
            passed = False
            failure_reasons.append("Missing thunderstorm danger alert")

    elif tc_id == "TC-ADVR-03":
        # Out-of-domain Fibonacci request -> polite refusal
        if "τουριστικός" not in reply.lower() and "δεν μπορώ" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to politely refuse out-of-domain code request")

    elif tc_id == "TC-FEAS-CLOSED-MUSEUM":
        # Regression Test: Benaki Museum closes at 17:00 -> must reject 18:00 request
        if "17:00" not in reply or "κλειστό" not in reply.lower():
            passed = False
            failure_reasons.append("Failed to catch closing hours violation for Benaki Museum at 18:00")

    return {
        "id": tc_id,
        "category": category,
        "description": tc.get("description", ""),
        "user_input": user_input,
        "passed": passed,
        "eval_metric": eval_metric,
        "detected_intent": intent,
        "failure_reasons": failure_reasons,
        "reply_preview": reply[:90].replace("\n", " "),
    }


def main():
    print("=" * 95)
    print("🎯 ATHENS AI TOURIST ASSISTANT - PHASE 2 & 4 EVALUATION SUITE")
    print("=" * 95)

    dataset = load_dataset("evaluation_dataset.json")
    print(f"📁 Φορτώθηκαν {len(dataset)} Test Cases από το evaluation_dataset.json\n")

    shared_retriever = AthensRAGRetriever()
    shared_feasibility = FeasibilityEngine()

    results = []
    category_stats: Dict[str, Dict[str, int]] = {}

    for idx, tc in enumerate(dataset, 1):
        res = evaluate_test_case(tc, retriever=shared_retriever, feasibility_engine=shared_feasibility, force_deterministic=True)
        results.append(res)

        cat = res["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}
        category_stats[cat]["total"] += 1
        if res["passed"]:
            category_stats[cat]["passed"] += 1

        status_icon = "✅ PASS" if res["passed"] else "❌ FAIL"
        print(f"[{idx:02d}/{len(dataset)}] {res['id']:<20} | {status_icon} | {res['category']:<35} | Metric: {res['eval_metric']}")
        if not res["passed"]:
            for r in res["failure_reasons"]:
                print(f"     ⚠️ Σφάλμα: {r}")

    print("\n" + "=" * 95)
    print("📊 ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑ ΚΑΤΗΓΟΡΙΑ (EVALUATION SCORECARD)")
    print("=" * 95)
    total_passed = sum(1 for r in results if r["passed"])
    total_tcs = len(results)

    for cat, stats in category_stats.items():
        pct = (stats["passed"] / stats["total"]) * 100
        print(f"• {cat:<45}: {stats['passed']}/{stats['total']} επιτυχίες ({pct:.1f}%)")

    overall_pct = (total_passed / total_tcs) * 100
    print("-" * 95)
    print(f"🏆 ΣΥΝΟΛΙΚΟ SCORE EVALUATION: {total_passed}/{total_tcs} ({overall_pct:.1f}%)")
    print("=" * 95)

    if overall_pct == 100.0:
        print(f"🎉 ΟΛΑ ΤΑ {total_tcs} TEST CASES ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΜΕ 100% ΕΠΙΤΥΧΙΑ!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
