"""
Quality Testing Suite for Athens AI Tourist Assistant (4 Use Cases).
Validates all 4 core scenarios:
  1. Factual Q&A (RAG Only, bypass Weather & Feasibility)
  2. Dynamic Weather Replanning (Weather trigger, outdoor exclusion -> indoor replacement)
  3. Feasibility Constraint Rejection (Excessive request 370m > 120m, feasible fallback)
  4. Multi-turn Accessibility Update (Wheelchair flag, exclusion of stairs/Anafiotika)
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import sys
import io

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
os.environ["OPENAI_API_KEY"] = ""

from orchestrator.agent import AthensTouristAgent, IntentType, UserState
from rag.retriever import AthensRAGRetriever
from engine.feasibility import FeasibilityEngine
from tools.weather import get_current_weather


def run_use_case_1():
    print("=" * 80)
    print("1️⃣ USE CASE 1: Απλό Ιστορικό / Πραγματολογικό Ερώτημα (Intent A - RAG Only)")
    print("=" * 80)
    agent = AthensTouristAgent()
    query = "Ποια είναι η ιστορία του Ναού του Ηφαίστου στην Αρχαία Αγορά και ποιες είναι οι ώρες λειτουργίας;"
    print(f"👤 Χρήστης: \"{query}\"")

    res = agent.chat(query)
    intent = res["intent"]
    reply = res["reply"]
    citations = res["citations"]

    print(f"🤖 Αναγνωρισμένο Intent: {intent}")
    assert intent == IntentType.FACTUAL_QA.value, f"Expected {IntentType.FACTUAL_QA.value}, got {intent}"

    # Verify RAG retrieved the Temple of Hephaestus
    matched_ids = [c["source_id"] for c in citations]
    print(f"📚 Ανακτηθείσες Πηγές: {matched_ids}")
    assert "hephaestus_temple" in matched_ids, "hephaestus_temple should be in retrieved citations"

    # Verify opening hours and historical context in reply
    assert "08:00" in reply and "20:00" in reply, "Reply must contain opening hours 08:00 - 20:00"
    assert "Ηφαίστου" in reply or "Ηφαιστείο" in reply, "Reply must contain historical information about Hephaestus"

    # Verify that Feasibility Engine did NOT generate an itinerary
    assert res["raw_plan"] is None, "Feasibility Engine must be bypassed for factual Q&A"

    print("\n📝 Απάντηση Συστήματος:")
    print(reply[:350] + "...\n")
    print("✅ USE CASE 1: ΕΠΙΤΥΧΙΑ (RAG Only, Bypass Engines, Grounded Citations)\n")


def run_use_case_2():
    print("=" * 80)
    print("2️⃣ USE CASE 2: Δυναμικός Ανασχεδιασμός λόγω Καιρού (Intent C - Weather Triggered Replanning)")
    print("=" * 80)
    agent = AthensTouristAgent()

    # Προετοιμασία ενεργού δρομολογίου με Λόφο Λυκαβηττού στις 17:00
    lycabettus_poi = agent.retriever.get_poi("lycabettus_hill")
    agent.user_state.start_time = "17:00"
    agent.user_state.time_budget_hours = 2.0
    agent.user_state.active_itinerary = {
        "feasible": True,
        "start_time": "17:00",
        "end_time": "19:00",
        "actual_end_time": "18:00",
        "total_distance_km": 0.0,
        "weather_adjusted": False,
        "schedule": [
            {
                "type": "activity",
                "poi_id": "lycabettus_hill",
                "poi_name": lycabettus_poi["name"],
                "env_type": "outdoor",
                "time_slot": "17:00-18:00",
                "duration_mins": 60,
            }
        ],
        "selected_poi_ids": ["lycabettus_hill"],
    }

    query = "Έχουμε κανονίσει περίπατο στον Λόφο Λυκαβηττού στις 17:00, αλλά φαίνεται πως συννέφιασε απότομα. Τι να κάνουμε;"
    print(f"👤 Χρήστης: \"{query}\"")

    res = agent.chat(query)
    intent = res["intent"]
    reply = res["reply"]
    plan = res["raw_plan"]

    print(f"🤖 Αναγνωρισμένο Intent: {intent}")
    assert intent == IntentType.REPLANNING.value, f"Expected {IntentType.REPLANNING.value}, got {intent}"

    # Verify weather triggered adjustment
    assert plan["weather_adjusted"] is True, "Itinerary should be weather_adjusted"
    
    # Verify Lycabettus is excluded
    assert "lycabettus_hill" not in plan["selected_poi_ids"], "Lycabettus hill (outdoor) must be excluded"
    assert "lycabettus_hill" in agent.user_state.blacklisted_poi_ids, "Lycabettus hill must be blacklisted"

    # Verify indoor replacement (e.g. Cycladic Art Museum)
    selected_env_types = [
        s["env_type"] for s in plan["schedule"] if s["type"] == "activity"
    ]
    print(f"🏛️ Επιλεγμένα POIs μετά τον ανασχεδιασμό: {plan['selected_poi_ids']}")
    print(f"🌦️ Τύπος περιβάλλοντος επιλεγμένων POIs: {selected_env_types}")
    assert all(env == "indoor" for env in selected_env_types), "All selected POIs must be indoor during bad weather"
    assert "cycladic_art_museum" in plan["selected_poi_ids"] or any("museum" in pid for pid in plan["selected_poi_ids"]), "An indoor museum should be selected"

    print("\n📝 Απάντηση Συστήματος:")
    print(reply[:380] + "...\n")
    print("✅ USE CASE 2: ΕΠΙΤΥΧΙΑ (Weather Tool Call, Lycabettus Excluded, Indoor Replacement Validated)\n")


def run_use_case_3():
    print("=" * 80)
    print("3️⃣ USE CASE 3: Ανέφικτο Αίτημα & Αυστηρός Έλεγχος Περιορισμών (Constraint Rejection)")
    print("=" * 80)
    agent = AthensTouristAgent()
    query = "Θέλω να επισκεφθώ την Ακρόπολη, το Μουσείο Ακρόπολης και το Εθνικό Αρχαιολογικό Μουσείο μέσα σε 2 ώρες (17:00 - 19:00)."
    print(f"👤 Χρήστης: \"{query}\"")

    res = agent.chat(query)
    intent = res["intent"]
    reply = res["reply"]
    plan = res["raw_plan"]

    print(f"🤖 Αναγνωρισμένο Intent: {intent}")
    assert intent == IntentType.ITINERARY_REQUEST.value, f"Expected {IntentType.ITINERARY_REQUEST.value}, got {intent}"

    print(f"⏱️ Time Budget: {agent.user_state.time_budget_hours * 60} λεπτά (17:00 - 19:00)")
    print(f"🛑 Constraint Rejected: {plan.get('constraint_rejected')}")
    print(f"⌛ Total Needed Minutes for all 3 POIs: {plan.get('total_needed_mins')}")

    # Check constraint rejection
    assert plan.get("constraint_rejected") is True, "Plan must flag constraint_rejected = True"
    assert plan.get("total_needed_mins") >= 330, "Total needed time must be at least 330-370 mins"

    # Verify fallback includes only feasible subset (only 1 POI fits in 2 hours)
    selected = plan["selected_poi_ids"]
    print(f"🎯 Επιλεγμένα POIs στο εφικτό εναλλακτικό πλάνο: {selected}")
    assert len(selected) == 1, f"Expected exactly 1 POI in 2-hour budget, got {len(selected)}"
    assert selected[0] in ["acropolis_museum", "acropolis_hill"], "Selected POI should be Acropolis Museum or Hill"

    # Verify synthesis explains why the request is infeasible
    assert "δεν είναι εφικτό" in reply or "ανέφικτο" in reply.lower(), "Reply must state the request is not feasible"

    print("\n📝 Απάντηση Συστήματος:")
    print(reply[:400] + "...\n")
    print("✅ USE CASE 3: ΕΠΙΤΥΧΙΑ (Strict Feasibility Constraint Rejection, 370m > 120m, 1-POI Fallback)\n")


def run_use_case_4():
    print("=" * 80)
    print("4️⃣ USE CASE 4: Multi-turn Context & Ενημέρωση Προσβασιμότητας (Wheelchair Update)")
    print("=" * 80)
    agent = AthensTouristAgent()

    # --- TURN 1 ---
    turn1_query = "Φτιάξε μου ένα πρόγραμμα 3 ωρών για το απόγευμα στην Πλάκα."
    print(f"👤 Turn 1: \"{turn1_query}\"")
    res1 = agent.chat(turn1_query)
    plan1 = res1["raw_plan"]
    print(f"🤖 Turn 1 Intent: {res1['intent']}")
    print(f"📍 Turn 1 Selected POIs: {plan1['selected_poi_ids']}")
    assert agent.user_state.active_itinerary is not None, "Turn 1 must produce active itinerary"
    assert any("plaka" in pid or "anafiotika" in pid for pid in plan1["selected_poi_ids"]), "Plaka POIs expected"

    # --- TURN 2 ---
    turn2_query = "Ξέχασα να σου πω, είμαι με αναπηρικό αμαξίδιο και δεν μπορώ να ανεβαίνω σκαλιά."
    print(f"\n👤 Turn 2: \"{turn2_query}\"")
    res2 = agent.chat(turn2_query)
    intent2 = res2["intent"]
    reply2 = res2["reply"]
    plan2 = res2["raw_plan"]

    print(f"🤖 Turn 2 Intent: {intent2}")
    assert intent2 == IntentType.REPLANNING.value, f"Expected {IntentType.REPLANNING.value}, got {intent2}"

    # Verify state mutation
    assert agent.user_state.wheelchair_accessible is True, "wheelchair_accessible must be True"
    print(f"♿ UserState wheelchair_accessible: {agent.user_state.wheelchair_accessible}")

    # Verify Anafiotika (stairs) is excluded
    assert "anafiotika" not in plan2["selected_poi_ids"], "Anafiotika (stairs) must be excluded"
    print(f"🚫 Anafiotika excluded from Turn 2 schedule: {'anafiotika' not in plan2['selected_poi_ids']}")

    # Verify all selected POIs in Turn 2 are wheelchair accessible
    for pid in plan2["selected_poi_ids"]:
        poi = agent.retriever.get_poi(pid)
        assert poi.get("wheelchair_accessible") is True, f"POI {pid} is not wheelchair accessible!"
    print(f"✅ All Turn 2 POIs are wheelchair accessible: {plan2['selected_poi_ids']}")

    # Verify synthesis mentions wheelchair accessibility adaptation
    assert "αμαξίδιο" in reply2.lower() or "προσβασιμότητα" in reply2.lower(), "Reply must address accessibility"

    print("\n📝 Απάντηση Συστήματος (Turn 2):")
    print(reply2[:400] + "...\n")
    print("✅ USE CASE 4: ΕΠΙΤΥΧΙΑ (Multi-turn Context Maintained, Wheelchair State Updated, Anafiotika Excluded)\n")


def main():
    print("\n🚀 ΕΚΚΙΝΗΣΗ QUALITY TESTING ΓΙΑ ΤΑ 4 ΣΕΝΑΡΙΑ ΧΡΗΣΗΣ ΤΟΥ ΣΥΣΤΗΜΑΤΟΣ\n")
    run_use_case_1()
    run_use_case_2()
    run_use_case_3()
    run_use_case_4()
    print("=" * 80)
    print("🎉 ΟΛΑ ΤΑ 4 USE CASES ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΜΕ 100% ΕΠΙΤΥΧΙΑ! ΠΛΗΡΗΣ ΕΠΑΛΗΘΕΥΣΗ ΠΟΙΟΤΗΤΑΣ.")
    print("=" * 80)


if __name__ == "__main__":
    main()
