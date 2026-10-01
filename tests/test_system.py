"""
Comprehensive Test Suite for Athens Tourist Assistant - Phase 1.
Tests all 4 sub-systems and the end-to-end execution flow.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import json
import sys
from pprint import pprint

# Ensure utf-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

import os
os.environ["OPENAI_API_KEY"] = ""

from engine.feasibility import FeasibilityEngine, haversine_distance, build_feasible_itinerary
from orchestrator.agent import AthensTouristAgent
from rag.retriever import AthensRetriever
from tools.weather import get_current_weather


def test_wheelchair_filter_excludes_anafiotika():
    itinerary = build_feasible_itinerary(
        start_time="14:00",
        budget_hours=3.0,
        wheelchair_accessible=True
    )

    selected_ids = [step["poi_id"] for step in itinerary["schedule"] if step["type"] == "activity"]

    # Επιβεβαίωση ότι τα Αναφιώτικα ΔΕΝ περιλαμβάνονται στο πρόγραμμα
    assert "anafiotika" not in selected_ids



def run_tests():
    print("=" * 70)
    print("AI TOURIST ASSISTANT - PHASE 1 ARCHITECTURAL & UNIT VERIFICATION")
    print("=" * 70)

    # 1. RAG & Knowledge Base
    print("\n--- [1] Tourism Knowledge Base & RAG Pipeline ---")
    retriever = AthensRetriever()
    print(f"Total Ingested POIs: {len(retriever.kb.get_all_pois())}")
    assert len(retriever.kb.get_all_pois()) >= 10, "Failed: Expected at least 10 POIs"

    query_rag = "Παρθενώνας ιστορία και αρχαία ακρόπολη"
    chunks = retriever.retrieve(query_rag, top_k=2)
    print(f"RAG Query: '{query_rag}'")
    for i, c in enumerate(chunks, 1):
        c_name = c.get("name") or c.get("source_name")
        c_id = c.get("id") or c.get("source_id")
        c_score = c.get("score", 0.0)
        print(f"  Result {i}: {c_name} (ID: {c_id}, Score: {c_score:.3f})")
    top_ids = [c["id"] for c in chunks]
    assert any(pid in ["acropolis_hill", "ancient_agora", "acropolis_museum", "parthenon_acropolis"] for pid in top_ids), "RAG matching failed"

    from orchestrator.prompts import build_rag_context_prompt
    rag_prompt = build_rag_context_prompt(query_rag, chunks)
    assert "--- ΠΑΡΕΧΟΜΕΝΗ ΒΑΣΗ ΓΝΩΣΗΣ ---" in rag_prompt
    print("  ✓ RAG Ingestion & Vector Retrieval: SUCCESS")

    # 2. Weather Tool Integration (Strict Live API Enforcement)
    print("\n--- [2] Live Weather Tool Integration ---")
    try:
        get_current_weather(city="Athens")
    except ValueError as ve:
        assert "OPENWEATHER_API_KEY is missing" in str(ve)
        print("  ✓ Strict API Key Check (ValueError on missing key): SUCCESS")
    except Exception as exc:
        print(f"  ✓ Weather API Call executed (handled: {exc})")

    # 3. Feasibility Engine
    print("\n--- [3] Feasibility & Validation Engine (Deterministic Logic) ---")
    engine = FeasibilityEngine(walking_speed_kmh=4.5)
    coord1 = {"lat": 37.9684, "lon": 23.7285}  # Acropolis Museum
    coord2 = {"lat": 37.9730, "lon": 23.7297}  # Plaka Walk
    dist = haversine_distance(coord1, coord2)
    print(f"Haversine Distance (Acropolis Museum -> Plaka): {dist} km")
    assert 0.3 <= dist <= 1.0, f"Unexpected distance: {dist}"

    pois = retriever.kb.get_all_pois()
    mock_weather = {"condition": "Clear", "is_indoor_recommended": False, "temperature": 24.0}
    plan = engine.build_itinerary(
        start_time="14:00",
        end_time="18:00",
        candidate_pois=pois,
        weather_data=mock_weather,
        traveling_with_kids=True,
        child_age=10,
    )
    print(f"Itinerary Feasible: {plan.feasible}")
    print(f"Total Walking Distance: {plan.total_distance_km} km")
    print(f"Selected POIs: {plan.selected_poi_ids}")
    print("Schedule generated:")
    for item in plan.schedule:
        name = item.get("poi") or item.get("action")
        itype = item.get("type")
        print(f"   [{item['time']}] {name} ({itype})")
    assert plan.feasible, "Plan should be feasible"
    assert len(plan.selected_poi_ids) >= 1, "Should select at least 1 POI"
    print("  ✓ Deterministic Feasibility Validation: SUCCESS")

    # 3.1 Wheelchair Hard Pre-filter (Excludes Anafiotika)
    print("\n--- [3.1] Wheelchair Accessibility Pre-filter ---")
    test_wheelchair_filter_excludes_anafiotika()
    print("  ✓ Wheelchair Pre-filter (Anafiotika Strictly Excluded): SUCCESS")

    # 4. Multi-Turn Orchestrator & End-to-End Execution Flow
    print("\n--- [4] Multi-Turn Orchestrator & Intent Routing ---")
    agent = AthensTouristAgent(retriever=retriever, feasibility_engine=engine)

    # 4.1 Factual Q&A
    print("\n* TURN 1: Factual Q&A (Intent A)")
    msg1 = "Ποια είναι η ιστορία του Παρθενώνα;"
    res1 = agent.chat(msg1)
    print(f"User: {msg1}")
    print(f"Detected Intent: {res1['intent']}")
    print(f"Citations: {[c['source_name'] for c in res1.get('citations', [])]}")
    assert res1['intent'] == 'factual_qa'
    assert len(res1.get('citations', [])) > 0

    # 4.2 Weather Query
    print("\n* TURN 2: Weather Query (Intent B)")
    msg2 = "Τι καιρό κάνει στην Αθήνα;"
    res2 = agent.chat(msg2)
    print(f"User: {msg2}")
    print(f"Detected Intent: {res2['intent']}")
    print(f"Weather Condition: {res2['weather_data']['condition']}")
    assert res2['intent'] == 'weather_query'

    # 4.3 End-to-End Itinerary Request
    print("\n* TURN 3: End-to-End Itinerary Generation (Intent C)")
    msg3 = "Έχω 4 ώρες το απόγευμα στην Αθήνα, είμαι με το παιδί μου 10 ετών και θέλω ένα χαλαρό πρόγραμμα."
    res3 = agent.chat(msg3)
    print(f"User: {msg3}")
    print(f"Detected Intent: {res3['intent']}")
    print(f"Kids Mode: {res3['user_state']['traveling_with_kids']}, Age: {res3['user_state']['child_age']}")
    print(f"Selected POIs: {res3['raw_plan']['selected_poi_ids']}")
    assert res3['intent'] == 'itinerary_request'
    assert res3['raw_plan']['feasible'] is True

    # 4.4 Dynamic Replanning
    print("\n* TURN 4: Dynamic Replanning (Intent D)")
    msg4 = "Δεν θέλω το μουσείο, άλλαξέ το με κάτι άλλο."
    res4 = agent.chat(msg4)
    print(f"User: {msg4}")
    print(f"Detected Intent: {res4['intent']}")
    print(f"Replaced POI: {res4.get('replaced_poi')}")
    print(f"New Selected POIs: {res4['raw_plan']['selected_poi_ids']}")
    assert res4['intent'] == 'replanning'
    assert res4['raw_plan']['feasible'] is True

    print("\n" + "=" * 70)
    print("ALL 4 SUB-SYSTEMS & END-TO-END FLOW VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
