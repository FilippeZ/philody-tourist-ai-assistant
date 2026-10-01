"""
Unit & Integration Test Suite for Intent Classifier, Pydantic Parsers & Mock APIs (test_tools_and_parsers.py).

Validates:
1. 🔧 Intent Classifier False-Negative Prevention on Complex & Compound Queries
2. 🔄 Strict Pydantic Output Parsers for all tool arguments with self-healing
3. ➕ Public Transit Mock API (OASA/STASY routing, schedules, accessibility)
4. ➕ Live Tickets & Availability Mock API (pricing, time slots, simulated reservations)
"""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import sys
from pathlib import Path


import sys
import io

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from orchestrator.agent import AthensTouristAgent, IntentType, UserState, extract_named_pois
from orchestrator.tool_parsers import (
    WeatherToolArgs,
    RAGQueryArgs,
    FeasibilityPlanningArgs,
    TransitToolArgs,
    TicketingToolArgs,
    WearableIoTToolArgs,
    weather_args_parser,
    rag_args_parser,
    feasibility_args_parser,
    transit_args_parser,
    ticketing_args_parser,
    wearable_args_parser,
    PydanticToolOutputParser,
)
from tools.transit import (
    get_transit_route,
    get_station_schedule,
    get_transit_alerts,
)
from tools.ticketing import (
    check_ticket_availability,
    get_ticket_pricing,
    simulate_ticket_reservation,
    ATHENS_TICKET_CATALOG,
)


def test_intent_classifier_false_negative_prevention():
    print("=" * 80)
    print("🧪 1. INTENT CLASSIFIER FALSE NEGATIVE VERIFICATION ON COMPLEX QUERIES")
    print("=" * 80)
    agent = AthensTouristAgent()

    # 1. Complex compound: Itinerary request + transit question -> MUST BE ITINERARY_REQUEST (NOT False Negative FACTUAL_QA)
    q1 = "Έχω 3 ώρες στην Αθήνα και θέλω να μάθω πώς πάω στο Μουσείο Ακρόπολης, φτιάξε μου ένα πρόγραμμα."
    intent1 = agent.detect_intent(q1)
    print(f"Query 1: '{q1}'\n -> Detected: {intent1}")
    assert intent1 == IntentType.ITINERARY_REQUEST.value, f"Expected itinerary_request, got {intent1}"

    # 2. Complex compound: Time budget + ticket question + itinerary -> MUST BE ITINERARY_REQUEST
    q2 = "Έχω 4 ώρες διαθέσιμες, πόσο κοστίζει το εισιτήριο και τι πρόγραμμα προτείνεις για το απόγευμα;"
    intent2 = agent.detect_intent(q2)
    print(f"Query 2: '{q2}'\n -> Detected: {intent2}")
    assert intent2 == IntentType.ITINERARY_REQUEST.value, f"Expected itinerary_request, got {intent2}"

    # 3. Complex compound during active itinerary: fatigue + question -> MUST BE REPLANNING
    agent.user_state.active_itinerary = {"feasible": True, "schedule": []}
    q3 = "Είμαστε κουρασμένοι, πώς πάμε στο πλησιέστερο καφέ και άλλαξε το πρόγραμμά μας;"
    intent3 = agent.detect_intent(q3)
    print(f"Query 3: '{q3}'\n -> Detected: {intent3}")
    assert intent3 == IntentType.REPLANNING.value, f"Expected replanning, got {intent3}"

    # 4. Complex compound during active itinerary: weather rain + info -> MUST BE REPLANNING
    q4 = "Βρέχει τώρα, τι ώρα κλείνει το μουσείο και τι προτείνεις να αλλάξουμε στο πλάνο;"
    intent4 = agent.detect_intent(q4)
    print(f"Query 4: '{q4}'\n -> Detected: {intent4}")
    assert intent4 == IntentType.REPLANNING.value, f"Expected replanning, got {intent4}"

    # 5. Multi-POI compound request with duration -> MUST BE ITINERARY_REQUEST
    agent.user_state.active_itinerary = None
    q5 = "Θέλω να επισκεφθώ την Ακρόπολη, το Μουσείο Ακρόπολης και το Εθνικό Αρχαιολογικό Μουσείο μέσα σε 2 ώρες (17:00 - 19:00)."
    intent5 = agent.detect_intent(q5)
    print(f"Query 5: '{q5}'\n -> Detected: {intent5}")
    assert intent5 == IntentType.ITINERARY_REQUEST.value, f"Expected itinerary_request, got {intent5}"

    # 6. Disambiguation check: both Acropolis and Acropolis Museum extracted
    pois = extract_named_pois(q5)
    print(f"Extracted POIs for Query 5: {pois}")
    assert "acropolis_hill" in pois and "acropolis_museum" in pois, "Both Acropolis hill and museum must be preserved"

    # 7. Public Transit Route Query
    q6 = "Πώς πάω από το Σύνταγμα στην Ακρόπολη με το μετρό;"
    intent6 = agent.detect_intent(q6)
    print(f"Query 6: '{q6}'\n -> Detected: {intent6}")
    assert intent6 == IntentType.TRANSIT_QUERY.value, f"Expected transit_query, got {intent6}"

    print("✅ Intent Classifier: 100% SUCCESS — Zero False Negatives on Complex Queries!\n")


def test_pydantic_output_parsers():
    print("=" * 80)
    print("🧪 2. STRICT PYDANTIC OUTPUT PARSERS & TOOL ARGUMENTS VERIFICATION")
    print("=" * 80)

    # 1. Weather Args Parser
    w_parsed = weather_args_parser.parse({"city": "Αθήνα", "mock_scenario": "βροχή"})
    print(f"Weather Args: city={w_parsed.city}, scenario={w_parsed.mock_scenario}")
    assert w_parsed.city == "Athens"
    assert w_parsed.mock_scenario == "rain"

    # Self-healing from natural language
    w_healed = weather_args_parser.parse("Κάνει καύσωνα 40 βαθμούς στην Αθήνα")
    assert w_healed.city == "Athens"
    assert w_healed.mock_scenario == "heatwave"

    # 2. RAG Args Parser
    r_parsed = rag_args_parser.parse({"query": "Παρθενώνας ιστορία", "top_k": 4})
    print(f"RAG Args: query='{r_parsed.query}', top_k={r_parsed.top_k}")
    assert r_parsed.query == "Παρθενώνας ιστορία"
    assert r_parsed.top_k == 4

    # 3. Feasibility Args Parser
    f_parsed = feasibility_args_parser.parse({
        "start_time": "14:30",
        "end_time": "18:30",
        "time_budget_hours": 4.0,
        "traveling_with_kids": True,
        "child_age": 8,
        "wheelchair_accessible": True,
        "preferred_pace": "χαλαρός ρυθμός",
    })
    print(f"Feasibility Args: start={f_parsed.start_time}, budget={f_parsed.time_budget_hours}h, pace={f_parsed.preferred_pace}, wheelchair={f_parsed.wheelchair_accessible}")
    assert f_parsed.start_time == "14:30"
    assert f_parsed.preferred_pace == "relaxed"
    assert f_parsed.wheelchair_accessible is True
    assert f_parsed.child_age == 8

    # 4. Transit Args Parser
    t_parsed = transit_args_parser.parse({
        "origin": "Σύνταγμα",
        "destination": "Μουσείο Ακρόπολης",
        "wheelchair_accessible": True,
    })
    print(f"Transit Args: from={t_parsed.origin}, to={t_parsed.destination}, wheelchair={t_parsed.wheelchair_accessible}")
    assert t_parsed.origin == "Σύνταγμα"
    assert t_parsed.destination == "Μουσείο Ακρόπολης"
    assert t_parsed.wheelchair_accessible is True

    # 5. Ticketing Args Parser
    tkt_parsed = ticketing_args_parser.parse({
        "poi_id": "Μουσείο Ακρόπολης",
        "time_slot": "10:00-11:00",
        "num_tickets": 2,
        "ticket_tier": "adult",
    })
    print(f"Ticketing Args: poi={tkt_parsed.poi_id}, slot={tkt_parsed.time_slot}, count={tkt_parsed.num_tickets}")
    assert tkt_parsed.poi_id == "acropolis_museum"
    assert tkt_parsed.num_tickets == 2

    # 6. JSON Schema Instruction Generation
    instructions = feasibility_args_parser.get_format_instructions()
    assert "properties" in instructions and "time_budget_hours" in instructions
    print("Format Instructions generated successfully via Pydantic model_json_schema()")

    print("✅ Pydantic Output Parsers: 100% SUCCESS — Type Safety & Self-Healing Validated!\n")


def test_public_transit_mock_api():
    print("=" * 80)
    print("🧪 3. PUBLIC TRANSIT MOCK API (OASA / STASY / METRO) VERIFICATION")
    print("=" * 80)

    # 1. Routing Syntagma -> Acropolis
    route = get_transit_route(origin="Σύνταγμα", destination="Ακρόπολη", wheelchair_accessible=True)
    print(f"Route: {route['origin']} -> {route['destination']}")
    print(f"Duration: {route['total_duration_mins']} mins | Distance: {route['total_distance_km']} km | Fare: {route['fare_eur']} €")
    assert route["total_duration_mins"] > 0
    assert route["fare_eur"] == 1.20
    assert len(route["legs"]) >= 2
    assert route["wheelchair_accessible"] is True

    # 2. Station live schedule
    sched = get_station_schedule("Syntagma")
    print(f"Station: {sched['station_name']} | Lines: {sched['lines']}")
    print(f"Live arrivals: {len(sched['live_arrivals'])} trains")
    assert len(sched["live_arrivals"]) >= 2
    assert sched["wheelchair_accessible"] is True

    # 3. Transit network alerts
    alerts = get_transit_alerts()
    print(f"Transit Network Status: {alerts['status']}")
    assert alerts["status"] == "normal_operation"
    assert len(alerts["alerts"]) >= 1

    print("✅ Public Transit Mock API: 100% SUCCESS — Routing, Lines & Accessibility Verified!\n")


def test_live_tickets_and_availability_mock_api():
    print("=" * 80)
    print("🧪 4. LIVE TICKETS & AVAILABILITY MOCK API VERIFICATION")
    print("=" * 80)

    # 1. Ticket Pricing Catalog
    pricing = get_ticket_pricing("acropolis_hill")
    print(f"POI: {pricing['poi_name']} | Regular: {pricing['regular_price_eur']} € | Reduced: {pricing['reduced_price_eur']} €")
    print(f"Combined Ticket Eligible: {pricing['combined_ticket_eligible']}")
    assert pricing["found"] is True
    assert pricing["regular_price_eur"] == 20.0
    assert pricing["reduced_price_eur"] == 10.0
    assert pricing["combined_ticket_eligible"] is True

    # Free entry catalog
    garden_pricing = get_ticket_pricing("national_garden")
    assert garden_pricing["is_free"] is True
    assert garden_pricing["regular_price_eur"] == 0.0

    # 2. Slot Availability Check
    avail = check_ticket_availability("acropolis_museum", time_slot="10:00-11:00")
    print(f"Availability for {avail['poi_name']}: {len(avail['slots'])} time slots")
    assert len(avail["slots"]) >= 10
    sold_out_slots = [s for s in avail["slots"] if s["status"] == "sold_out"]
    assert len(sold_out_slots) >= 1
    print(f"Sold-out peak slot detected: {sold_out_slots[0]['time_slot']}")

    # 3. Simulated Ticket Reservation
    res = simulate_ticket_reservation(
        poi_id="acropolis_museum",
        visit_date="2026-10-05",
        time_slot="14:00-15:00",
        num_tickets=2,
        ticket_tier="adult",
        visitor_name="Maria Papadopoulou",
    )
    print(f"Booking Reference: {res['booking_reference']} | Total: {res['total_amount_eur']} €")
    print(f"QR Token: {res['qr_token']} | Status: {res['status']}")
    assert res["status"] == "CONFIRMED"
    assert res["total_amount_eur"] == 30.0  # 15€ * 2
    assert "ATH-" in res["booking_reference"]
    assert "QR-" in res["qr_token"]

    print("✅ Live Tickets & Availability Mock API: 100% SUCCESS — Catalog, Quotas & Booking Verified!\n")


def main():
    print("\n🚀 ΕΚΚΙΝΗΣΗ ΠΛΗΡΟΥΣ ΕΛΕΓΧΟΥ: INTENT CLASSIFIER, PYDANTIC PARSERS & MOCK APIS\n")
    test_intent_classifier_false_negative_prevention()
    test_pydantic_output_parsers()
    test_public_transit_mock_api()
    test_live_tickets_and_availability_mock_api()
    print("=" * 80)
    print("🎉 ΟΛΟΙ ΟΙ ΕΛΕΓΧΟΙ ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΜΕ 100% ΕΠΙΤΥΧΙΑ! ΠΛΗΡΗΣ ΚΑΛΥΨΗ ΑΠΑΙΤΗΣΕΩΝ.")
    print("=" * 80)


if __name__ == "__main__":
    main()
