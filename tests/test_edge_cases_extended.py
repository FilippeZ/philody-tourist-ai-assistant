"""
tests/test_edge_cases_extended.py

10 Additional edge-case tests covering:
  1.  haversine_distance  — Αθήνα → Πειραιάς (γνωστή απόσταση)
  2.  time_to_minutes / minutes_to_time — round-trip accuracy
  3.  filter_candidate_pois — wheelchair filter
  4.  filter_candidate_pois — blacklist category
  5.  filter_candidate_pois — blacklist tags
  6.  optimize_itinerary_duration — zero remaining (no change)
  7.  optimize_itinerary_duration — over-budget (negative remaining, no change)
  8.  calculate_total_mins — dict payload with "schedule" key
  9.  calculate_total_mins — empty list returns 0
  10. DELIMITED_NATURAL_SYNTHESIS_PROMPT — BOS/EOS tokens present
  + Bonus: render_natural_synthesis — EU AI Act disclaimer always appended
"""

import sys
import unittest

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
os.environ["OPENAI_API_KEY"] = ""

from engine.feasibility import (
    haversine_distance,
    time_to_minutes,
    minutes_to_time,
    filter_candidate_pois,
    optimize_itinerary_duration,
    calculate_total_mins,
)
from orchestrator.prompts import (
    NATURAL_SYNTHESIS_PROMPT,
    DELIMITED_NATURAL_SYNTHESIS_PROMPT,
)
from orchestrator.agent import render_natural_synthesis, handle_user_request


# ---------------------------------------------------------------------------
# Test 1 — haversine_distance: Αθήνα ↔ Πειραιάς
# ---------------------------------------------------------------------------
class TestHaversineDistance(unittest.TestCase):
    def test_athens_to_piraeus_roughly_8km(self):
        """Γνωστή γεωγραφική απόσταση Αθήνα-Πειραιάς ≈ 8-10 km."""
        athens = {"lat": 37.9755, "lon": 23.7348}
        piraeus = {"lat": 37.9475, "lon": 23.6463}
        dist = haversine_distance(athens, piraeus)
        self.assertGreater(dist, 7.0, "Απόσταση μικρότερη του αναμενόμενου")
        self.assertLess(dist, 12.0, "Απόσταση μεγαλύτερη του αναμενόμενου")

    def test_same_point_returns_zero(self):
        """Ίδιο σημείο → απόσταση = 0."""
        coord = {"lat": 37.9715, "lon": 23.7257}
        dist = haversine_distance(coord, coord)
        self.assertAlmostEqual(dist, 0.0, places=5)


# ---------------------------------------------------------------------------
# Test 2 — time_to_minutes / minutes_to_time round-trip
# ---------------------------------------------------------------------------
class TestTimeConversions(unittest.TestCase):
    def test_time_to_minutes_known_values(self):
        self.assertEqual(time_to_minutes("00:00"), 0)
        self.assertEqual(time_to_minutes("01:00"), 60)
        self.assertEqual(time_to_minutes("14:30"), 870)
        self.assertEqual(time_to_minutes("23:59"), 1439)

    def test_minutes_to_time_known_values(self):
        self.assertEqual(minutes_to_time(0), "00:00")
        self.assertEqual(minutes_to_time(60), "01:00")
        self.assertEqual(minutes_to_time(870), "14:30")
        self.assertEqual(minutes_to_time(1439), "23:59")

    def test_round_trip_fidelity(self):
        """time_to_minutes(minutes_to_time(x)) == x για 0..1440."""
        for mins in range(0, 1440, 37):
            self.assertEqual(time_to_minutes(minutes_to_time(mins)), mins)


# ---------------------------------------------------------------------------
# Test 3 — filter_candidate_pois: wheelchair filter
# ---------------------------------------------------------------------------
class TestFilterCandidatePoisWheelchair(unittest.TestCase):
    def setUp(self):
        self.pois = [
            {"id": "museum_a", "name": "Μουσείο Α", "wheelchair_accessible": True, "category": "museum", "tags": []},
            {"id": "ruin_b",   "name": "Αρχαία Β",  "wheelchair_accessible": False, "category": "ruins",  "tags": []},
            {"id": "park_c",   "name": "Πάρκο Γ",   "wheelchair_accessible": True,  "category": "park",   "tags": []},
        ]

    def test_wheelchair_filter_keeps_only_accessible(self):
        result = filter_candidate_pois(self.pois, wheelchair_accessible=True)
        ids = [p["id"] for p in result]
        self.assertIn("museum_a", ids)
        self.assertIn("park_c", ids)
        self.assertNotIn("ruin_b", ids)

    def test_no_wheelchair_filter_keeps_all(self):
        result = filter_candidate_pois(self.pois, wheelchair_accessible=False)
        self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# Test 4 — filter_candidate_pois: blacklist category
# ---------------------------------------------------------------------------
class TestFilterCandidatePoisBlacklistCategory(unittest.TestCase):
    def setUp(self):
        self.pois = [
            {"id": "museum_a", "name": "Μουσείο Α", "category": "museum",     "tags": [], "wheelchair_accessible": True},
            {"id": "club_b",   "name": "Club B",    "category": "nightlife",   "tags": [], "wheelchair_accessible": True},
            {"id": "church_c", "name": "Εκκλησία",  "category": "religious",   "tags": [], "wheelchair_accessible": True},
        ]

    def test_blacklist_nightlife_removes_club(self):
        result = filter_candidate_pois(self.pois, blacklisted_categories={"nightlife"})
        ids = [p["id"] for p in result]
        self.assertNotIn("club_b", ids)
        self.assertIn("museum_a", ids)
        self.assertIn("church_c", ids)

    def test_multiple_blacklisted_categories(self):
        result = filter_candidate_pois(self.pois, blacklisted_categories={"nightlife", "religious"})
        ids = [p["id"] for p in result]
        self.assertEqual(ids, ["museum_a"])


# ---------------------------------------------------------------------------
# Test 5 — filter_candidate_pois: blacklist tags
# ---------------------------------------------------------------------------
class TestFilterCandidatePoisBlacklistTags(unittest.TestCase):
    def test_blacklist_tag_crowded_removes_poi(self):
        pois = [
            {"id": "top_a", "name": "Top A", "category": "attraction", "tags": ["crowded", "outdoor"], "wheelchair_accessible": True},
            {"id": "gem_b", "name": "Gem B",  "category": "attraction", "tags": ["hidden_gem"],         "wheelchair_accessible": True},
        ]
        result = filter_candidate_pois(pois, blacklisted_tags={"crowded"})
        ids = [p["id"] for p in result]
        self.assertNotIn("top_a", ids)
        self.assertIn("gem_b", ids)


# ---------------------------------------------------------------------------
# Test 6 — optimize_itinerary_duration: zero remaining (exact fit)
# ---------------------------------------------------------------------------
class TestOptimizeZeroRemainder(unittest.TestCase):
    def test_exact_fit_no_modification(self):
        """Όταν δεν περισσεύει χρόνος, το itinerary επιστρέφεται αναλλοίωτο."""
        itinerary = [
            {"name": "Ακρόπολη",   "duration": 90, "type": "museum"},
            {"name": "Μετάβαση",   "duration": 15, "type": "walking"},
            {"name": "Σύνταγμα",   "duration": 75, "type": "attraction"},
        ]
        # Total = 180, budget = 180 → remaining = 0
        result = optimize_itinerary_duration(itinerary[:], 180, 180)
        self.assertEqual(len(result), 3)
        self.assertEqual(calculate_total_mins(result), 180)


# ---------------------------------------------------------------------------
# Test 7 — optimize_itinerary_duration: over-budget (negative remaining)
# ---------------------------------------------------------------------------
class TestOptimizeOverBudget(unittest.TestCase):
    def test_over_budget_no_modification(self):
        """Αν το ήδη σχεδιασμένο πρόγραμμα υπερβαίνει το budget, δεν γίνεται καμία αλλαγή."""
        itinerary = [
            {"name": "Ακρόπολη",   "duration": 120, "type": "museum"},
            {"name": "Μετάβαση",   "duration": 20,  "type": "walking"},
            {"name": "Παρθενώνας", "duration": 90,  "type": "ruins"},
        ]
        # Total = 230, budget = 180 → remaining = -50
        result = optimize_itinerary_duration(itinerary[:], 180, 230)
        self.assertEqual(len(result), 3)
        self.assertEqual(calculate_total_mins(result), 230)  # unchanged


# ---------------------------------------------------------------------------
# Test 8 — calculate_total_mins: dict payload with "schedule" key
# ---------------------------------------------------------------------------
class TestCalculateTotalMinsDict(unittest.TestCase):
    def test_dict_payload_with_schedule_key(self):
        """Δέχεται dict (ItineraryPlan.to_dict()) με κλειδί 'schedule'."""
        payload = {
            "feasible": True,
            "start_time": "10:00",
            "end_time": "13:00",
            "schedule": [
                {"name": "POI α", "duration": 60,  "type": "museum"},
                {"name": "Walk",  "duration": 15,  "type": "walking"},
                {"name": "POI β", "duration": 105, "type": "attraction"},
            ]
        }
        self.assertEqual(calculate_total_mins(payload), 180)


# ---------------------------------------------------------------------------
# Test 9 — calculate_total_mins: empty list returns 0
# ---------------------------------------------------------------------------
class TestCalculateTotalMinsEmpty(unittest.TestCase):
    def test_empty_list_returns_zero(self):
        self.assertEqual(calculate_total_mins([]), 0)

    def test_empty_schedule_dict_returns_zero(self):
        self.assertEqual(calculate_total_mins({"schedule": []}), 0)

    def test_items_missing_duration_default_to_zero(self):
        """Αντικείμενα χωρίς 'duration' ή 'duration_mins' συνεισφέρουν 0."""
        result = calculate_total_mins([{"name": "No duration", "type": "museum"}])
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# Test 10 — DELIMITED_NATURAL_SYNTHESIS_PROMPT: BOS / EOS tokens
# ---------------------------------------------------------------------------
class TestDelimitedPromptTokens(unittest.TestCase):
    def test_bos_eos_tokens_present(self):
        self.assertIn("[BOS]", DELIMITED_NATURAL_SYNTHESIS_PROMPT)
        self.assertIn("[EOS]", DELIMITED_NATURAL_SYNTHESIS_PROMPT)

    def test_delimited_is_superset_of_base(self):
        """Το delimited prompt εμπεριέχει όλο το βασικό prompt."""
        self.assertIn("Είσαι ο Philody", DELIMITED_NATURAL_SYNTHESIS_PROMPT)
        self.assertIn("ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ", DELIMITED_NATURAL_SYNTHESIS_PROMPT)

    def test_bos_before_eos(self):
        """Σωστή σειρά token: BOS πριν το EOS."""
        bos_idx = DELIMITED_NATURAL_SYNTHESIS_PROMPT.index("[BOS]")
        eos_idx = DELIMITED_NATURAL_SYNTHESIS_PROMPT.index("[EOS]")
        self.assertLess(bos_idx, eos_idx)


# ---------------------------------------------------------------------------
# Bonus — EU AI Act disclaimer: appended by handle_user_request pipeline
# ---------------------------------------------------------------------------
class TestEUAIActDisclaimer(unittest.TestCase):
    def test_pipeline_appends_eu_ai_act_disclaimer(self):
        """handle_user_request προσθέτει πάντα το EU AI Act disclaimer στο chat_bubble_text."""
        result = handle_user_request(
            user_request="Πρόγραμμα 3 ωρών στην Αθήνα",
            time_budget=3.0,
        )
        chat_text = result["chat_bubble_text"]
        self.assertIn("EU AI Act", chat_text)
        self.assertIn("Άρθρο 50", chat_text)

    def test_disclaimer_is_at_end_of_chat_text(self):
        """Το disclaimer βρίσκεται στο τέλος (τελευταία 400 chars) του chat_bubble_text."""
        result = handle_user_request(
            user_request="1 ώρα στο Μουσείο Ακρόπολης",
            time_budget=1.0,
        )
        chat_text = result["chat_bubble_text"]
        disclaimer_idx = chat_text.rfind("EU AI Act")
        self.assertGreater(disclaimer_idx, 0)
        # Ο disclaimer είναι κοντά στο τέλος — τίποτα ουσιαστικό μετά (max 400 chars)
        self.assertLess(len(chat_text) - disclaimer_idx, 400)

    def test_render_natural_synthesis_generates_itinerary_text(self):
        """render_natural_synthesis επιστρέφει ελληνικό αφηγηματικό κείμενο με Ξεκινάμε."""
        dummy_itinerary = [
            {"name": "Ακρόπολη", "duration": 90,  "type": "museum",     "start_time": "10:00", "end_time": "11:30"},
            {"name": "Μετάβαση", "duration": 15,  "type": "walking"},
            {"name": "Πλάκα",   "duration": 75,  "type": "attraction", "start_time": "11:45", "end_time": "13:00"},
        ]
        output = render_natural_synthesis(dummy_itinerary, user_request="3 ώρες")
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 20)
        # Πρέπει να ξεκινά με φιλική χαιρετιστήρια ή τον Philody
        self.assertIn("Philody", output)


if __name__ == "__main__":
    unittest.main()
