"""
tests/test_natural_synthesis_and_feasibility.py

Comprehensive tests for:
Step 1: NATURAL_SYNTHESIS_PROMPT system prompt definition and strict rules
Step 2: optimize_itinerary_duration (Time-Filling Buffer & Dynamic Scaling)
Step 3: Two-Step Generation Pipeline (handle_user_request) & API endpoint
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

from orchestrator.prompts import (
    NATURAL_SYNTHESIS_PROMPT,
    DELIMITED_NATURAL_SYNTHESIS_PROMPT,
)
from engine.feasibility import (
    FeasibilityEngine,
    optimize_itinerary_duration,
    calculate_total_mins,
)
from orchestrator.agent import (
    handle_user_request,
    render_natural_synthesis,
    AthensTouristAgent,
)
from fastapi.testclient import TestClient
from api import app


class TestStep1NaturalSynthesisPrompt(unittest.TestCase):
    """Step 1: Tests for the Natural Language Synthesis System Prompt."""

    def test_prompt_presence_and_identity(self):
        self.assertIn("Είσαι ο Philody", NATURAL_SYNTHESIS_PROMPT)
        self.assertIn("Feasibility Engine", NATURAL_SYNTHESIS_PROMPT)
        self.assertIn("ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ", NATURAL_SYNTHESIS_PROMPT)

    def test_strict_rules_enforced_in_prompt(self):
        # Rule 1: Αφηγηματικός τόνος
        self.assertIn("ΑΦΗΓΗΜΑΤΙΚΟΣ ΤΟΝΟΣ", NATURAL_SYNTHESIS_PROMPT)
        # Rule 2: Τι απαγορεύεται
        self.assertIn("ΤΙ ΑΠΑΓΟΡΕΥΕΤΑΙ", NATURAL_SYNTHESIS_PROMPT)
        self.assertIn('"ID"', NATURAL_SYNTHESIS_PROMPT)
        self.assertIn('"Category"', NATURAL_SYNTHESIS_PROMPT)
        self.assertIn('"wheelchair_accessible"', NATURAL_SYNTHESIS_PROMPT)
        # Rule 3: Ωράρια
        self.assertIn("ΩΡΑΡΙΑ", NATURAL_SYNTHESIS_PROMPT)
        # Rule 4: Καμία παραίσθηση
        self.assertIn("ΚΑΜΙΑ ΠΑΡΑΙΣΘΗΣΗ", NATURAL_SYNTHESIS_PROMPT)
        # Rule 5: Πηγές
        self.assertIn("ΠΗΓΕΣ", NATURAL_SYNTHESIS_PROMPT)
        self.assertIn("📚 Πηγές που χρησιμοποιήθηκαν:", NATURAL_SYNTHESIS_PROMPT)


class TestStep2FeasibilityTimeBufferAndScaling(unittest.TestCase):
    """Step 2: Tests for optimize_itinerary_duration & calculate_total_mins."""

    def test_rest_slot_insertion_for_medium_remainder(self):
        """When 15 <= remaining_mins <= 45, a Rest Slot is appended."""
        itinerary = [
            {"name": "Μουσείο Ακρόπολης", "duration": 90, "type": "museum"},
            {"name": "Μετάβαση", "duration": 15, "type": "walking"},
            {"name": "Πλάκα", "duration": 45, "type": "attraction"},
        ]
        # Total current = 150 mins. Budget = 180 mins. Remaining = 30 mins.
        current_mins = calculate_total_mins(itinerary)
        self.assertEqual(current_mins, 150)

        optimized = optimize_itinerary_duration(itinerary, 180, current_mins)
        self.assertEqual(len(optimized), 4)

        rest_item = optimized[-1]
        self.assertEqual(rest_item["type"], "rest_area")
        self.assertEqual(rest_item["duration"], 30)
        self.assertIn("καφέ/ξεκούραση", rest_item["name"])

    def test_dynamic_scaling_for_small_remainder(self):
        """When 0 < remaining_mins < 15, extra minutes are distributed to non-walking POIs."""
        itinerary = [
            {"name": "Μουσείο Ακρόπολης", "duration": 90, "type": "museum"},
            {"name": "Μετάβαση", "duration": 15, "type": "walking"},
            {"name": "Πλάκα", "duration": 65, "type": "attraction"},
        ]
        # Total current = 170 mins. Budget = 180 mins. Remaining = 10 mins.
        # Non-walking POIs = 2 (extra_per_poi = 5).
        current_mins = calculate_total_mins(itinerary)
        self.assertEqual(current_mins, 170)

        optimized = optimize_itinerary_duration(itinerary, 180, current_mins)
        # No extra item added
        self.assertEqual(len(optimized), 3)

        # Walking step must remain unchanged
        walking_step = next(item for item in optimized if item["type"] == "walking")
        self.assertEqual(walking_step["duration"], 15)

        # Non-walking items received extra minutes (90+5 = 95, 65+5 = 70)
        poi1 = optimized[0]
        poi2 = optimized[2]
        self.assertEqual(poi1["duration"], 95)
        self.assertEqual(poi2["duration"], 70)
        self.assertEqual(calculate_total_mins(optimized), 180)

    def test_feasibility_engine_generate_draft(self):
        engine = FeasibilityEngine()
        draft = engine.generate_draft(
            user_request="Θέλω 3 ώρες χαλαρή βόλτα",
            time_budget=3.0,
        )
        self.assertIsInstance(draft, list)
        self.assertGreater(len(draft), 0)
        total = calculate_total_mins(draft)
        self.assertGreater(total, 0)


class TestStep3TwoStepGenerationPipeline(unittest.TestCase):
    """Step 3: Tests for Two-Step Generation Pipeline and API Integration."""

    def test_handle_user_request_structure(self):
        result = handle_user_request(
            user_request="Φτιάξε μου ένα πρόγραμμα 3 ωρών για το απόγευμα στην Αθήνα",
            time_budget=3.0,
        )
        self.assertIn("chat_bubble_text", result)
        self.assertIn("background_json_payload", result)

        chat_text = result["chat_bubble_text"]
        payload = result["background_json_payload"]

        # Check payload
        self.assertIsInstance(payload, list)
        self.assertGreater(len(payload), 0)

        # Check natural language synthesis rules
        self.assertIn("Philody", chat_text)
        self.assertIn("Ξεκινάμε στις", chat_text)
        self.assertIn("📚 Πηγές που χρησιμοποιήθηκαν:", chat_text)

        # Check EU AI Act transparency disclaimer
        self.assertIn("EU AI Act", chat_text)
        self.assertIn("Άρθρο 50", chat_text)

        # Verify no raw technical keys leaked in narrative
        for forbidden in ["'category':", "'indoor':", "'wheelchair_accessible':", "POIs ID:"]:
            self.assertNotIn(forbidden, chat_text)

    def test_agent_handle_user_request(self):
        agent = AthensTouristAgent()
        result = agent.handle_user_request(
            user_request="Πρόγραμμα 3 ωρών για την Αθήνα",
            time_budget=3.0,
        )
        self.assertIn("chat_bubble_text", result)
        self.assertIn("background_json_payload", result)

    def test_api_two_step_pipeline_endpoint(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/pipeline/two-step",
            json={
                "user_request": "Πρόγραμμα 3 ωρών για την Αθήνα",
                "time_budget": 3.0,
                "user_preferences": {"wheelchair_accessible": False}
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("chat_bubble_text", data)
        self.assertIn("background_json_payload", data)
        self.assertIn("EU AI Act", data["chat_bubble_text"])

    def test_api_chat_endpoint_includes_new_payload_keys(self):
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={
                "session_id": "test_step3_session",
                "message": "Ποιες είναι οι ώρες λειτουργίας στο Μουσείο Ακρόπολης;",
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("chat_bubble_text", data)
        self.assertIn("background_json_payload", data)


if __name__ == "__main__":
    unittest.main()
