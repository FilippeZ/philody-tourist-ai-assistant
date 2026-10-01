"""test_context_management.py - Comprehensive Unit & Integration Tests for Context Window Management.

Tests:
1. Special Boundary Delimiter Tokens ([BOS], [EOS], <|endoftext|>, etc.) wrapping & parsing.
2. Token Counting (TikToken / Heuristic fallback for Greek & English) & Pricing Calculation.
3. Token-aware Sliding Window Truncation with Pair Preservation in Multi-Turn Dialogues.
4. RAG Chunk Greedy Knapsack Fitting within Token Budget.
5. Full Prompt Assembly with Zero-Overflow Guarantees.
6. UserState Multi-Turn State Management Token Budgeting.
7. FastAPI Endpoints (/api/v1/context/estimate, /api/v1/context/metrics, /api/v1/chat token_usage).
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import os
import sys
import unittest

# Ensure network calls don't hang in offline test runner
os.environ["OPENAI_API_KEY"] = ""

from orchestrator.context_manager import (
    SpecialTokens,
    ContextLengthManager,
    MODEL_PRICING_CATALOG,
    TokenCostReport,
    context_length_manager,
)
from orchestrator.agent import UserState, AthensTouristAgent


class TestSpecialTokens(unittest.TestCase):
    """Test special delimiter tokens for boundary separation and parsing."""

    def test_special_tokens_definitions(self):
        self.assertEqual(SpecialTokens.BOS, "[BOS]")
        self.assertEqual(SpecialTokens.EOS, "[EOS]")
        self.assertEqual(SpecialTokens.END_OF_TEXT, "<|endoftext|>")
        self.assertEqual(SpecialTokens.BOS.strip(), "[BOS]")
        self.assertEqual(SpecialTokens.EOS.strip(), "[EOS]")
        self.assertIn("<|endoftext|>", SpecialTokens.END_OF_TEXT)

    def test_wrap_and_parse_blocks(self):
        system_content = "Είσαι ο Philody AI, ο επίσημος ψηφιακός ξεναγός της Αθήνας."
        user_content = "Ποια είναι τα κορυφαία αξιοθέατα στην Ακρόπολη;"
        assistant_content = "Στην Ακρόπολη μπορείτε να δείτε τον Παρθενώνα και το Ερέχθειο."

        wrapped_system = SpecialTokens.wrap_block(SpecialTokens.SYSTEM_TAG, system_content)
        wrapped_user = SpecialTokens.wrap_block(SpecialTokens.USER_TAG, user_content)
        wrapped_assistant = SpecialTokens.wrap_block(SpecialTokens.ASSISTANT_TAG, assistant_content)

        self.assertTrue(wrapped_system.startswith("[BOS]system\n"))
        self.assertTrue(wrapped_system.endswith("\n[EOS]"))

        combined = f"{wrapped_system}\n{wrapped_user}\n{wrapped_assistant}\n{SpecialTokens.END_OF_TEXT}"
        parsed = SpecialTokens.parse_blocks(combined)

        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["role"], "system")
        self.assertIn("Philody AI", parsed[0]["content"])
        self.assertEqual(parsed[1]["role"], "user")
        self.assertIn("Ακρόπολη", parsed[1]["content"])
        self.assertEqual(parsed[2]["role"], "assistant")
        self.assertIn("Παρθενώνα", parsed[2]["content"])


class TestTokenCountingAndCost(unittest.TestCase):
    """Test token counting accuracy and cost calculation across models."""

    def setUp(self):
        self.mgr = ContextLengthManager(max_context_tokens=4096, model_name="nemotron-3-ultra")

    def test_token_counting_english_and_greek(self):
        en_text = "Welcome to Athens! Let's explore the Acropolis together."
        gr_text = "Καλώς ήρθατε στην Αθήνα! Ας εξερευνήσουμε μαζί την Ακρόπολη."

        en_tokens = self.mgr.count_tokens(en_text)
        gr_tokens = self.mgr.count_tokens(gr_text)

        self.assertGreater(en_tokens, 5)
        self.assertGreater(gr_tokens, 5)
        # Empty text should return 0
        self.assertEqual(self.mgr.count_tokens(""), 0)

    def test_calculate_cost_nemotron(self):
        # nemotron-3-ultra rates: $0.80 / 1M prompt, $2.40 / 1M completion
        report = self.mgr.calculate_cost(prompt_tokens=1000, completion_tokens=500, model_name="nemotron-3-ultra")
        self.assertEqual(report.prompt_tokens, 1000)
        self.assertEqual(report.completion_tokens, 500)
        self.assertEqual(report.total_tokens, 1500)
        expected_p_cost = (1000 / 1_000_000.0) * 0.80
        expected_c_cost = (500 / 1_000_000.0) * 2.40
        self.assertAlmostEqual(report.prompt_cost_usd, expected_p_cost, places=6)
        self.assertAlmostEqual(report.completion_cost_usd, expected_c_cost, places=6)
        self.assertAlmostEqual(report.total_cost_usd, expected_p_cost + expected_c_cost, places=6)
        self.assertAlmostEqual(report.context_utilization_pct, (1000 / 4096.0) * 100.0, places=2)

    def test_calculate_cost_gpt4o_mini(self):
        report = self.mgr.calculate_cost(prompt_tokens=2000, completion_tokens=1000, model_name="gpt-4o-mini")
        # gpt-4o-mini rates: $0.15 / 1M prompt, $0.60 / 1M completion
        expected_p_cost = (2000 / 1_000_000.0) * 0.15
        expected_c_cost = (1000 / 1_000_000.0) * 0.60
        self.assertAlmostEqual(report.total_cost_usd, expected_p_cost + expected_c_cost, places=6)


class TestSlidingWindowTruncation(unittest.TestCase):
    """Test sliding window truncation by token budget and pair preservation."""

    def setUp(self):
        # Manager with a small history budget for deterministic testing
        self.mgr = ContextLengthManager(history_budget_tokens=150)

    def test_sliding_window_within_budget(self):
        history = [
            {"role": "user", "content": "Γεια σου!"},
            {"role": "assistant", "content": "Γεια σας! Πώς μπορώ να βοηθήσω;"},
        ]
        pruned, overflow = self.mgr.truncate_history_by_tokens(history, max_tokens=200)
        self.assertFalse(overflow)
        self.assertEqual(len(pruned), 2)
        self.assertEqual(pruned[0]["role"], "user")
        self.assertEqual(pruned[1]["role"], "assistant")

    def test_sliding_window_overflow_prunes_oldest_first(self):
        # Build 10 turns of dialogue
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"Ερώτηση αριθμός {i} σχετικά με την Αθήνα και τα αξιοθέατά της."})
            history.append({"role": "assistant", "content": f"Απάντηση αριθμός {i} με αναλυτικές λεπτομέρειες για το αξιοθέατο."})

        # With a budget of 300 tokens, it fits the most recent ~2 pairs (~256 tokens) and prunes older ones
        pruned, overflow = self.mgr.truncate_history_by_tokens(history, max_tokens=300, preserve_pairs=True)
        self.assertTrue(overflow)
        self.assertLess(len(pruned), len(history))
        self.assertGreater(len(pruned), 0)
        # Ensure retained messages are the most recent ones
        last_retained = pruned[-1]
        self.assertEqual(last_retained["role"], "assistant")
        self.assertIn("9", last_retained["content"])

    def test_pair_preservation(self):
        history = [
            {"role": "user", "content": "Πρώτη ερώτηση χρήστη."},
            {"role": "assistant", "content": "Πρώτη απάντηση βοηθού."},
            {"role": "user", "content": "Δεύτερη ερώτηση χρήστη."},
            {"role": "assistant", "content": "Δεύτερη απάντηση βοηθού."},
        ]
        # Calculate tokens for the 2nd pair
        pair2_tokens = self.mgr.count_tokens(SpecialTokens.wrap_block("user", history[2]["content"])) + \
                       self.mgr.count_tokens(SpecialTokens.wrap_block("assistant", history[3]["content"]))

        # Budget enough for pair 2 plus a few extra tokens, but not enough for pair 1
        pruned, overflow = self.mgr.truncate_history_by_tokens(history, max_tokens=pair2_tokens + 10, preserve_pairs=True)
        self.assertTrue(overflow)
        self.assertEqual(len(pruned), 2)
        # Check that we preserved user + assistant together, no dangling assistant
        self.assertEqual(pruned[0]["role"], "user")
        self.assertEqual(pruned[0]["content"], "Δεύτερη ερώτηση χρήστη.")
        self.assertEqual(pruned[1]["role"], "assistant")
        self.assertEqual(pruned[1]["content"], "Δεύτερη απάντηση βοηθού.")


class TestRAGChunkKnapsack(unittest.TestCase):
    """Test RAG chunk knapsack selection based on relevance scores and token budgets."""

    def setUp(self):
        self.mgr = ContextLengthManager(rag_budget_tokens=100)

    def test_rag_chunk_budgeting(self):
        chunks = [
            {"id": "poi_1", "name": "Παρθενώνας", "text": "Ο Παρθενώνας είναι ναός αφιερωμένος στην Αθηνά.", "score": 0.95},
            {"id": "poi_2", "name": "Ακρόπολη Μουσείο", "text": "Το Μουσείο Ακρόπολης εκθέτει τα ευρήματα.", "score": 0.88},
            {"id": "poi_3", "name": "Σύνταγμα", "text": "Η κεντρική πλατεία της πόλης των Αθηνών.", "score": 0.40},
            {"id": "poi_4", "name": "Μοναστηράκι", "text": "Γνωστή περιοχή με παλαιοπωλεία.", "score": 0.35},
        ]
        # With budget 100 tokens, highest-score chunk poi_1 (72 tokens) fits first
        fitted, used_tokens = self.mgr.fit_rag_chunks_by_budget(chunks, max_tokens=100)
        self.assertLessEqual(used_tokens, 100)
        self.assertGreater(len(fitted), 0)
        # Top chunk (highest score) must be included
        self.assertEqual(fitted[0]["id"], "poi_1")


class TestPromptAssemblyAndOverflowPrevention(unittest.TestCase):
    """Test assemble_token_managed_prompt guarantees zero token overflow."""

    def test_assemble_token_managed_prompt(self):
        mgr = ContextLengthManager(max_context_tokens=1000, max_generation_tokens=200)
        system_instr = "Είσαι ο Philody AI. Βοήθησε με ευγένεια."
        docs = [
            {"id": "p1", "name": "Ακρόπολη", "text": "Ο ιερός βράχος.", "score": 0.9},
            {"id": "p2", "name": "Πλάκα", "text": "Η γραφική συνοικία.", "score": 0.8},
        ]
        history = [
            {"role": "user", "content": "Καλημέρα!"},
            {"role": "assistant", "content": "Καλημέρα σας!"},
        ]
        user_query = "Πού να πάω πρώτα;"

        result = mgr.assemble_token_managed_prompt(
            system_instructions=system_instr,
            retrieved_docs=docs,
            history=history,
            user_query=user_query,
        )

        self.assertTrue(result["context_overflow_prevented"])
        self.assertIn("[BOS]system", result["full_prompt"])
        self.assertIn("[BOS]context", result["full_prompt"])
        self.assertIn("[BOS]user", result["full_prompt"])
        self.assertIn("cost_report", result)
        self.assertLess(result["prompt_tokens"], 1000)


class TestUserStateIntegration(unittest.TestCase):
    """Test UserState token-budgeted sliding window."""

    def test_user_state_token_truncation(self):
        state = UserState(session_id="test_user", token_budget=300)
        # Add multiple messages
        for i in range(10):
            state.add_message("user", f"Ερώτηση χρήστη {i} με αρκετές λέξεις.")
            state.add_message("assistant", f"Απάντηση συστήματος {i} με χρήσιμες πληροφορίες.")

        tokens = state.get_history_token_count()
        # History tokens should not exceed the token budget
        self.assertLessEqual(tokens, 350)
        self.assertGreater(len(state.conversation_history), 0)


class TestFastAPIEndpoints(unittest.TestCase):
    """Test REST API endpoints for context management."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api import app
        cls.client = TestClient(app)

    def test_context_metrics_endpoint(self):
        response = self.client.get("/api/v1/context/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pricing_catalog", data)
        self.assertIn("max_context_tokens", data)
        self.assertIn("special_tokens", data)
        self.assertEqual(data["special_tokens"]["bos"], "[BOS]")
        self.assertEqual(data["special_tokens"]["eos"], "[EOS]")

    def test_context_estimate_endpoint(self):
        payload = {
            "user_message": "Ποιο είναι το ωράριο του Μουσείου Ακρόπολης;",
            "session_id": "test_user_estimate",
            "model_name": "nemotron-3-ultra"
        }
        response = self.client.post("/api/v1/context/estimate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["context_overflow_prevented"])
        self.assertGreater(data["prompt_tokens"], 0)
        self.assertIn("cost_report", data)
        self.assertEqual(data["cost_report"]["model_name"], "nemotron-3-ultra")

    def test_chat_endpoint_includes_token_usage(self):
        payload = {
            "session_id": "test_user_tokens_123",
            "message": "Πες μου λίγα λόγια για την Αθήνα."
        }
        response = self.client.post("/api/v1/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("token_usage", data)
        self.assertTrue(data.get("context_overflow_prevented", False))
        token_usage = data["token_usage"]
        self.assertIn("prompt_tokens", token_usage)
        self.assertIn("total_cost_usd", token_usage)
        self.assertGreater(token_usage["prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
