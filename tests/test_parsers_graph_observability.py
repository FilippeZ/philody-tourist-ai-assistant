"""test_parsers_graph_observability.py - Comprehensive Unit & Integration Test Suite.

Tests:
1. Output Parsers Fallback Chains & Retry Policy (JSON repair, backoff, multi-tier fallback).
2. LangGraph State Graph with Reflection Loops (state transitions, self-critique, autonomous replanning).
3. LLM Observability & LangSmith Tracing (span creation, traces, telemetry, metric aggregation).
4. FastAPI REST Endpoints (/api/v1/observability/*, /api/v1/chat/graph).
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import json
import os
import unittest
from unittest.mock import MagicMock

# Offline deterministic environment
os.environ["OPENAI_API_KEY"] = ""

from orchestrator.tool_parsers import (
    PydanticToolOutputParser,
    RetryPolicy,
    OutputFixingParser,
    FallbackChain,
    WeatherToolArgs,
    RAGQueryArgs,
    FeasibilityPlanningArgs,
    weather_args_parser,
)
from orchestrator.observability import (
    ObservabilityHub,
    observability_hub,
    trace_tool_call,
    trace_rag_chain,
)
from orchestrator.graph import LangGraphOrchestrator, TouristAgentState
from orchestrator.agent import AthensTouristAgent, UserState


class TestOutputParsersFallbackAndRetry(unittest.TestCase):
    """Tests Output Parsers Retry Policy and Multi-Tier Fallback Chains."""

    def test_output_fixing_parser_prompt_generation(self):
        parser = PydanticToolOutputParser(WeatherToolArgs)
        fixer = OutputFixingParser(parser)
        prompt = fixer.build_repair_prompt(
            failed_raw_text="{'city': Athens, scenario: bad}",
            error_details="Invalid JSON syntax: single quotes and unquoted values"
        )
        self.assertIn("JSON Schema Validation Failure", prompt)
        self.assertIn("Athens", prompt)
        self.assertIn("JSON Schema", prompt)

    def test_retry_policy_heals_malformed_json_on_second_attempt(self):
        parser = PydanticToolOutputParser(WeatherToolArgs)
        attempts = 0

        def mock_llm_callable(prompt: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # Return malformed JSON
                return "Here is your output: {city: 'Athens', mock_scenario: 'rain'}"
            else:
                # Return corrected valid JSON on retry
                return '{"city": "Athens", "mock_scenario": "rain"}'

        policy = RetryPolicy(max_retries=2, initial_delay=0.01, backoff_factor=1.0)
        res, telemetry = parser.parse_with_retry(
            llm_callable=mock_llm_callable,
            initial_prompt="Extract weather parameters",
            retry_policy=policy,
        )

        self.assertEqual(res.city, "Athens")
        self.assertEqual(res.mock_scenario, "rain")
        self.assertTrue(telemetry["success"])
        self.assertEqual(telemetry["retries_attempted"], 1)
        self.assertEqual(telemetry["method"], "repaired_via_retry")

    def test_fallback_chain_activates_when_all_retries_fail(self):
        parser = PydanticToolOutputParser(FeasibilityPlanningArgs)

        # Callable that consistently returns natural language text instead of JSON
        def broken_llm_callable(prompt: str) -> str:
            return "Θέλω ένα πρόγραμμα 3 ώρες ξεκινώντας στις 15:00 με το παιδί μου 6 ετών."

        policy = RetryPolicy(max_retries=2, initial_delay=0.01)
        res, telemetry = parser.parse_with_retry(
            llm_callable=broken_llm_callable,
            initial_prompt="Generate feasibility args",
            retry_policy=policy,
        )

        self.assertTrue(telemetry["success"])
        self.assertTrue(telemetry["fallback_used"])
        self.assertEqual(telemetry["retries_attempted"], 2)
        # Check that heuristic tier 3 extracted the values from natural text
        self.assertEqual(res.time_budget_hours, 3.0)
        self.assertEqual(res.start_time, "15:00")
        self.assertTrue(res.traveling_with_kids)
        self.assertEqual(res.child_age, 6)

    def test_fallback_chain_tiers_direct_dict_and_codeblock(self):
        parser = PydanticToolOutputParser(RAGQueryArgs)
        
        # Tier 1 direct dictionary
        res_dict = parser.parse({"query": "Μουσείο Ακρόπολης", "top_k": 5})
        self.assertEqual(res_dict.query, "Μουσείο Ακρόπολης")
        self.assertEqual(res_dict.top_k, 5)

        # Tier 2 markdown codeblock
        codeblock_text = "Βεβαίως! Ορίστε το JSON:\n```json\n{\"query\": \"Πλάκα ιστορία\", \"top_k\": 4}\n```"
        res_codeblock = parser.parse(codeblock_text)
        self.assertEqual(res_codeblock.query, "Πλάκα ιστορία")
        self.assertEqual(res_codeblock.top_k, 4)


class TestLangGraphStateGraphAndReflection(unittest.TestCase):
    """Tests LangGraph State Graph with reflection loops & strict Cloud LLM enforcement."""

    @classmethod
    def setUpClass(cls):
        cls.agent = AthensTouristAgent()

    def test_langgraph_orchestrator_initialization(self):
        orchestrator = LangGraphOrchestrator(self.agent)
        self.assertIsNotNone(orchestrator.graph)

    def test_deterministic_synthesizer_permanently_disabled(self):
        """Verify DeterministicSynthesizer is permanently disabled and strictly raises 503."""
        from orchestrator.llm_client import DeterministicSynthesizer
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            DeterministicSynthesizer()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Cloud LLM is currently unavailable.")

    def test_cloud_llm_missing_key_raises_503(self):
        """Verify CloudOllamaClient strictly raises 503 when API key is missing."""
        from orchestrator.llm_client import CloudOllamaClient
        from fastapi import HTTPException

        client = CloudOllamaClient(api_key="", host="https://ollama.com")
        with self.assertRaises(HTTPException) as ctx:
            client.chat([{"role": "user", "content": "test"}])
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Cloud LLM is currently unavailable.")

    def test_cloud_llm_timeout_raises_503_no_fallback(self):
        """Verify CloudOllamaClient raises 503 when timeout or connection error occurs (no fallback)."""
        from orchestrator.llm_client import CloudOllamaClient
        from fastapi import HTTPException
        from unittest.mock import patch
        import requests

        client = CloudOllamaClient(api_key="valid-key", host="https://ollama.com", timeout=0.01)
        with patch("requests.post", side_effect=requests.Timeout("Connection timed out")):
            with self.assertRaises(HTTPException) as ctx:
                client.chat([{"role": "user", "content": "test"}])
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertEqual(ctx.exception.detail, "Cloud LLM is currently unavailable.")

    def test_langgraph_factual_qa_flow(self):
        from unittest.mock import patch
        state = UserState(session_id="qa_user")
        orchestrator = LangGraphOrchestrator(self.agent)
        with patch("orchestrator.llm_client.CloudOllamaClient.chat", return_value="Το Μουσείο Ακρόπολης λειτουργεί καθημερινά 09:00 - 20:00."):
            res = orchestrator.execute("Ποιο είναι το ωράριο του Μουσείου Ακρόπολης;", state)

        self.assertIn("final_response", res)
        self.assertIn("retrieval", res.get("execution_steps", []))
        self.assertIn("reflection", res.get("execution_steps", []))
        self.assertIn("synthesis", res.get("execution_steps", []))
        self.assertIn("compliance", res.get("execution_steps", []))
        self.assertGreater(len(res["final_response"]), 20)

    def test_langgraph_reflection_loop_on_closed_hours(self):
        """
        Tests that when a POI's closing time conflicts with the requested schedule,
        the reflection node critiques the plan and triggers autonomous re-planning.
        """
        from unittest.mock import patch
        state = UserState(session_id="reflection_user", start_time="16:00", time_budget_hours=4.0)
        orchestrator = LangGraphOrchestrator(self.agent)

        msg = "Θέλω να πάω στο Μουσείο Ακρόπολης στις 16:00 για 4 ώρες."
        with patch("orchestrator.llm_client.CloudOllamaClient.chat", return_value="Ετοίμασα το πρόγραμμά σας προσαρμόζοντας το ωράριο."):
            res = orchestrator.execute(msg, state)

        self.assertIn("final_response", res)
        self.assertIn("reflection", res.get("execution_steps", []))
        self.assertTrue(res.get("reflection_approved", True))

    def test_athens_tourist_agent_chat_graph(self):
        from unittest.mock import patch
        agent = AthensTouristAgent()
        with patch("orchestrator.llm_client.CloudOllamaClient.chat", return_value="Ακολουθεί το πρόγραμμα 3 ωρών στην Αθήνα."):
            res = agent.chat("Φτιάξε μου ένα πρόγραμμα 3 ωρών στην Αθήνα.", use_langgraph=True)

        self.assertIn("reply", res)
        self.assertIn("engine", res)
        self.assertEqual(res["engine"], "langgraph_state_graph")
        self.assertIn("execution_steps", res)
        self.assertIn("guardrails", res["execution_steps"])
        self.assertIn("reflection", res["execution_steps"])
        self.assertIn("synthesis", res["execution_steps"])
        self.assertIn("compliance", res["execution_steps"])
        self.assertIn("token_usage", res)


class TestObservabilityAndLangSmith(unittest.TestCase):
    """Tests LLM Observability and distributed tracing."""

    def setUp(self):
        self.hub = ObservabilityHub()

    def test_start_and_finish_trace(self):
        trace = self.hub.start_trace("test_user_turn", session_id="obs_user_1")
        self.assertIsNotNone(trace.trace_id)
        self.assertEqual(trace.status, "running")

        # Create child spans
        span1 = self.hub.create_span(trace.trace_id, name="weather_check", span_type="tool")
        span1.finish(outputs={"temp": 25, "city": "Athens"})
        self.assertEqual(span1.status, "success")
        self.assertGreaterEqual(span1.duration_ms, 0.0)

        finished_trace = self.hub.finish_trace(trace.trace_id, status="success")
        self.assertIsNotNone(finished_trace)
        self.assertEqual(finished_trace.status, "success")
        self.assertEqual(len(finished_trace.spans), 1)

    def test_trace_decorators(self):
        hub = ObservabilityHub()
        trace = hub.start_trace("decorated_run")

        @trace_tool_call("mock_weather_tool")
        def mock_weather(city: str):
            return {"temp": 28, "status": "sunny"}

        res = mock_weather("Athens", _trace_id=trace.trace_id)
        self.assertEqual(res["status"], "sunny")

    def test_metrics_summary_aggregation(self):
        hub = ObservabilityHub()
        t1 = hub.start_trace("trace_1")
        s1 = hub.create_span(t1.trace_id, name="weather_tool", span_type="tool")
        s1.finish(outputs={"ok": True})
        hub.finish_trace(t1.trace_id, status="success")

        metrics = hub.get_metrics_summary()
        self.assertEqual(metrics["total_traces"], 1)
        self.assertEqual(metrics["error_rate_pct"], 0.0)
        self.assertIn("weather_tool", metrics["tool_calls_breakdown"])


class TestFastAPIObservabilityEndpoints(unittest.TestCase):
    """Tests FastAPI endpoints for observability and LangGraph chat."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api import app
        cls.client = TestClient(app)

    def test_observability_status_endpoint(self):
        response = self.client.get("/api/v1/observability/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("langsmith_available", data)
        self.assertIn("langsmith_tracing_enabled", data)
        self.assertIn("langsmith_project", data)

    def test_observability_traces_endpoint(self):
        # Generate at least one trace first
        observability_hub.start_trace("endpoint_test_trace")
        response = self.client.get("/api/v1/observability/traces")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_observability_metrics_endpoint(self):
        response = self.client.get("/api/v1/observability/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_traces", data)
        self.assertIn("avg_latency_ms", data)
        self.assertIn("error_rate_pct", data)

    def test_chat_langgraph_endpoint(self):
        from unittest.mock import patch
        payload = {
            "session_id": "test_graph_session_99",
            "message": "Πες μου για το Μουσείο Ακρόπολης."
        }
        with patch("orchestrator.llm_client.CloudOllamaClient.chat", return_value="Το Μουσείο Ακρόπολης είναι ένα από τα σημαντικότερα μουσεία."):
            response = self.client.post("/api/v1/chat/graph", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], "test_graph_session_99")
        self.assertGreater(len(data["response"]), 10)
        self.assertIn("token_usage", data)

    def test_chat_langgraph_endpoint_raises_503_on_cloud_llm_failure(self):
        from unittest.mock import patch
        from fastapi import HTTPException
        payload = {
            "session_id": "test_graph_session_fail",
            "message": "Πες μου για το Μουσείο Ακρόπολης."
        }
        with patch("orchestrator.llm_client.CloudOllamaClient.chat", side_effect=HTTPException(status_code=503, detail="Cloud LLM is currently unavailable.")):
            response = self.client.post("/api/v1/chat/graph", json=payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Cloud LLM is currently unavailable.")


if __name__ == "__main__":
    unittest.main()
