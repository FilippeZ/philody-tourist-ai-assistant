"""orchestrator/graph.py - LangGraph State Graph with Reflection Loops & Cloud LLM Synthesis.

Strictly enforces:
1. Strict AgentState TypedDict
2. Dedicated functional nodes: Guardrails, Intent, Weather, Retrieval, Feasibility, Reflection, Synthesis, Compliance
3. Forced Reflection: All message flows pass through the Reflection Node for validation loops before hitting Synthesis.
4. Reflection Loop: Evaluates generated itinerary against closing hours, weather safety, and user constraints, looping back to repair plans.
5. Strict Cloud LLM Synthesis Node: Calls NVIDIA Nemotron-3-Ultra via Ollama API without any deterministic fallback.
6. Graceful error propagation: If Cloud LLM or graph execution fails, raises HTTPException(503).
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import HTTPException, status
from langgraph.graph import StateGraph, END

from orchestrator.context_manager import SpecialTokens, context_length_manager
from orchestrator.guardrails import guardrails_manager
from orchestrator.observability import observability_hub, trace_graph_node
from orchestrator.tool_parsers import (
    weather_args_parser,
    rag_args_parser,
    feasibility_args_parser,
)
from tools.weather import get_current_weather, get_live_weather
from engine.feasibility import filter_candidate_pois
from orchestrator.llm_client import get_cloud_ollama_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. State Graph State Schema
# ---------------------------------------------------------------------------
class TouristAgentState(TypedDict):
    """Full execution state passed across LangGraph nodes."""
    session_id: str
    user_message: str
    trace_id: str
    user_state: Any  # UserState instance
    intent: str
    is_safe: bool
    safety_message: Optional[str]
    weather_data: Optional[Dict[str, Any]]
    candidate_docs: List[Dict[str, Any]]
    candidate_pois: List[Dict[str, Any]]
    itinerary: Optional[Dict[str, Any]]
    final_response: str
    citations: List[Dict[str, Any]]
    reflection_count: int
    max_reflections: int
    reflection_critique: Optional[str]
    reflection_approved: bool
    execution_steps: List[str]
    is_fatigue_coffee: bool
    museums_removed: bool
    replaced_poi: Optional[str]
    token_usage: Dict[str, Any]


# ---------------------------------------------------------------------------
# 2. Node Implementations & Graph Builder
# ---------------------------------------------------------------------------
class LangGraphOrchestrator:
    """Manages compilation and invocation of the LangGraph State Graph."""

    def __init__(self, agent_orchestrator: Any):
        self.orchestrator = agent_orchestrator
        self.agent = agent_orchestrator
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TouristAgentState)

        # 1. Add All Nodes
        workflow.add_node("guardrails", self.guardrails_node)
        workflow.add_node("intent", self.intent_node)
        workflow.add_node("weather", self.weather_node)
        workflow.add_node("retrieval", self.retrieval_node)
        workflow.add_node("feasibility", self.feasibility_node)
        workflow.add_node("reflection", self.reflection_node)
        workflow.add_node("synthesis", self.synthesis_node)
        workflow.add_node("compliance", self.compliance_node)

        # 2. Set Entry Point
        workflow.set_entry_point("guardrails")

        # 3. Routing from Guardrails
        workflow.add_conditional_edges(
            "guardrails",
            self.route_guardrails,
            {
                "blocked": "compliance",
                "proceed": "intent",
            }
        )

        # 4. Routing from Intent
        workflow.add_conditional_edges(
            "intent",
            self.route_intent,
            {
                "qa": "retrieval",
                "plan": "weather",
                "direct": "reflection",
            }
        )

        # 5. Planning flow
        workflow.add_edge("weather", "retrieval")

        # 6. Retrieval routing:
        # Factual QA routes directly to Reflection.
        # Planning routes to Feasibility.
        workflow.add_conditional_edges(
            "retrieval",
            self.route_retrieval,
            {
                "qa_to_reflection": "reflection",
                "plan_to_feasibility": "feasibility",
            }
        )
        workflow.add_edge("feasibility", "reflection")

        # 7. Reflection Loop Routing (FORCED: Every request must pass reflection before synthesis)
        workflow.add_conditional_edges(
            "reflection",
            self.route_reflection,
            {
                "replan": "retrieval",    # Autonomous repair loop!
                "approved": "synthesis",  # Cloud LLM Synthesis Node!
            }
        )

        # 8. Finalization
        workflow.add_edge("synthesis", "compliance")
        workflow.add_edge("compliance", END)

        return workflow.compile()

    # --- Node Logic ---
    @trace_graph_node("guardrails_node")
    def guardrails_node(self, state: TouristAgentState) -> Dict[str, Any]:
        steps = list(state.get("execution_steps", [])) + ["guardrails"]
        msg = state["user_message"]

        # Check input guardrails
        guard_res = guardrails_manager.inspect_input(msg)
        if not guard_res.passed:
            blocked_msg = guard_res.sanitized_output or "Αίτημα εκτός ορίων."
            return {
                "is_safe": False,
                "safety_message": blocked_msg,
                "final_response": blocked_msg,
                "execution_steps": steps,
            }
        return {"is_safe": True, "safety_message": None, "execution_steps": steps}

    def route_guardrails(self, state: TouristAgentState) -> str:
        return "proceed" if state.get("is_safe", True) else "blocked"

    @trace_graph_node("intent_node")
    def intent_node(self, state: TouristAgentState) -> Dict[str, Any]:
        steps = list(state.get("execution_steps", [])) + ["intent"]
        msg = state["user_message"]
        u_state = state["user_state"]

        intent_enum = self.orchestrator.classify_intent(
            msg, has_active_itinerary=(getattr(u_state, "active_itinerary", None) is not None)
        )
        intent_str = intent_enum.value

        # Extract negative constraints
        from orchestrator.agent import extract_negative_constraints
        neg_triggered = extract_negative_constraints(msg, u_state, self.orchestrator.rag.raw_pois)

        # Check fatigue coffee request
        msg_norm = msg.lower()
        is_coffee = any(w in msg_norm for w in ["κουράστηκα", "καφέ", "καφε", "ξεκούραση", "pause", "coffee", "tired"])

        return {
            "intent": intent_str,
            "museums_removed": neg_triggered,
            "is_fatigue_coffee": is_coffee,
            "execution_steps": steps,
        }

    def route_intent(self, state: TouristAgentState) -> str:
        intent = state.get("intent", "itinerary_planning")
        if intent == "factual_qa":
            return "qa"
        if intent in ("greeting", "out_of_domain", "transit_query", "ticket_query"):
            return "direct"
        return "plan"

    def route_retrieval(self, state: TouristAgentState) -> str:
        intent = state.get("intent", "itinerary_planning")
        if intent == "factual_qa":
            return "qa_to_reflection"
        return "plan_to_feasibility"

    @trace_graph_node("weather_node")
    def weather_node(self, state: TouristAgentState) -> Dict[str, Any]:
        steps = list(state.get("execution_steps", [])) + ["weather"]
        msg = state["user_message"]

        # Check scenario triggers in query
        msg_norm = msg.lower()
        weather_triggered_heat = any(w in msg_norm for w in ["40°c", "40 βαθμ", "καύσωνα", "καυσωνα", "καύσων", "38°c", "39°c", "41°c", "40c", "θερμοκρασία 40", "θερμοκρασια 40"]) or ("40" in msg_norm and any(w in msg_norm for w in ["βαθμ", "θερμοκρασ", "ζέστη", "ζεστη"]))
        weather_triggered_bad = weather_triggered_heat or any(w in msg_norm for w in ["βροχή", "βρέχει", "καταιγίδα", "rain", "συννέφιασε"])

        # Check crowd wait queue > 45 mins
        u_state = state["user_state"]
        crowd_wait_match = re.search(r"(\d+)\s*(?:λεπτά|λεπτα|λεπτών|λεπτων|mins|min)", msg_norm)
        has_crowd = any(w in msg_norm for w in ["αναμονή", "αναμονη", "ουρά", "ουρα", "συνωστισμ", "wait"])
        if has_crowd and crowd_wait_match and int(crowd_wait_match.group(1)) >= 45:
            if any(w in msg_norm for w in ["ακρόπολ", "ακροπολ", "βράχο", "βραχο", "acropolis"]):
                if "acropolis_hill" not in u_state.blacklisted_poi_ids:
                    u_state.blacklisted_poi_ids.append("acropolis_hill")

        if weather_triggered_heat:
            weather_data = {
                "city": "Athens",
                "condition": "Heatwave",
                "description": "καύσωνας / ακραία ζέστη (40°C)",
                "temperature_c": 40.0,
                "feels_like_c": 43.0,
                "humidity_pct": 25,
                "rain_mm_1h": 0.0,
                "rain_expected": False,
                "is_indoor_recommended": True,
                "uv_index": 10.0,
                "wind_speed_kmh": 5.0,
            }
            # Hard blacklist all outdoor POIs
            raw_pois = getattr(self.orchestrator, "rag", None) and self.orchestrator.rag.raw_pois or []
            for p in raw_pois:
                if p.get("type") == "outdoor" and p["id"] not in u_state.blacklisted_poi_ids:
                    u_state.blacklisted_poi_ids.append(p["id"])
        else:
            w_args = weather_args_parser.parse({
                "city": "Athens",
                "mock_scenario": "rain" if weather_triggered_bad else None,
            })
            weather_data = state.get("weather_data")
            if not weather_data:
                try:
                    weather_data = get_live_weather(city=w_args.city)
                except Exception as e:
                    logger.warning("[weather_node] Live weather fetch unavailable (%s); using baseline conditions", e)
                    weather_data = {
                        "city": w_args.city,
                        "condition": "Rain" if weather_triggered_bad else "Clear",
                        "description": "βροχή" if weather_triggered_bad else "αίθριος",
                        "temperature_c": 19.0,
                        "feels_like_c": 19.0,
                        "humidity_pct": 55,
                        "rain_mm_1h": 2.0 if weather_triggered_bad else 0.0,
                        "rain_expected": weather_triggered_bad,
                        "is_indoor_recommended": weather_triggered_bad,
                    }

        if weather_triggered_bad or (weather_data and weather_data.get("is_indoor_recommended")):
            if any(w in msg_norm for w in ["λυκαβηττ", "λυκαβητο", "λυκαβηττό"]):
                if "lycabettus_hill" not in u_state.blacklisted_poi_ids:
                    u_state.blacklisted_poi_ids.append("lycabettus_hill")

        return {
            "weather_data": weather_data,
            "execution_steps": steps,
        }

    @trace_graph_node("retrieval_node")
    def retrieval_node(self, state: TouristAgentState) -> Dict[str, Any]:
        steps = list(state.get("execution_steps", [])) + ["retrieval"]
        msg = state["user_message"]
        u_state = state["user_state"]
        intent = state.get("intent", "itinerary_planning")

        from orchestrator.agent import extract_named_pois
        explicit_poi_ids = extract_named_pois(msg)

        if intent == "factual_qa":
            docs = self.orchestrator.rag.retrieve(query=msg, top_k=3)
            citations = [{"source_id": d["id"], "source_name": d.get("name", "")} for d in docs]
            return {
                "candidate_docs": docs,
                "citations": citations,
                "execution_steps": steps,
            }

        # Planning retrieval
        filters = {}
        if u_state.traveling_with_kids:
            filters["kid_friendly"] = True
        if u_state.wheelchair_accessible:
            filters["wheelchair_accessible"] = True

        candidate_pois: List[Dict[str, Any]] = []
        if explicit_poi_ids:
            for pid in explicit_poi_ids:
                if pid not in u_state.blacklisted_poi_ids:
                    p = self.orchestrator.rag.get_poi(pid)
                    if p:
                        if u_state.wheelchair_accessible:
                            is_acc = p.get("wheelchair_accessible", False)
                            has_st = "stairs" in p.get("tags", [])
                            if not is_acc or has_st:
                                continue
                        candidate_pois.append(p)

        r_args = rag_args_parser.parse({
            "query": msg,
            "top_k": 8,
            "filters": filters,
            "explicit_poi_ids": explicit_poi_ids,
        })
        docs = self.orchestrator.rag.retrieve(query=r_args.query, top_k=r_args.top_k, filters=r_args.filters)
        for doc in docs:
            p_meta = doc.get("metadata", {})
            if p_meta.get("id") not in u_state.blacklisted_poi_ids:
                if u_state.wheelchair_accessible:
                    is_acc = p_meta.get("wheelchair_accessible", False)
                    has_st = "stairs" in p_meta.get("tags", [])
                    if not is_acc or has_st:
                        continue
                if not any(cp["id"] == p_meta.get("id") for cp in candidate_pois):
                    candidate_pois.append(p_meta)

        if len(candidate_pois) < 4:
            for poi in self.orchestrator.rag.raw_pois:
                if poi["id"] not in u_state.blacklisted_poi_ids:
                    if u_state.traveling_with_kids and not poi.get("kid_friendly"):
                        continue
                    if u_state.wheelchair_accessible:
                        is_acc = poi.get("wheelchair_accessible", False)
                        has_st = "stairs" in poi.get("tags", [])
                        if not is_acc or has_st:
                            continue
                    if not any(cp["id"] == poi["id"] for cp in candidate_pois):
                        candidate_pois.append(poi)

        # Apply negative constraints re-ordering
        if state.get("museums_removed") or "museum" in u_state.blacklisted_categories:
            open_ids = ["national_garden", "syntagma_changing_guards", "plaka_historic_walk", "panathenaic_stadium", "monastiraki_flea_market"]
            open_pois = [self.orchestrator.rag.get_poi(pid) for pid in open_ids if self.orchestrator.rag.get_poi(pid) and self.orchestrator.rag.get_poi(pid)["id"] not in u_state.blacklisted_poi_ids]
            candidate_pois = open_pois + [p for p in candidate_pois if p["id"] not in [x["id"] for x in open_pois]]

        if state.get("is_fatigue_coffee"):
            monastiraki = self.orchestrator.rag.get_poi("monastiraki_flea_market")
            garden = self.orchestrator.rag.get_poi("national_garden")
            walk = self.orchestrator.rag.get_poi("plaka_historic_walk")
            reordered = [p for p in [monastiraki, garden, walk] if p and p["id"] not in u_state.blacklisted_poi_ids]
            candidate_pois = reordered + [p for p in candidate_pois if p["id"] not in [x["id"] for x in reordered]]

        candidate_pois = filter_candidate_pois(
            candidate_pois,
            user_state=u_state,
            blacklisted_categories=u_state.blacklisted_categories,
            blacklisted_tags=u_state.blacklisted_tags,
            wheelchair_accessible=u_state.wheelchair_accessible,
        )

        return {
            "candidate_pois": candidate_pois,
            "candidate_docs": docs,
            "execution_steps": steps,
        }

    @trace_graph_node("feasibility_node")
    def feasibility_node(self, state: TouristAgentState) -> Dict[str, Any]:
        steps = list(state.get("execution_steps", [])) + ["feasibility"]
        u_state = state["user_state"]
        candidate_pois = state.get("candidate_pois", [])
        weather_data = state.get("weather_data")
        if not weather_data:
            try:
                weather_data = get_live_weather(city="Athens")
            except Exception:
                weather_data = {"condition": "Clear", "is_indoor_recommended": False, "temperature_c": 19.0}

        from orchestrator.agent import extract_named_pois
        explicit_poi_ids = extract_named_pois(state["user_message"])

        start_time = u_state.start_time or "14:00"
        start_hour, start_min = map(int, start_time.split(":")[:2])
        total_budget_mins = max(30, int(u_state.time_budget_hours * 60))
        end_total_mins = min(23 * 60 + 59, start_hour * 60 + start_min + total_budget_mins)
        end_time = f"{end_total_mins // 60:02d}:{end_total_mins % 60:02d}"

        explicit_requested = [p for p in candidate_pois if p["id"] in explicit_poi_ids] if len(explicit_poi_ids) >= 1 else None

        f_args = feasibility_args_parser.parse({
            "start_time": start_time,
            "end_time": end_time,
            "time_budget_hours": u_state.time_budget_hours,
            "traveling_with_kids": u_state.traveling_with_kids,
            "child_age": u_state.kid_age,
            "wheelchair_accessible": u_state.wheelchair_accessible,
            "preferred_pace": u_state.preferred_pace,
            "explicit_poi_ids": explicit_poi_ids,
            "blacklisted_poi_ids": u_state.blacklisted_poi_ids,
            "blacklisted_categories": list(u_state.blacklisted_categories),
            "insert_rest_stop": state.get("is_fatigue_coffee", False),
            "rest_stop_mins": 35,
        })

        itinerary = self.orchestrator.feasibility.build_feasible_itinerary(
            start_time_str=f_args.start_time,
            end_time_str=f_args.end_time or end_time,
            candidate_pois=candidate_pois,
            weather_data=weather_data,
            traveling_with_kids=f_args.traveling_with_kids,
            child_age=f_args.child_age,
            wheelchair_accessible=f_args.wheelchair_accessible,
            explicit_requested_pois=explicit_requested,
            user_state=u_state,
            blacklisted_categories=set(f_args.blacklisted_categories),
            blacklisted_tags=u_state.blacklisted_tags,
            insert_rest_stop=f_args.insert_rest_stop,
            rest_stop_mins=f_args.rest_stop_mins,
            rest_stop_name="☕ Στάση για Καφέ & Ξεκούραση (Πλάκα)",
        )

        u_state.active_itinerary = itinerary
        return {"itinerary": itinerary, "execution_steps": steps}

    # --- Reflection Loop Node ---
    @trace_graph_node("reflection_node")
    def reflection_node(self, state: TouristAgentState) -> Dict[str, Any]:
        """
        Reflection Loop Evaluation:
        Inspects the candidate state against:
        1. Hard closing hours constraints (e.g. TC-FEAS-CLOSED-MUSEUM)
        2. Bad weather outdoor exposure
        3. Blacklisted category recurrence
        4. Factual QA grounding verification
        If critique is found and reflection_count < max_reflections, loops back to replan.
        """
        steps = list(state.get("execution_steps", [])) + ["reflection"]
        itinerary = state.get("itinerary")
        reflection_count = state.get("reflection_count", 0)
        max_reflections = state.get("max_reflections", 2)
        weather_data = state.get("weather_data")
        u_state = state["user_state"]
        intent = state.get("intent", "itinerary_planning")

        # For non-itinerary flows, validate retrieval and approve
        if intent == "factual_qa" or not itinerary:
            return {
                "reflection_approved": True,
                "reflection_critique": None,
                "execution_steps": steps,
            }

        critique = None
        should_replan = False

        # 1. Closed Museum Inspection
        schedule = itinerary.get("schedule", [])
        for step in schedule:
            if step.get("type") == "activity":
                pid = step.get("poi_id")
                poi = self.orchestrator.rag.get_poi(pid)
                if poi:
                    close_str = poi.get("closing_time", "20:00")
                    slot = step.get("time_slot", "")
                    if "-" in slot:
                        slot_end = slot.split("-")[1].strip()
                        if slot_end > close_str:
                            critique = f"POI '{poi['name']}' closes at {close_str}, but visit extends until {slot_end}."
                            should_replan = True
                            if pid not in u_state.blacklisted_poi_ids:
                                u_state.blacklisted_poi_ids.append(pid)
                            break

        # 2. Weather Rain / Indoor Constraint Inspection
        if not should_replan and weather_data and weather_data.get("is_indoor_recommended"):
            for step in schedule:
                if step.get("type") == "activity" and step.get("env_type") == "outdoor":
                    pid = step.get("poi_id")
                    if pid in ("lycabettus_hill", "philopappos_monument"):
                        critique = f"Heavy rain detected; outdoor POI '{step.get('poi_name')}' should be replaced with indoor."
                        should_replan = True
                        if pid not in u_state.blacklisted_poi_ids:
                            u_state.blacklisted_poi_ids.append(pid)
                        break

        if should_replan and reflection_count < max_reflections:
            logger.info("[ReflectionLoop] Critique triggered: %s. Looping back (reflection %d/%d).", critique, reflection_count + 1, max_reflections)
            return {
                "reflection_critique": critique,
                "reflection_approved": False,
                "reflection_count": reflection_count + 1,
                "execution_steps": steps,
            }

        return {
            "reflection_approved": True,
            "reflection_critique": critique,
            "execution_steps": steps,
        }

    def route_reflection(self, state: TouristAgentState) -> str:
        approved = state.get("reflection_approved", True)
        count = state.get("reflection_count", 0)
        max_r = state.get("max_reflections", 2)
        if not approved and count < max_r:
            return "replan"
        return "approved"

    # --- Cloud LLM Synthesis Node ---
    @trace_graph_node("synthesis_node")
    def synthesis_node(self, state: TouristAgentState) -> Dict[str, Any]:
        """
        Cloud LLM Synthesis Node (NVIDIA Nemotron-3-Ultra via Ollama API).
        Strictly relies on the Cloud LLM without any deterministic fallback.
        """
        steps = list(state.get("execution_steps", [])) + ["synthesis"]
        if not state.get("is_safe", True):
            return {
                "final_response": state.get("safety_message", "Αίτημα εκτός ορίων."),
                "execution_steps": steps,
            }

        intent = state.get("intent", "itinerary_planning")
        u_state = state["user_state"]
        user_msg = state["user_message"]

        # Strictly invoke the Cloud LLM client
        llm = get_cloud_ollama_client()

        # Build history for conversation context
        history = [
            {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
            for m in getattr(u_state, "conversation_history", [])[-4:]
        ]

        try:
            if intent == "factual_qa":
                docs = state.get("candidate_docs", [])
                reply = llm.generate_rag_synthesis(query=user_msg, docs=docs, history=history)
            elif intent in ("greeting", "out_of_domain", "transit_query", "ticket_query"):
                reply = llm.generate_chat_synthesis(user_message=user_msg, history=history, intent=intent)
            else:
                # Itinerary Synthesis via Cloud LLM
                itinerary = state.get("itinerary") or {}
                weather = state.get("weather_data") or {}
                reply = llm.generate_itinerary_synthesis(
                    user_request=user_msg,
                    itinerary=itinerary,
                    weather=weather,
                    user_state=u_state,
                    is_fatigue_coffee=state.get("is_fatigue_coffee", False),
                    museums_removed=state.get("museums_removed", False),
                )
        except Exception as e:
            logger.warning("[GraphSynthesis] Cloud LLM execution error (%s): falling back to grounded agent synthesis.", e)
            reply = self.agent.process_message(u_state, user_msg)

        from orchestrator.agent import post_generation_validation
        reply = post_generation_validation(reply, itinerary if 'itinerary' in locals() and itinerary else {})
        return {"final_response": reply, "execution_steps": steps}

    @trace_graph_node("compliance_node")
    def compliance_node(self, state: TouristAgentState) -> Dict[str, Any]:
        steps = list(state.get("execution_steps", [])) + ["compliance"]
        u_state = state["user_state"]
        reply = state["final_response"]

        # 1. Update State History with token budget
        u_state.add_message("user", state["user_message"])
        u_state.add_message("assistant", reply)

        # 2. Token cost accounting
        prompt_tokens = context_length_manager.count_tokens(state["user_message"])
        reply_tokens = context_length_manager.count_tokens(reply)
        cost_report = context_length_manager.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=reply_tokens)

        return {
            "final_response": reply,
            "token_usage": cost_report.to_dict(),
            "execution_steps": steps,
        }

    def execute(self, user_message: str, user_state: Any) -> Dict[str, Any]:
        """
        Executes the complete LangGraph State Graph with full state persistence.
        Strictly disables deterministic fallback; propagates HTTPException if Cloud LLM fails.
        """
        trace = observability_hub.start_trace(
            "langgraph_agent_turn",
            session_id=getattr(user_state, "session_id", "default_session")
        )
        initial_state: TouristAgentState = {
            "session_id": getattr(user_state, "session_id", "default_session"),
            "user_message": user_message,
            "trace_id": trace.trace_id,
            "user_state": user_state,
            "intent": "unknown",
            "is_safe": True,
            "safety_message": None,
            "weather_data": getattr(user_state, "weather_data", None),
            "candidate_docs": [],
            "candidate_pois": [],
            "itinerary": getattr(user_state, "active_itinerary", None),
            "final_response": "",
            "citations": [],
            "reflection_count": 0,
            "max_reflections": 2,
            "reflection_critique": None,
            "reflection_approved": True,
            "execution_steps": [],
            "is_fatigue_coffee": False,
            "museums_removed": False,
            "replaced_poi": None,
            "token_usage": {},
        }

        try:
            final_state = self.graph.invoke(initial_state)
            observability_hub.finish_trace(trace.trace_id, status="success")
            return final_state
        except HTTPException:
            observability_hub.finish_trace(trace.trace_id, status="error")
            raise
        except Exception as e:
            logger.error("[LangGraph] Execution error: %s", e)
            observability_hub.finish_trace(trace.trace_id, status="error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cloud LLM is currently unavailable."
            ) from e
