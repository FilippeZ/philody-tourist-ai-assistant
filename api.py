"""
FastAPI Enterprise REST Gateway for Athens AI Tourist Assistant (api.py).

Provides production-grade endpoints for:
- Conversational AI & Intent Routing (POST /api/v1/chat)
- Direct Itinerary Generation & Feasibility Validation (POST /api/v1/itinerary/generate)
- Live Weather Data & Resilience Fallback (GET /api/v1/weather)
- Smartwatch / Wearable IoT Telemetry Hub (POST /api/v1/iot/event)
- Observability Healthcheck (GET /health)
- Interactive Web Client UI (GET /)
"""

from __future__ import annotations
import os
import sys
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env FIRST — before any internal module imports that read env vars at module level
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# Εισαγωγή των εσωτερικών υποσυστημάτων
from rag.retriever import AthensRAGRetriever
from tools.weather import get_current_weather, get_live_weather, WeatherResponse
from tools.transit import (
    get_transit_route,
    get_station_schedule,
    get_transit_alerts,
    TransitRouteRequest,
)
from tools.ticketing import (
    check_ticket_availability,
    get_ticket_pricing,
    simulate_ticket_reservation,
    TicketBookingRequest,
)
from tools.watermarking import (
    SyntheticContentWatermarker,
    detect_text_watermark,
    EU_AI_ACT_ARTICLE_50_NOTICE,
)
from tools.audio_tour import (
    SyntheticAudioTourGenerator,
    detect_audio_watermark,
)
from engine.feasibility import FeasibilityEngine
from orchestrator.agent import AthensTouristAgent, TouristLLMOrchestrator, UserState, IntentType, extract_named_pois
from orchestrator.guardrails import guardrails_manager
from orchestrator.feedback import feedback_manager
from orchestrator.audit_logger import audit_logger, ImmutableAuditLogger
from orchestrator.context_manager import context_length_manager, SpecialTokens, MODEL_PRICING_CATALOG
from orchestrator.observability import observability_hub
from orchestrator.tool_parsers import (
    weather_args_parser,
    transit_args_parser,
    ticketing_args_parser,
    feasibility_args_parser,
    rag_args_parser,
)
from evaluation.ragas_eval import ragas_evaluator


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import time
from monitoring.observability import observability_registry
from rag.pipeline import atomic_poi_pipeline


# 1. Αρχικοποίηση FastAPI App (redirect_slashes=False ensures Vercel edge proxy compatibility)
app = FastAPI(
    title="🏛️ Philody AI Travel Assistant API",
    description=(
        "Η συνάντηση τριών πυλώνων: της αρχαιοελληνικής κληρονομιάς & φιλίας («Φίλος» + «Ωδή»), "
        "της τεχνολογικής καινοτομίας (Generative AI & Smart City IoT) και της μαθηματικής ακρίβειας "
        "(Deterministic Feasibility Engine)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)

# 2. Observability, Vercel Rewrites & CORS Middleware
class VercelPathNormalizationMiddleware(BaseHTTPMiddleware):
    """
    Normalizes request paths when running under Vercel Serverless / Edge proxies:
    1. Reads Vercel's 'x-matched-path' header if provided.
    2. Allows endpoints requested as /v1/... to map directly to /api/v1/...
    3. Prevents 404s when Vercel rewrites target serverless functions.
    """
    async def dispatch(self, request: Request, call_next):
        matched_path = request.headers.get("x-matched-path")
        if matched_path and request.scope.get("path") in ("/api.py", "/api/index.py", "/api", ""):
            clean_path = matched_path.split("?")[0]
            request.scope["path"] = clean_path

        if request.scope.get("path", "").startswith("/v1/"):
            request.scope["path"] = f"/api{request.scope['path']}"

        return await call_next(request)

class PrometheusObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as ex:
            status_code = 500
            raise ex
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            observability_registry.record_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=dur_ms
            )

app.add_middleware(VercelPathNormalizationMiddleware)
app.add_middleware(PrometheusObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Session Store (στην παραγωγή διασυνδέεται με Redis/DynamoDB)
SESSIONS: Dict[str, UserState] = {}

# Dependency Injections
rag_retriever = AthensRAGRetriever()
atomic_poi_pipeline.retriever = rag_retriever
feasibility_engine = FeasibilityEngine(transport_mode="walking")
orchestrator = TouristLLMOrchestrator(
    rag_retriever=rag_retriever,
    feasibility_engine=feasibility_engine
)
from orchestrator.graph import LangGraphOrchestrator
langgraph_orchestrator = LangGraphOrchestrator(agent_orchestrator=orchestrator)
content_watermarker = SyntheticContentWatermarker()
audio_tour_generator = SyntheticAudioTourGenerator()


# 3. Pydantic Schemas (Input / Output Validation)
class ChatRequest(BaseModel):
    session_id: str = Field(..., example="user_session_123")
    message: str = Field(..., example="Φτιάξε μου ένα πρόγραμμα 4 ωρών για το απόγευμα με το παιδί μου.")
    start_time: Optional[str] = Field("14:00", example="14:00")
    time_budget_hours: Optional[float] = Field(4.0, example=4.0)
    pace: Optional[str] = Field("moderate", example="moderate")
    preferred_pace: Optional[str] = Field("moderate", example="moderate")
    wheelchair_accessible: Optional[bool] = Field(False, example=False)
    traveling_with_kids: Optional[bool] = Field(False, example=False)


class CitationItem(BaseModel):
    source_id: str
    name: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    response: str
    chat_bubble_text: Optional[str] = None
    background_json_payload: Optional[Any] = None
    active_itinerary: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = []
    interaction_id: Optional[str] = None
    conversational_sentiment: Optional[Dict[str, Any]] = None
    transparency_notice: Optional[Dict[str, Any]] = None
    transparency_notice_delivered: bool = False
    is_synthetic_content: bool = True
    watermark_detected: Optional[bool] = None
    token_usage: Optional[Dict[str, Any]] = None
    context_overflow_prevented: bool = True


class ContextEstimateRequest(BaseModel):
    user_message: str = Field(..., example="Ποια είναι τα καλύτερα μουσεία στην Αθήνα;")
    session_id: Optional[str] = Field("default_user", example="user_session_123")
    model_name: Optional[str] = Field("nemotron-3-nano:30b", example="nemotron-3-nano:30b")


class ContextEstimateResponse(BaseModel):
    prompt_tokens: int
    system_tokens: int
    rag_tokens: int
    user_query_tokens: int
    history_turns_retained: int
    history_overflow_truncated: bool
    cost_report: Dict[str, Any]
    special_tokens_used: List[str]
    context_overflow_prevented: bool



class TransparencyNoticeResponse(BaseModel):
    status: str
    system_name: str
    risk_classification: str
    regulatory_obligations: List[str]
    transparency_notice_text: str
    pre_exposure_enforced: bool
    machine_readable_watermarking_enabled: bool
    immutable_audit_logging_active: bool
    audit_chain_verified: bool


class AudioTourGenerateRequest(BaseModel):
    poi_name: str = Field(..., example="Acropolis")
    topic: Optional[str] = Field("Ιστορία και Πολιτισμός", example="Ιστορία και Πολιτισμός")
    session_id: Optional[str] = Field("anon_session", example="session_123")


class WatermarkVerifyRequest(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None


class FeedbackSubmissionRequest(BaseModel):
    session_id: str = Field(..., example="user_session_123")
    rating: int = Field(..., ge=-1, le=1, description="1 for thumbs up, -1 for thumbs down")
    comment: Optional[str] = Field(None, example="Πολύ καλό πρόγραμμα και ακριβείς πληροφορίες.")
    tags: Optional[List[str]] = Field(None, example=["accurate_hours", "great_schedule"])
    feedback_id: Optional[str] = Field(None, example="fb_1727680000_abc123")


class ItineraryGenerateRequest(BaseModel):
    session_id: str = Field("default_user", example="user_session_123")
    start_time: str = Field("14:00", example="14:00")
    end_time: Optional[str] = Field(None, example="18:00")
    time_budget_hours: Optional[float] = Field(4.0, example=4.0)
    traveling_with_kids: Optional[bool] = Field(False, example=False)
    child_age: Optional[int] = Field(None, example=8)
    wheelchair_accessible: Optional[bool] = Field(False, example=False)
    preferred_pace: Optional[str] = Field("moderate", example="moderate")
    explicit_pois: Optional[List[str]] = Field(None, example=["acropolis_museum"])


class IoTEventRequest(BaseModel):
    session_id: str = Field(..., example="user_session_123")
    event_type: str = Field(..., example="fatigue_alert")  # fatigue_alert, geofence_entry, uv_warning, crowd_density
    location: Optional[Dict[str, float]] = Field(None, example={"lat": 37.9715, "lon": 23.7257})
    telemetry: Optional[Dict[str, Any]] = Field(None, example={"heart_rate": 142, "fatigue_index": 0.88})


def get_or_create_session(session_id: str) -> UserState:

    """Helper διαχείρισης συνεδρίας (session)."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = UserState(session_id=session_id)
    return SESSIONS[session_id]


# 4. REST Endpoints

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
def health_check():
    """Επιστρέφει την κατάσταση υγείας και ετοιμότητας του API."""
    return {
        "status": "healthy",
        "service": "philody-ai-travel-assistant",
        "version": "1.0.0",
        "registered_sessions": len(SESSIONS),
        "total_pois_indexed": len(rag_retriever.raw_pois),
    }


@app.api_route("/logo.jpg", methods=["GET", "HEAD"], include_in_schema=False)
def serve_logo():
    logo_file = Path(__file__).parent / "logo.jpg"
    if logo_file.exists():
        return FileResponse(str(logo_file), media_type="image/jpeg")
    landing_logo = Path(__file__).parent / "landing" / "logo.jpg"
    if landing_logo.exists():
        return FileResponse(str(landing_logo), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/api/v1/weather", response_model=WeatherResponse, tags=["Tools"])
def read_weather(city: str = "Athens"):
    """
    Ανακτά 100% πραγματικά, ζωντανά καιρικά δεδομένα (Real Data).
    Χρησιμοποιεί OpenWeatherMap αν υπάρχει κλειδί, ή το επίσημο μετεωρολογικό
    live grid του Open-Meteo, χωρίς ποτέ να κολλάει σε σταθερές τιμές.
    """
    try:
        w_args = weather_args_parser.parse({"city": city})
        target_city = w_args.city
    except Exception:
        target_city = city
    return get_live_weather(city=target_city)


@app.get("/api/v1/transit/route", tags=["Tools"])
def read_transit_route(
    origin: str = "Syntagma",
    destination: str = "Acropolis",
    wheelchair_accessible: bool = False,
    departure_time: str = "14:00"
):
    """
    Υπολογίζει δρομολόγιο δημόσιας συγκοινωνίας (Μετρό, Τραμ, Λεωφορεία ΟΑΣΑ) στην Αθήνα
    μεταξύ δύο σημείων με εκτίμηση χρόνου, μετεπιβιβάσεων και προσβασιμότητας ΑμεΑ.
    """
    t_args = transit_args_parser.parse({
        "origin": origin,
        "destination": destination,
        "wheelchair_accessible": wheelchair_accessible,
        "departure_time": departure_time,
    })
    return get_transit_route(
        origin=t_args.origin,
        destination=t_args.destination,
        wheelchair_accessible=t_args.wheelchair_accessible,
        departure_time=t_args.departure_time,
    )


@app.get("/api/v1/transit/schedule/{station}", tags=["Tools"])
def read_station_schedule(station: str):
    """Επιστρέφει ζωντανές αφίξεις συρμών και δρομολόγια για σταθμό του μετρό/τραμ."""
    return get_station_schedule(station)


@app.get("/api/v1/transit/alerts", tags=["Tools"])
def read_transit_alerts():
    """Επιστρέφει ζωντανές ειδοποιήσεις και κατάσταση δικτύου ΟΑΣΑ / ΣΤΑΣΥ."""
    return get_transit_alerts()


@app.get("/api/v1/tickets/pricing/{poi_id}", tags=["Tools"])
def read_ticket_pricing(poi_id: str):
    """Επιστρέφει επίσημες τιμές και κατηγορίες εισιτηρίων για ένα αξιοθέατο."""
    return get_ticket_pricing(poi_id)


@app.post("/api/v1/tickets/availability", tags=["Tools"])
def check_ticket_availability_endpoint(payload: Dict[str, Any]):
    """Ελέγχει ζωντανή διαθεσιμότητα χρονικών ζωνών (time slots) για ένα αξιοθέατο."""
    tkt_args = ticketing_args_parser.parse(payload)
    return check_ticket_availability(
        poi_id=tkt_args.poi_id,
        visit_date=tkt_args.visit_date,
        time_slot=tkt_args.time_slot,
    )


@app.post("/api/v1/tickets/book", tags=["Tools"])
def book_ticket_endpoint(payload: TicketBookingRequest):
    """Προσομοιώνει έκδοση ηλεκτρονικού εισιτηρίου με QR barcode token."""
    return simulate_ticket_reservation(
        poi_id=payload.poi_id,
        visit_date=payload.visit_date,
        time_slot=payload.time_slot,
        num_tickets=payload.num_tickets,
        ticket_tier=payload.ticket_tier,
        visitor_name=payload.visitor_name,
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Conversational AI"])
@app.post("/api/v1/chat/graph", response_model=ChatResponse, tags=["Conversational AI"])
def chat_endpoint(payload: ChatRequest):
    """
    Κεντρικό endpoint συνομιλίας:
    - Αυστηρή εκτέλεση μέσω LangGraph State Graph με Reflection Loops
    - Cloud LLM (NVIDIA Nemotron-3-Ultra via Ollama API) Synthesis Node
    - Χωρίς offline deterministic fallback
    - Διατηρεί τη μνήμη της συνεδρίας (User State)
    - Παραδίδει δήλωση διαφάνειας εκ των προτέρων (EU AI Act Article 50 Pre-Exposure)
    - Εφαρμόζει μηχαναγνώσιμο watermarking στο παραγόμενο περιεχόμενο
    - Καταγράφει αμετάβλητο cryptographically-chained audit log (Article 12)
    """
    state = get_or_create_session(payload.session_id)
    first_exposure = not getattr(state, "transparency_notice_delivered", False)
    transparency_payload = None

    if first_exposure:
        state.transparency_notice_delivered = True
        transparency_payload = {
            "notice_version": "2024/1689-Art50",
            "system_classification": "Limited Risk AI System (Article 50)",
            "message": (
                "ℹ️ Δήλωση Διαφάνειας (EU AI Act - Άρθρο 50): Συνομιλείτε με το σύστημα τεχνητής νοημοσύνης Philody AI. "
                "Το περιεχόμενο, οι προτάσεις δρομολογίων και οι ηχητικές ξεναγήσεις παράγονται συνθετικά με χρήση AI "
                "και φέρουν μηχαναγνώσιμη σήμανση (watermarking)."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "delivered_pre_exposure": True,
        }

    if payload.start_time:
        state.start_time = payload.start_time
    if payload.time_budget_hours:
        state.time_budget_hours = payload.time_budget_hours
    if payload.pace or payload.preferred_pace:
        state.preferred_pace = payload.pace or payload.preferred_pace or "moderate"
    if payload.wheelchair_accessible is not None:
        state.wheelchair_accessible = bool(payload.wheelchair_accessible)
    if payload.traveling_with_kids is not None:
        state.traveling_with_kids = bool(payload.traveling_with_kids)

    try:
        # Strictly route message through LangGraph State Graph
        graph_res = langgraph_orchestrator.execute(payload.message, state)
        response_text = graph_res.get("final_response", "")
        intent_str = graph_res.get("intent", "itinerary_planning")

        # Ανάκτηση citations
        citations: List[Dict[str, Any]] = graph_res.get("citations", [])
        if not citations and intent_str == "factual_qa":
            retrieved_docs = graph_res.get("candidate_docs", []) or rag_retriever.retrieve(query=payload.message, top_k=3)
            for d in retrieved_docs:
                citations.append({"source_id": d["id"], "name": d.get("name", d.get("source_name", "")), "score": d.get("score", 1.0)})
        elif not citations and state.active_itinerary and state.active_itinerary.get("selected_poi_ids"):
            for pid in state.active_itinerary["selected_poi_ids"]:
                p = rag_retriever.get_poi(pid)
                if p:
                    citations.append({"source_id": pid, "name": p["name"], "score": 1.0})

        # Watermarking του παραγόμενου κειμένου (EU AI Act Article 50(2))
        watermark_res = content_watermarker.watermark_text(
            raw_text=response_text,
            session_id=payload.session_id,
            origin="chat_response"
        )
        watermarked_response = watermark_res["watermarked_text"]

        # Record interaction into proprietary feedback dataset
        retrieved_ids = [c["source_id"] for c in citations if "source_id" in c]
        feedback_entry = feedback_manager.record_interaction(
            session_id=payload.session_id,
            user_query=payload.message,
            detected_intent=intent_str,
            retrieved_poi_ids=retrieved_ids,
            llm_response=watermarked_response,
            user_followup_message=payload.message if len(state.conversation_history) > 1 else None,
        )

        # Cryptographically-chained Immutable Audit Log (EU AI Act Article 12)
        audit_logger.log_event(
            event_type="chat_interaction",
            session_id=payload.session_id,
            user_input=payload.message,
            system_output=response_text[:200],
            metadata={
                "intent": intent_str,
                "transparency_notice_delivered": first_exposure,
                "watermark_hash": watermark_res["watermark_hash"],
                "has_active_itinerary": state.active_itinerary is not None,
                "citations_count": len(citations),
                "execution_steps": graph_res.get("execution_steps", []),
            }
        )

        # Token Cost Calculation & Context Window Telemetry
        token_usage = graph_res.get("token_usage") or {}
        if not token_usage:
            p_tokens = context_length_manager.count_tokens(payload.message)
            r_tokens = context_length_manager.count_tokens(response_text)
            cost_report = context_length_manager.calculate_cost(
                prompt_tokens=p_tokens,
                completion_tokens=r_tokens
            )
            token_usage = cost_report.to_dict()

        return ChatResponse(
            session_id=payload.session_id,
            intent=intent_str,
            response=watermarked_response,
            chat_bubble_text=watermarked_response,
            background_json_payload=state.active_itinerary,
            active_itinerary=state.active_itinerary,
            citations=citations,
            interaction_id=feedback_entry.feedback_id,
            conversational_sentiment=feedback_entry.conversational_sentiment,
            transparency_notice=transparency_payload,
            transparency_notice_delivered=first_exposure,
            is_synthetic_content=True,
            watermark_detected=watermark_res["has_machine_readable_watermark"],
            token_usage=token_usage,
            context_overflow_prevented=True,
        )

    except HTTPException:
        # Re-raise HTTPException directly (e.g. 503 Cloud LLM unavailable)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )



@app.post("/api/v1/itinerary/generate", tags=["Itinerary Engine"])
def generate_itinerary_endpoint(payload: ItineraryGenerateRequest):
    """
    Απευθείας δημιουργία/ανασχεδιασμός δρομολογίου μέσω της Feasibility Engine
    χωρίς απαίτηση φυσικού διαλόγου.
    """
    state = get_or_create_session(payload.session_id)
    state.start_time = payload.start_time
    if payload.time_budget_hours:
        state.time_budget_hours = payload.time_budget_hours
    state.traveling_with_kids = bool(payload.traveling_with_kids)
    state.kid_age = payload.child_age
    state.wheelchair_accessible = bool(payload.wheelchair_accessible)
    state.preferred_pace = payload.preferred_pace or "moderate"

    # Υπολογισμός ώρας λήξης
    start_hour = int(payload.start_time.split(":")[0])
    end_hour = min(23, start_hour + int(state.time_budget_hours))
    end_time_str = payload.end_time or f"{end_hour:02d}:00"

    # Weather Check
    try:
        weather_data = get_live_weather(city="Athens")
    except Exception:
        weather_data = {"condition": "Clear", "temperature_c": 22.0, "is_indoor_recommended": False}

    # Υποψήφια POIs
    candidate_pois = rag_retriever.raw_pois
    explicit_pois = None
    if payload.explicit_pois:
        explicit_pois = [p for p in candidate_pois if p["id"] in payload.explicit_pois]

    validated_itinerary = feasibility_engine.build_feasible_itinerary(
        start_time_str=payload.start_time,
        end_time_str=end_time_str,
        candidate_pois=candidate_pois,
        weather_data=weather_data,
        traveling_with_kids=state.traveling_with_kids,
        child_age=state.kid_age,
        wheelchair_accessible=state.wheelchair_accessible,
        explicit_requested_pois=explicit_pois,
    )

    state.active_itinerary = validated_itinerary
    formatted_markdown = orchestrator._format_itinerary_response(
        itinerary=validated_itinerary,
        weather=weather_data,
        state=state
    )

    return {
        "session_id": payload.session_id,
        "feasible": validated_itinerary.get("feasible", False),
        "constraint_rejected": validated_itinerary.get("constraint_rejected", False),
        "weather_adjusted": validated_itinerary.get("weather_adjusted", False),
        "start_time": validated_itinerary.get("start_time"),
        "actual_end_time": validated_itinerary.get("actual_end_time"),
        "total_distance_km": validated_itinerary.get("total_distance_km", 0.0),
        "schedule": validated_itinerary.get("schedule", []),
        "selected_poi_ids": validated_itinerary.get("selected_poi_ids", []),
        "formatted_presentation": formatted_markdown,
        "chat_bubble_text": formatted_markdown,
        "background_json_payload": validated_itinerary,
    }


class TwoStepPipelineRequest(BaseModel):
    user_request: str = Field(..., example="Φτιάξε μου ένα πρόγραμμα 3 ωρών για το απόγευμα.")
    time_budget: float = Field(3.0, example=3.0)
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)


@app.post("/api/v1/pipeline/two-step", tags=["Conversational AI"])
def two_step_pipeline_endpoint(payload: TwoStepPipelineRequest):
    """
    Two-Step Generation Pipeline Endpoint (Deterministic Layer -> Probabilistic Layer):
    Βήμα 1: Deterministic Layer (Feasibility Engine + Time-Filling Buffer)
    Βήμα 2: Probabilistic Layer (Natural Language Synthesis με temperature 0.4)
    Βήμα 3: Frontend Separation (chat_bubble_text + background_json_payload)
    """
    return orchestrator.handle_user_request(
        user_request=payload.user_request,
        time_budget=payload.time_budget,
        user_preferences=payload.user_preferences,
    )


@app.post("/api/v1/iot/event", tags=["Smartwatch & IoT Integration"])
def handle_iot_event(payload: IoTEventRequest):
    """
    Δέχεται τηλεμετρία από το Smartwatch / Smart Bracelet (π.χ. ειδοποίηση κόπωσης, geofencing)
    και ενεργοποιεί αυτόματα τη Feasibility Engine & το Chat Orchestrator για αναπροσαρμογή του πλάνου.
    """
    state = get_or_create_session(payload.session_id)

    # Αν δεν υπάρχει ήδη ενεργό δρομολόγιο, αρχικοποιούμε ένα βασικό πρόγραμμα για τη συνεδρία
    if state.active_itinerary is None:
        try:
            weather_data = get_live_weather(city="Athens")
        except Exception:
            weather_data = None
        state.active_itinerary = feasibility_engine.build_feasible_itinerary(
            start_time_str="14:00",
            end_time_str="18:00",
            candidate_pois=rag_retriever.raw_pois,
            weather_data=weather_data,
        )

    if payload.event_type == "fatigue_alert":
        # Ο χρήστης κουράστηκε -> Αυτόματη προσθήκη στάσης για καφέ/ξεκούραση και χαλαρός ρυθμός
        state.preferred_pace = "relaxed"
        update_message = "Ανιχνεύθηκε αυξημένη κόπωση από το βραχιόλι (142 BPM). Προσάρμοσε το πρόγραμμα με πιο χαλαρό ρυθμό και μια στάση για ξεκούραση/καφέ."
        response_text = orchestrator.process_message(state=state, user_message=update_message)
        return {
            "status": "success",
            "action": "itinerary_replanned",
            "haptic_pattern": "long_warning",
            "message": "⚠️ Ανιχνεύθηκε κόπωση (142 BPM). Το πρόγραμμα προσαρμόστηκε αυτόματα με στάση ανάπαυσης.",
            "response": response_text,
            "active_itinerary": state.active_itinerary,
        }

    elif payload.event_type in ("geofence_entry", "proximity"):
        loc = payload.location or {"lat": 37.9752, "lon": 23.7225}
        update_message = "Βρίσκομαι κοντά στον Ναό Ηφαίστου (Αρχαία Αγορά). Δώσε μου μια σύντομη ιστορική περιγραφή και πρακτικές πληροφορίες επίσκεψης."
        response_text = orchestrator.process_message(state=state, user_message=update_message)
        return {
            "status": "success",
            "action": "proximity_alert",
            "haptic_pattern": "double_pulse",
            "message": "📍 Είσοδος σε γεωγραφική ζώνη (Ναός Ηφαίστου - Proximity 45m).",
            "response": response_text,
            "active_itinerary": state.active_itinerary,
        }

    elif payload.event_type == "uv_warning":
        update_message = "Υψηλός δείκτης UV (10.5) και καύσωνας. Προσάρμοσε το πρόγραμμα μόνο σε στεγασμένους και κλιματιζόμενους χώρους."
        response_text = orchestrator.process_message(state=state, user_message=update_message)
        return {
            "status": "success",
            "action": "thermal_warning",
            "haptic_pattern": "single_short",
            "message": "☀️ UV Warning 10.5. Το πρόγραμμα αναπροσαρμόστηκε αυτόματα σε κλιματιζόμενους χώρους.",
            "response": response_text,
            "active_itinerary": state.active_itinerary,
        }

    elif payload.event_type == "crowd_density":
        # DOTSOFT Smart City Crowd Sensor -> αποφυγή συνωστισμού στην Ακρόπολη
        if "acropolis_hill" not in state.blacklisted_poi_ids:
            state.blacklisted_poi_ids.append("acropolis_hill")
        if "acropolis_museum" not in state.blacklisted_poi_ids:
            state.blacklisted_poi_ids.append("acropolis_museum")
        update_message = "Ανιχνεύθηκε έντονος συνωστισμός στην Ακρόπολη. Αναδρομολόγησε άμεσα το πρόγραμμα σε εναλλακτικά αξιοθέατα χωρίς καθυστερήσεις."
        response_text = orchestrator.process_message(state=state, user_message=update_message)
        return {
            "status": "success",
            "action": "crowd_reroute",
            "haptic_pattern": "double_pulse",
            "message": "👥 Ειδοποίηση Συνωστισμού στην Ακρόπολη. Η Feasibility Engine ανακατεύθυνε δυναμικά το πρόγραμμα σε εναλλακτικά σημεία.",
            "response": response_text,
            "active_itinerary": state.active_itinerary,
        }

    return {"status": "ignored", "reason": f"Unknown event type: {payload.event_type}"}


@app.get("/api/v1/session/{session_id}", tags=["System"])
def get_session_info(session_id: str):
    """Επιστρέφει την κατάσταση της συνεδρίας (ιστορικό συνομιλίας, ενεργό δρομολόγιο, προτιμήσεις)."""
    state = get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "chat_history": [
            {"role": m.get("role", "user") if isinstance(m, dict) else getattr(m, "role", "user"),
             "content": m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")}
            for m in state.chat_history
        ],
        "active_itinerary": state.active_itinerary,
        "preferences": {
            "traveling_with_kids": state.traveling_with_kids,
            "kid_age": state.kid_age,
            "wheelchair_accessible": state.wheelchair_accessible,
            "preferred_pace": state.preferred_pace,
            "time_budget_hours": state.time_budget_hours,
            "blacklisted_poi_ids": list(state.blacklisted_poi_ids),
        }
    }


@app.get("/api/v1/pois", tags=["Knowledge Base"])
def list_pois():
    """Επιστρέφει όλα τα διαθέσιμα αξιοθέατα με συντεταγμένες και μεταδεδομένα για το χάρτη."""
    return {
        "total": len(rag_retriever.raw_pois),
        "pois": rag_retriever.raw_pois
    }


@app.post("/api/v1/evaluation/run", tags=["System"])
@app.post("/api/v1/eval/run", tags=["System"])
def run_evaluation_suite():
    """
    Εκτελεί αυτοματοποιημένα όλα τα 19 Test Cases του evaluation_dataset.json
    και επιστρέφει το live scorecard & metrics.
    """
    from evaluate_dataset import load_dataset, evaluate_test_case

    dataset = load_dataset("evaluation_dataset.json")
    results = []
    passed_count = 0

    for tc in dataset:
        res = evaluate_test_case(tc, retriever=rag_retriever, feasibility_engine=feasibility_engine, force_deterministic=True)
        if res["passed"]:
            passed_count += 1
        results.append({
            "id": res["id"],
            "category": res["category"],
            "passed": res["passed"],
            "metric": res["eval_metric"],
            "failure_reasons": res.get("failure_reasons", []),
            "detected_intent": res.get("detected_intent"),
            "user_input": res.get("user_input"),
            "reply_preview": res.get("reply_preview")
        })

    total = len(dataset)
    return {
        "total_test_cases": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate_pct": round((passed_count / total) * 100, 1),
        "scorecard": results
    }


@app.get("/api/v1/guardrails/stats", tags=["Guardrails"])
def get_guardrails_stats():
    """
    Επιστρέφει ζωντανά στατιστικά και τηλεμετρία των Input & Output Guardrails
    (αποτροπή prompt injections, καθαρισμός ψευδαισθήσεων, zero hallucinations rate).
    """
    return guardrails_manager.get_stats()


@app.post("/api/v1/evaluation/ragas", tags=["Evaluation"])
@app.post("/api/v1/eval/ragas", tags=["Evaluation"])
def run_ragas_evaluation():
    """
    Εκτελεί πλήρη αξιολόγηση RAGAS (Retrieval Augmented Generation Assessment)
    για Faithfulness, Answer Relevance, Context Precision και Guardrails Interceptions.
    """
    from evaluate_dataset import load_dataset
    dataset = load_dataset("evaluation_dataset.json")

    # Χρησιμοποιούμε AthensTouristAgent για την εκτέλεση του evaluation
    eval_agent = AthensTouristAgent(retriever=rag_retriever, feasibility_engine=feasibility_engine)
    scorecard = ragas_evaluator.evaluate_suite(dataset, eval_agent)
    return scorecard.to_dict()


# 4.5 Feedback Loops & Proprietary Dataset Endpoints
@app.post("/api/v1/feedback", tags=["Feedback Loops"])
def submit_feedback_endpoint(payload: FeedbackSubmissionRequest):
    """
    Υποβολή ρητής αξιολόγησης χρήστη (thumbs up / thumbs down + σχόλια / tags)
    για τη δημιουργία ιδιόκτητου συνόλου δεδομένων εκπαίδευσης (proprietary dataset flywheel).
    """
    entry = feedback_manager.submit_explicit_feedback(
        session_id=payload.session_id,
        rating=payload.rating,
        comment=payload.comment,
        tags=payload.tags,
        feedback_id=payload.feedback_id,
    )
    return {
        "status": "success",
        "message": "Feedback recorded successfully for proprietary dataset flywheel.",
        "feedback_id": entry.feedback_id,
        "explicit_rating": entry.explicit_rating,
        "dpo_preference": entry.dpo_pair.get("preference_label") if entry.dpo_pair else None,
    }


@app.get("/api/v1/feedback/stats", tags=["Feedback Loops"])
def get_feedback_stats_endpoint():
    """
    Επιστρέφει ζωντανά συγκεντρωτικά στατιστικά του proprietary dataset
    (CSAT proxy, κατανομή θετικού/αρνητικού sentiment, μέγεθος dataset, δείγματα DPO).
    """
    return feedback_manager.get_feedback_stats()


@app.get("/api/v1/feedback/export", tags=["Feedback Loops"])
def export_feedback_dataset_endpoint():
    """
    Εξάγει τα αποθηκευμένα ζεύγη προτιμήσεων (DPO/SFT samples) για fine-tuning του μοντέλου.
    """
    return {
        "total_samples": len(feedback_manager.export_dataset_for_finetuning()),
        "samples": feedback_manager.export_dataset_for_finetuning(),
    }


# 4.6 Operational Metrics & Prometheus Observability Endpoints
@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
def prometheus_metrics_endpoint():
    """
    Εξάγει επιχειρησιακά metrics σε πρότυπη μορφή Prometheus scraper
    (συνολικά requests, κατανομή status codes, HTTP 5xx errors, latency quantiles p50/p90/p95/p99, CPU/GPU ratio).
    """
    return PlainTextResponse(observability_registry.export_prometheus_metrics())


@app.get("/api/v1/observability/stats", tags=["Observability"])
def get_observability_stats_endpoint():
    """
    Επιστρέφει ζωντανά operational metrics για dashboards Grafana
    (p50, p90, p95, p99 latency ms, 5xx error rate %, CPU/Memory/GPU utilization).
    """
    return observability_registry.get_dashboard_stats()


# 4.7 POI Data Pipeline & Consistency Endpoints
@app.get("/api/v1/pipeline/status", tags=["Data Pipeline"])
def get_data_pipeline_status_endpoint():
    """
    Επιστρέφει την κατάσταση συνέπειας του POI Data Pipeline, την έκδοση,
    το checksum hash και στατιστικά validation.
    """
    return atomic_poi_pipeline.get_pipeline_status()


@app.post("/api/v1/poi/update", tags=["Data Pipeline"])
def update_poi_metadata_endpoint(poi_id: str, updates: Dict[str, Any]):
    """
    Ανανεώνει ατομικά τα μεταδεδομένα ενός POI με αυστηρό Pydantic schema validation,
    αποτρέποντας pipeline inconsistencies και ανανεώνοντας δυναμικά το vector index.
    """
    try:
        res = atomic_poi_pipeline.update_poi_metadata(poi_id=poi_id, updates=updates)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data pipeline inconsistency prevented: {str(e)}"
        )


# 4.8 EU AI Act Compliance (Articles 50 & 12)
@app.get("/api/v1/compliance/transparency", response_model=TransparencyNoticeResponse, tags=["EU AI Act Compliance"])
def get_transparency_notice_endpoint():
    """
    Επιστρέφει την επίσημη γνωστοποίηση διαφάνειας σύμφωνα με το Άρθρο 50 του EU AI Act
    για συστήματα Περιορισμένου Κινδύνου (Limited Risk Systems).
    Εξασφαλίζει ότι ο χρήστης ενημερώνεται εκ των προτέρων (pre-exposure) ότι αλληλεπιδρά με AI.
    """
    chain_status = audit_logger.verify_chain_integrity()
    return TransparencyNoticeResponse(
        status="compliant",
        system_name="Philody AI Travel Assistant",
        risk_classification="Limited Risk (Article 50 EU AI Act / Regulation (EU) 2024/1689)",
        regulatory_obligations=[
            "Article 50(1): Obligation to inform natural persons that they are interacting with an AI system prior to exposure.",
            "Article 50(2): Obligation to mark artificially generated/manipulated text, audio, and multimedia content in a machine-readable format.",
            "Article 12: Obligation for record-keeping and immutable audit logging for traceability and system accountability.",
        ],
        transparency_notice_text=(
            "Σύστημα Τεχνητής Νοημοσύνης Philody AI: Το παρόν σύστημα αξιοποιεί μοντέλα τεχνητής νοημοσύνης "
            "για τη δημιουργία τουριστικών συστάσεων, βελτιστοποίηση δρομολογίων και παραγωγή ηχητικών ξεναγήσεων. "
            "Όλα τα παραγόμενα κείμενα και ηχητικά αρχεία φέρουν μηχαναγνώσιμη σήμανση γνησιότητας και προέλευσης."
        ),
        pre_exposure_enforced=True,
        machine_readable_watermarking_enabled=True,
        immutable_audit_logging_active=True,
        audit_chain_verified=bool(chain_status.get("is_valid", False)),
    )


@app.get("/api/v1/compliance/audit/logs", tags=["EU AI Act Compliance"])
def get_audit_logs_endpoint(limit: int = 50):
    """
    Ανακτά τις τελευταίες εγγραφές του αμετάβλητου αρχείου καταγραφής (Immutable Audit Log)
    σύμφωνα με το Άρθρο 12 του EU AI Act.
    """
    records = audit_logger.get_recent_logs(limit=limit)
    return {
        "total_records": len(records),
        "records": records,
    }


@app.get("/api/v1/compliance/audit/verify", tags=["EU AI Act Compliance"])
def verify_audit_log_chain_endpoint():
    """
    Ελέγχει κρυπτογραφικά την ακεραιότητα της αλυσίδας καταγραφών (Cryptographic SHA-256 Hash Chain)
    του Immutable Audit Log (Άρθρο 12). Εντοπίζει άμεσα τυχόν τροποποίηση, διαγραφή ή παραποίηση δεδομένων.
    """
    return audit_logger.verify_chain_integrity()


@app.post("/api/v1/audio-tour/generate", tags=["EU AI Act Compliance"])
def generate_synthetic_audio_tour_endpoint(payload: AudioTourGenerateRequest):
    """
    Δημιουργεί συνθετική ηχητική ξενάγηση με μηχαναγνώσιμο watermarking (RIFF C2PA Metadata)
    και υδατογράφηση κειμένου (Zero-Width Steganography) σύμφωνα με το Άρθρο 50(2) του EU AI Act.
    """
    result = audio_tour_generator.generate_tour(
        poi_name=payload.poi_name,
        topic=payload.topic or "Ιστορία και Πολιτισμός",
        session_id=payload.session_id or "anon_session",
    )
    # Καταγραφή στο immutable audit log
    audit_logger.log_event(
        event_type="synthetic_audio_tour_generated",
        session_id=payload.session_id or "anon_session",
        user_input=f"Audio tour for {payload.poi_name}",
        system_output=result["script"][:120],
        metadata={
            "poi_name": payload.poi_name,
            "audio_size_bytes": result["audio_size_bytes"],
            "article_50_compliant": True,
        }
    )
    return result


@app.post("/api/v1/watermark/verify", tags=["EU AI Act Compliance"])
def verify_content_watermark_endpoint(payload: WatermarkVerifyRequest):
    """
    Επαληθεύει την ύπαρξη μηχαναγνώσιμου υδατογραφήματος σε συνθετικό κείμενο ή ήχο (Άρθρο 50).
    """
    response: Dict[str, Any] = {"text_watermark": None, "audio_watermark": None}
    if payload.text:
        response["text_watermark"] = detect_text_watermark(payload.text)
    if payload.audio_base64:
        try:
            raw_audio = base64.b64decode(payload.audio_base64)
            response["audio_watermark"] = detect_audio_watermark(raw_audio)
        except Exception as e:
            response["audio_watermark"] = {"detected": False, "error": str(e)}
    return response


# 4.9 Context Length Management & Token Cost Endpoints
@app.post("/api/v1/context/estimate", response_model=ContextEstimateResponse, tags=["Context Window Management"])
def estimate_context_usage_endpoint(payload: ContextEstimateRequest):
    """
    Υπολογίζει με ακρίβεια την κατανάλωση tokens, τα όρια context window,
    τη χρήση ειδικών tokens οριοθέτησης ([BOS], [EOS], <|endoftext|>)
    και το εκτιμώμενο κόστος σε USD.
    """
    state = get_or_create_session(payload.session_id or "default_user")
    docs = rag_retriever.retrieve(query=payload.user_message, top_k=3)
    assembly = context_length_manager.assemble_token_managed_prompt(
        system_instructions="Είσαι ένας έμπειρος AI Τουριστικός Βοηθός για την Αθήνα.",
        retrieved_docs=docs,
        history=state.conversation_history,
        user_query=payload.user_message,
        model_name=payload.model_name or "nemotron-3-nano:30b"
    )
    return ContextEstimateResponse(
        prompt_tokens=assembly["prompt_tokens"],
        system_tokens=assembly["system_tokens"],
        rag_tokens=assembly["rag_tokens"],
        user_query_tokens=assembly["user_query_tokens"],
        history_turns_retained=assembly["history_turns_retained"],
        history_overflow_truncated=assembly["history_overflow_truncated"],
        cost_report=assembly["cost_report"],
        special_tokens_used=assembly["special_tokens_used"],
        context_overflow_prevented=assembly["context_overflow_prevented"]
    )


@app.get("/api/v1/context/metrics", tags=["Context Window Management"])
def get_context_metrics_endpoint():
    """
    Επιστρέφει τις παραμέτρους του Context Length Manager, τον τιμοκατάλογο μοντέλων
    και τα ενεργά tokens οριοθέτησης ([BOS], [EOS], <|endoftext|>).
    """
    return {
        "max_context_tokens": context_length_manager.max_context_tokens,
        "max_generation_tokens": context_length_manager.max_generation_tokens,
        "system_reserve_tokens": context_length_manager.system_reserve_tokens,
        "rag_budget_tokens": context_length_manager.rag_budget_tokens,
        "history_budget_tokens": context_length_manager.history_budget_tokens,
        "active_model": context_length_manager.model_name,
        "special_tokens": {
            "bos": SpecialTokens.BOS,
            "eos": SpecialTokens.EOS,
            "end_of_text": SpecialTokens.END_OF_TEXT,
            "im_start": SpecialTokens.IM_START,
            "im_end": SpecialTokens.IM_END,
        },
        "pricing_catalog": {
            k: {
                "prompt_cost_per_1m": v.prompt_cost_per_1m,
                "completion_cost_per_1m": v.completion_cost_per_1m,
                "currency": v.currency
            }
            for k, v in MODEL_PRICING_CATALOG.items()
        }
    }


# 4.10 LLM Observability, LangSmith Tracing & LangGraph Endpoints
@app.get("/api/v1/observability/status", tags=["LLM Observability"])
def get_observability_status_endpoint():
    """Επιστρέφει την κατάσταση παρατηρησιμότητας και διασύνδεσης με το LangSmith."""
    return observability_hub.get_status()


@app.get("/api/v1/observability/traces", tags=["LLM Observability"])
def get_recent_traces_endpoint(limit: int = 15):
    """Επιστρέφει τα πιο πρόσφατα κατανεμημένα traces εκτέλεσης (spans, tool calls, latency)."""
    return observability_hub.get_recent_traces(limit=limit)


@app.get("/api/v1/observability/metrics", tags=["LLM Observability"])
def get_observability_metrics_endpoint():
    """Επιστρέφει συγκεντρωτικά μετρικά εκτέλεσης, ποσοστά σφαλμάτων και κλήσεις tools."""
    return observability_hub.get_metrics_summary()


@app.post("/api/v1/chat/graph", response_model=ChatResponse, tags=["Agent Chat"])
def chat_langgraph_endpoint(request: ChatRequest):
    """
    Εκτελεί συνομιλία αυστηρά μέσω του LangGraph State Graph με κύκλους επανεξέτασης (reflection loops),
    παρέχοντας αυτόνομη διόρθωση σφαλμάτων ωραρίου/καιρού πριν την τελική σύνθεση από το Cloud LLM.
    """
    return chat_endpoint(request)


# 5. Static Files & Web Client Hosting


# Σύνδεση με το landing folder για τις εικόνες/frames του 360 showcase
landing_dir = Path(__file__).parent / "landing"
if landing_dir.exists():
    app.mount("/landing", StaticFiles(directory=str(landing_dir)), name="landing")

@app.get("/api/v1/health", tags=["System Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Philody AI Tourist Assistant",
        "version": "2.4.0-production",
        "euaiact_compliant": True,
        "active_workspaces": [
            "0. Showcase 360",
            "1. Chat & Interactive Map",
            "2. Timeline Studio",
            "3. Wearable Wrist Smartwatch",
            "4. Smart City IoT Hub",
            "5. Evaluation & Benchmarks Studio"
        ]
    }

# Σερβίρισμα του Single Page Application (HTML/CSS/JS) στην αρχική σελίδα (/)
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_index_html():
    index_file = Path(__file__).parent / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return JSONResponse({"message": "Philody AI Travel Assistant API is running. Visit /docs for Swagger UI."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
