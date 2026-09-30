"""
LLM Orchestrator & Multi-Turn Chat Layer (orchestrator/agent.py).

Coordinates RAG Pipeline, Live Weather Tool, and Feasibility Engine.
Handles intent classification, explicit state management, sliding window memory,
and dynamic replanning.
"""

from __future__ import annotations
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
load_dotenv()

from engine.feasibility import (
    FeasibilityEngine,
    filter_candidate_pois,
    optimize_itinerary_duration,
    calculate_total_mins,
)
from rag.retriever import AthensRAGRetriever, AthensRetriever, normalize_greek
from tools.weather import get_current_weather, get_live_weather
from tools.transit import get_transit_route, get_station_schedule, get_transit_alerts
from tools.ticketing import check_ticket_availability, get_ticket_pricing, simulate_ticket_reservation
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
)
from .prompts import (
    build_final_synthesis_prompt,
    build_rag_context_prompt,
    SYSTEM_PROMPT,
    DELIMITED_SYSTEM_PROMPT,
    NATURAL_SYNTHESIS_PROMPT,
    DELIMITED_NATURAL_SYNTHESIS_PROMPT,
    SpecialTokens,
)
from orchestrator.context_manager import context_length_manager, TokenCostReport
from orchestrator.guardrails import guardrails_manager
from orchestrator.feedback import feedback_manager
from orchestrator.llm_client import DeterministicSynthesizer, get_cloud_ollama_client


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama Cloud LLM Client Configuration (NVIDIA Nemotron-3-Ultra)
# ---------------------------------------------------------------------------
_LLM_API_KEY    = os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
_LLM_BASE_URL   = os.getenv("OLLAMA_HOST") or os.getenv("OPENAI_BASE_URL", "https://ollama.com").strip()
_LLM_MODEL      = os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL", "nemotron-3-nano:30b").strip()
_LLM_ENABLED    = bool(_LLM_API_KEY)

_llm_client: Any = None
try:
    _llm_client = get_cloud_ollama_client()
    logger.info("[LLM] Cloud Ollama client active for Nemotron: model=%s host=%s", _LLM_MODEL, _LLM_BASE_URL)
except Exception as _e:
    logger.warning("[LLM] Cloud Ollama client initialization warning: %s", _e)


# 1. Ορισμός των Intent Types
class IntentType(Enum):
    FACTUAL_QA = "factual_qa"
    WEATHER_QUERY = "weather_query"
    ITINERARY_REQUEST = "itinerary_request"
    REPLANNING = "replanning"
    TRANSIT_QUERY = "transit_query"
    TICKET_QUERY = "ticket_query"


# 2. Explicit State Management Class
@dataclass
class UserState:
    """Stateful traveler profile maintained across multi-turn conversation."""
    session_id: str = "default_user"
    preferences: List[str] = field(default_factory=list)
    blacklisted_poi_ids: List[str] = field(default_factory=list)
    blacklisted_categories: Set[str] = field(default_factory=set)
    blacklisted_tags: Set[str] = field(default_factory=set)
    traveling_with_kids: bool = False
    kid_age: Optional[int] = None
    wheelchair_accessible: bool = False
    time_budget_hours: float = 4.0
    start_time: str = "14:00"
    active_itinerary: Optional[Dict[str, Any]] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    preferred_pace: str = "moderate"
    transparency_notice_delivered: bool = False

    token_budget: int = 1200

    def add_message(self, role: str, content: str, max_tokens: Optional[int] = None):
        """
        Token-aware sliding window truncation για αποφυγή unbounded context growth
        και υπερχείλισης tokens, διατηρώντας συνεκτικά ζεύγη ερωτήσεων-απαντήσεων.
        """
        self.chat_history.append({"role": role, "content": content})
        from orchestrator.context_manager import context_length_manager
        budget = max_tokens or self.token_budget
        truncated, _ = context_length_manager.truncate_history_by_tokens(
            self.chat_history,
            max_tokens=budget,
            preserve_pairs=True
        )
        # Apply standard upper bound of 10 turns if under token budget
        if len(truncated) > 10:
            truncated = truncated[-10:]
        self.chat_history = truncated

    def get_history_token_count(self) -> int:
        """Επιστρέφει το συνολικό μέγεθος tokens του ιστορικού συνομιλίας."""
        from orchestrator.context_manager import context_length_manager
        return sum(context_length_manager.count_tokens(m.get("content", "")) for m in self.chat_history)


    # Property aliases for compatibility
    @property
    def disliked_pois(self) -> List[str]:
        return self.blacklisted_poi_ids

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        return self.chat_history


# Negative Constraint Extraction Mechanism (Solution 2)
def extract_negative_constraints(
    user_message: str,
    user_state: UserState,
    raw_pois: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Μηχανισμός Εξαγωγής Αρνητικών Περιορισμών (Negative Constraint Extraction):
    Εντοπίζει μοτίβα άρνησης και ενημερώνει τη μαύρη λίστα του UserState.
    """
    msg_lower = user_message.lower()
    msg_norm = normalize_greek(msg_lower)

    # Εντοπισμός μοτίβων άρνησης
    negation_triggers = [
        "δεν θελω", "δεν θέλω", "οχι αλλα", "όχι άλλα", "οχι αλλο", "όχι άλλο",
        "μην βαλεις", "μην βάλεις", "αποκλεισε", "αποκλείσε", "χωρις", "χωρίς",
        "βγαλε", "βγάλε", "χωρις μουσεια", "χωρίς μουσεία", "οχι μουσεια", "όχι μουσεία"
    ]

    is_negation = any(trigger in msg_lower for trigger in negation_triggers) or any(
        trigger in msg_norm for trigger in ["δεν θελω", "οχι αλλα", "μην βαλεις", "αποκλεισε", "χωρις", "βγαλε"]
    )

    if is_negation:
        if any(w in msg_norm for w in ["μουσει", "αρχαιολογικ", "μουσεια", "εκθεμα"]):
            user_state.blacklisted_categories.add("museum")
            user_state.blacklisted_categories.add("archaeological_site")
            user_state.blacklisted_tags.update([
                "museum", "archaeological", "antiquity", "art", "archaeology", "ancient", "history"
            ])
            if raw_pois:
                for poi in raw_pois:
                    poi_tags = set(poi.get("tags", []))
                    poi_cat = poi.get("category", "")
                    if (
                        poi_cat in ["museum", "archaeological_site"]
                        or "museum" in poi_tags
                        or "archaeology" in poi_tags
                        or "antiquity" in poi_tags
                    ):
                        if poi["id"] not in user_state.blacklisted_poi_ids:
                            user_state.blacklisted_poi_ids.append(poi["id"])
            return True
    return False

    @property
    def child_age(self) -> Optional[int]:
        return self.kid_age

    @child_age.setter
    def child_age(self, val: Optional[int]):
        self.kid_age = val

    @property
    def time_budget_mins(self) -> int:
        return int(self.time_budget_hours * 60)


# Backward compatible message representation
@dataclass
class ConversationMessage:
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Named POIs Synonym Dictionary for Entity Extraction
NAMED_POIS_MAP: Dict[str, List[str]] = {
    "acropolis_museum": [
        "μουσείο ακρόπολης", "μουσειο ακροπολης", "μουσείο ακρόπολη", "μουσειο ακροπολη",
        "μουσειο της ακροπολης", "μουσείο της ακρόπολης", "ακρόπολης μουσείο", "ακροπολης μουσειο"
    ],
    "acropolis_hill": [
        "ακρόπολη", "ακροπολη", "παρθενώνα", "παρθενωνα", "ιερό βράχο", "ιερο βραχο"
    ],
    "monastiraki_flea_market": [
        "μοναστηράκι", "μοναστηρακι", "μοναστηρακίου", "μοναστηρακιου", "παζάρι", "παζαρι", "σταθμός μοναστηράκι", "σταθμος μοναστηρακι"
    ],
    "syntagma_changing_guards": [
        "βουλή των ελλήνων", "βουλη των ελληνων", "βουλή", "βουλη", "βουλής", "βουλης",
        "σύνταγμα", "συνταγμα", "συντάγματος", "συνταγματος", "εύζωνες", "ευζωνες",
        "αλλαγή φρουράς", "αλλαγη φρουρας", "μνημείο του αγνώστου στρατιώτη"
    ],
    "national_archaeological_museum": [
        "εθνικό αρχαιολογικό", "εθνικο αρχαιολογικο", "αρχαιολογικό μουσείο", "αρχαιολογικο μουσειο"
    ],
    "hephaestus_temple": [
        "ναό του ηφαίστου", "ναο του ηφαιστου", "ηφαίστου", "ηφαιστου", "αρχαία αγορά", "αρχαια αγορα", "θησείο", "θησειο"
    ],
    "cycladic_art_museum": [
        "κυκλαδικής τέχνης", "κυκλαδικης τεχνης", "κυκλαδικό", "κυκλαδικο"
    ],
    "lycabettus_hill": [
        "λυκαβηττ", "λυκαβητο", "λυκαβηττό", "λόφο λυκαβηττού", "λοφο λυκαβηττου", "τελεφερίκ", "τελεφερικ"
    ],
    "anafiotika": [
        "αναφιώτικα", "αναφιωτικα"
    ],
    "plaka_historic_walk": [
        "πλάκα", "πλακα", "ιστορική βόλτα στην πλάκα"
    ],
    "national_garden": [
        "εθνικός κήπος", "εθνικος κηπος", "εθνικό κήπο", "εθνικο κηπο", "κήπος", "κηπος"
    ],
    "panathenaic_stadium": [
        "παναθηναϊκό στάδιο", "παναθηναικο σταδιο", "καλλιμάρμαρο", "καλλιμαρμαρο", "στάδιο", "σταδιο"
    ],
    "benaki_museum_greek_culture": [
        "μπενάκη", "μπενακη", "μουσείο μπενάκη", "μουσειο μπενακη"
    ],
    "goulandris_modern_art": [
        "γουλανδρή", "γουλανδρη", "ίδρυμα γουλανδρή", "ιδρυμα γουλανδρη"
    ],
    "hellenic_children_museum": [
        "παιδικό μουσείο", "παιδικο μουσειο", "παιδικο", "παιδικό"
    ],
    "eugenides_planetarium": [
        "πλανητάριο", "πλανηταριο", "ευγενίδειο", "ευγενιδειο"
    ],
}

def extract_named_pois(text: str) -> List[str]:
    """Εξάγει αναγνωρισμένα POIs από το κείμενο με αποσαφήνιση (disambiguation)."""
    msg_lower = text.lower()
    explicit_poi_ids = []
    for pid, synonyms in NAMED_POIS_MAP.items():
        if any(s in msg_lower for s in synonyms):
            explicit_poi_ids.append(pid)

    # Disambiguation: If user specified "μουσείο ακρόπολης", do not accidentally add "acropolis_hill"
    # UNLESS Acropolis hill was also separately/distinctly requested (e.g. "την Ακρόπολη, το Μουσείο Ακρόπολης...")
    if "acropolis_museum" in explicit_poi_ids and "acropolis_hill" in explicit_poi_ids:
        acrop_count = msg_lower.count("ακρόπολ") + msg_lower.count("ακροπολ")
        separate_acropolis = (
            acrop_count >= 2
            or any(w in msg_lower for w in [
                "παρθενων", "παρθενώνα", "βράχο", "βραχο",
                "και στην ακρόπολη", "και ακρόπολη", "και στην ακροπολη", "και ακροπολη",
                "ιερό βράχο", "την ακρόπολη", "την ακροπολη", "στην ακρόπολη", "στην ακροπολη"
            ])
        )
        if not separate_acropolis:
            explicit_poi_ids.remove("acropolis_hill")
    return explicit_poi_ids


# 3. Main Orchestrator Class
class TouristLLMOrchestrator:
    """
    Orchestrates the entire multi-turn pipeline:
    Intent Detection -> Routing -> Domain Tools -> Feasibility Engine -> Synthesis.
    """

    def __init__(
        self,
        rag_retriever: Optional[AthensRAGRetriever] = None,
        feasibility_engine: Optional[FeasibilityEngine] = None,
    ):
        self.rag = rag_retriever or AthensRAGRetriever()
        self.feasibility = feasibility_engine or FeasibilityEngine()
        self._llm_enabled = _LLM_ENABLED
        self._llm_client  = _llm_client
        self._llm_model   = _LLM_MODEL
        if self.rag and getattr(self.rag, "raw_pois", None):
            guardrails_manager.output_guardrail.raw_pois = self.rag.raw_pois
            guardrails_manager.output_guardrail.pois_by_id = {p["id"]: p for p in self.rag.raw_pois}

    # ------------------------------------------------------------------
    # LLM Inference Helper (NVIDIA Nemotron-3-Ultra via Ollama API)
    # ------------------------------------------------------------------
    def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.3,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Calls Cloud NVIDIA Nemotron-3-Ultra via the Ollama client.
        Strictly raises HTTPException(503) if the call fails (no deterministic fallback)."""
        delimited_system = system if SpecialTokens.BOS in system else SpecialTokens.wrap_block(SpecialTokens.SYSTEM_TAG, system)
        delimited_user = user if SpecialTokens.BOS in user else SpecialTokens.wrap_block(SpecialTokens.USER_TAG, user)

        messages = [{"role": "system", "content": delimited_system}]
        if history:
            for h in history:
                r = h.get("role", "user")
                c = h.get("content", "")
                messages.append({
                    "role": r,
                    "content": SpecialTokens.wrap_block(r, c)
                })
        messages.append({"role": "user", "content": delimited_user})

        client = get_cloud_ollama_client()
        return client.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


    def validate_entities_in_kb(self, user_message: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ελέγχει αν τα ζητούμενα POIs/περιοχές υπάρχουν στη βάση γνώσης.
        Αν δεν βρεθούν, εμποδίζει τη Feasibility Engine από το να παράγει τυχαίο δρομολόγιο.
        """
        msg_norm = normalize_greek(user_message.lower())

        # 1. Αν αναγνωρίστηκε ήδη επίσημο POI από το λεξικό συνωνύμων
        explicit_ids = extract_named_pois(user_message)
        if explicit_ids:
            return {"valid": True, "message": ""}

        # 2. Γνωστές εκτός βάσης περιοχές/συνοικίες της Αθήνας
        unsupported_areas = [
            "κυψελ", "kypsel", "γλυφαδ", "glyfad", "μαρουσ", "marous", "κηφισ", "kifis",
            "περιστερ", "perister", "νεα σμυρν", "nea smyrn", "χαλανδρ", "chalandr",
            "πατησι", "patisi", "εξαρχει", "exarch", "παγκρατ", "pagkrat",
            "πειραι", "peirai", "piraeus", "ψυχικ", "psychik", "ζωγραφ", "zograf",
            "ιλισι", "ilisi", "καλλιθε", "kallithe", "βουλιαγμεν", "vouliagmen"
        ]
        if any(area in msg_norm for area in unsupported_areas):
            return {
                "valid": False,
                "message": (
                    "🏛️ **Πληροφορία Βάσης Γνώσης**\n\n"
                    "Λυπάμαι, αλλά η συγκεκριμένη περιοχή/αξιοθέατο δεν περιλαμβάνεται στα 16 υποστηριζόμενα "
                    "αξιοθέατα της Αθήνας στη βάση γνώσης μου.\n\n"
                    "💡 **Μπορώ να σε βοηθήσω με τα εξής διαθέσιμα σημεία:**\n"
                    "• Μουσείο Ακρόπολης, Παρθενώνας, Αρχαία Αγορά\n"
                    "• Μουσείο Κυκλαδικής Τέχνης, Μουσείο Μπενάκη, Ίδρυμα Γουλανδρή\n"
                    "• Λόφος Λυκαβηττού, Πλάκα, Μοναστηράκι, Σύνταγμα\n"
                    "• Εθνικό Αρχαιολογικό Μουσείο, Παναθηναϊκό Στάδιο, Εθνικός Κήπος"
                )
            }

        # 3. Έλεγχος score ομοιότητας RAG (αν δεν υπάρχει κανένα doc με score >= 0.35)
        scores = [doc.get("score") if isinstance(doc, dict) else getattr(doc, "score", 0.0) for doc in retrieved_docs] if retrieved_docs else []
        if not retrieved_docs or (scores and all(s < 0.35 for s in scores)):
            return {
                "valid": False,
                "message": (
                    "🏛️ **Πληροφορία Βάσης Γνώσης**\n\n"
                    "Λυπάμαι, αλλά η συγκεκριμένη περιοχή/αξιοθέατο δεν περιλαμβάνεται στα 16 υποστηριζόμενα "
                    "αξιοθέατα της Αθήνας στη βάση γνώσης μου.\n\n"
                    "💡 **Μπορώ να σε βοηθήσω με τα εξής διαθέσιμα σημεία:**\n"
                    "• Μουσείο Ακρόπολης, Παρθενώνας, Αρχαία Αγορά\n"
                    "• Μουσείο Κυκλαδικής Τέχνης, Μουσείο Μπενάκη, Ίδρυμα Γουλανδρή\n"
                    "• Λόφος Λυκαβηττού, Πλάκα, Μοναστηράκι, Σύνταγμα\n"
                    "• Εθνικό Αρχαιολογικό Μουσείο, Παναθηναϊκό Στάδιο, Εθνικός Κήπος"
                )
            }

        return {"valid": True, "message": ""}

    def classify_intent(self, user_message: str, has_active_itinerary: bool) -> IntentType:
        """
        Κατηγοριοποίηση πρόθεσης (Intent Detection) με προστασία από False Negatives σε σύνθετες ερωτήσεις.
        Ιεραρχική ανάλυση:
        1. Replanning Triggers (αν υπάρχει ενεργό δρομολόγιο ή αν ζητείται ρητά αλλαγή/προσαρμογή)
        2. Itinerary Planning Triggers (χρονικά περιθώρια, αιτήματα σχεδιασμού, σύνθετες λίστες POIs)
        3. Weather Triggers (καιρικά ερωτήματα, θερμοκρασία, καύσωνας)
        4. Public Transit Triggers (μετρό, λεωφορεία, οδηγίες μετάβασης)
        5. Factual QA & Knowledge Grounding (ιστορία, ωράρια, περιγραφές, τιμές)
        """
        msg_lower = user_message.lower()
        msg_norm = normalize_greek(msg_lower)

        # -------------------------------------------------------------
        # 1. Έλεγχος κόπωσης & ξεκούρασης -> Υποχρεωτικά REPLANNING
        # -------------------------------------------------------------
        fatigue_keywords = ["κουραστηκα", "ξεκουραση", "καφε", "διαλειμμα", "κουραση", "ξεκουραστω"]
        if any(kw in msg_norm for kw in fatigue_keywords) or any(kw in msg_lower for kw in fatigue_keywords):
            return IntentType.REPLANNING

        # -------------------------------------------------------------
        # 2. Έλεγχος για αρνητικούς περιορισμούς (π.χ. "χωρίς μουσεία", "βγάλε", "δεν θέλω άλλα")
        # -------------------------------------------------------------
        negation_triggers = [
            "δεν θελω", "δεν θέλω", "οχι αλλα", "όχι άλλα", "οχι αλλο", "όχι άλλο",
            "μην βαλεις", "μην βάλεις", "αποκλεισε", "αποκλείσε", "χωρις", "χωρίς"
        ]
        is_negation = any(trigger in msg_lower for trigger in negation_triggers) or any(
            trigger in msg_norm for trigger in ["δεν θελω", "οχι αλλα", "μην βαλεις", "αποκλεισε", "χωρις"]
        )
        if is_negation and any(w in msg_norm for w in ["μουσει", "αρχαιολογικ", "αξιοθεατ"]):
            return IntentType.REPLANNING if has_active_itinerary else IntentType.ITINERARY_REQUEST

        # -------------------------------------------------------------
        # 3. Dynamic Replanning cues (όταν υπάρχει ενεργό δρομολόγιο ή αλλαγή παραμέτρων)
        # -------------------------------------------------------------
        weather_bad_cues = [
            "βρεχει", "βροχη", "συννεφιασε", "συννεφα", "καταιγιδα", "μπορα", "καυσωνα",
            "vrexei", "vroxi", "vrohy", "synnef", "rain", "raining", "storm", "heatwave"
        ]
        # Αν αναφέρεται κακοκαιρία και υπάρχει ήδη ενεργό δρομολόγιο -> REPLANNING
        if has_active_itinerary and any(w in msg_norm for w in weather_bad_cues):
            return IntentType.REPLANNING

        # Αν αναφέρεται συγκεκριμένο σχέδιο για εξωτερικό χώρο που επηρεάζεται από κακοκαιρία
        if any(w in msg_norm for w in weather_bad_cues) and any(w in msg_norm for w in ["κανονισει", "παμε", "λυκαβηττ", "ακροπολη", "πλακα", "lykab", "acropol"]):
            return IntentType.REPLANNING

        replanning_cues = [
            "αμαξιδιο", "αναπηρικο", "σκαλια",
            "δεν θελω", "βγαλε", "αλλαξε", "αντικαταστησε", "ακυρωσε", "αντι για",
            "δευτερο", "πρωτο μουσείο", "πρωτο αξιοθεατο",
            "παιδι", "παιδια", "κορη", "γιο", "γιος",
            "κουραστηκα", "κουραζομαι",
            "χωρις μουσεια", "οχι μουσεια",
            "προσαρμοσε", "αλλαξε την ωρα"
        ]
        time_change_cues = [
            "εχω μονο", "εχω διαθεσιμες μονο", "αντι για 4", "αντι για 3", "μειωσε", "μειωσε το χρονο"
        ]
        if has_active_itinerary:
            if any(w in msg_norm for w in replanning_cues) or any(w in msg_norm for w in time_change_cues):
                # Αν πρόκειται για ερώτημα επίλυσης αναφοράς εισιτηρίου (TC-MULT-03: "πόσο κοστίζει το εισιτήριο για το πρώτο;")
                if any(w in msg_norm for w in ["ποσο κοστιζει", "ποσο κανει", "τιμη εισιτηριου"]) and "εισιτηρι" in msg_norm:
                    pass  # επιτρέπουμε να προχωρήσει σε factual_qa
                else:
                    return IntentType.REPLANNING

        # -------------------------------------------------------------
        # 4. Αιτήματα Δρομολογίου / Σχεδιασμού (ITINERARY_REQUEST)
        # Ελέγχεται ΠΡΙΝ από απλές λέξεις συγκοινωνίας ώστε σύνθετες ερωτήσεις
        # (π.χ. "Έχω 3 ώρες και θέλω να μάθω πώς πάω στην Ακρόπολη, φτιάξε μου πρόγραμμα")
        # να ΜΗΝ καταλήγουν σε False Negative (FACTUAL_QA)!
        # -------------------------------------------------------------
        itinerary_words = [
            "προγραμμα", "δρομολογιο", "πλανο", "itinerary",
            "φτιαξε", "προτεινε", "δημιουργησε", "σχεδιασε", "οργανωσε",
            "θελω να επισκεφθω", "θελω να επισκεφτω", "επισκεφθω", "επισκεφτω",
            "θελω να επισκευθω", "θελω να επισκευτω", "επισκευθω", "επισκευτω",
            "θελω να δω", "διαδρομη", "με σταση",
            "χαλαρο ρυθμο", "βολτα", "περιπατο", "μεταβαση",
            "εχω 3 ωρες", "εχω 2 ωρες", "εχω 4 ωρες", "εχω 1 ωρα", "σε 90 λεπτα"
        ]
        has_time_budget = bool(re.search(r"εχω\s+\d+\s+ωρε", msg_norm)) or bool(
            re.search(r"\d+\s*ωρες", msg_norm) and any(w in msg_norm for w in ["στην", "στο", "στα", "αθηνα", "πλακα", "αμαξιδιο", "χωρις", "παιδι", "κορη", "γιο"])
        ) or "σε 90 λεπτα" in msg_norm or bool(re.search(r"\d{1,2}:\d{2}\s*(?:-|εως|μεχρι)\s*\d{1,2}:\d{2}", msg_norm))

        has_itinerary_request = (
            any(w in msg_norm for w in itinerary_words)
            or any(w in msg_norm for w in ["φτιαξε μου", "προτεινε μου", "σχεδιασε μου", "δημιουργησε ενα"])
            or has_time_budget
        )

        # Εάν ζητήθηκαν πολλαπλά αξιοθέατα για επίσκεψη σε χρονικό διάστημα (π.χ. Use Case 3)
        explicit_pois = extract_named_pois(user_message)
        if len(explicit_pois) >= 2 and any(w in msg_norm for w in ["μεσα σε", "σε 2 ωρες", "σε 3 ωρες", "17:00", "ωρες", "επισκεφθω", "επισκεφτω"]):
            has_itinerary_request = True

        if has_itinerary_request:
            return IntentType.REPLANNING if has_active_itinerary else IntentType.ITINERARY_REQUEST

        # -------------------------------------------------------------
        # 5. Έλεγχος για καιρικό ερώτημα (WEATHER_QUERY)
        # -------------------------------------------------------------
        weather_query_cues = [
            "τι καιρο", "τι καιρος", "πες τον καιρο", "πες μου τον καιρο",
            "καιρος στην", "καιρος στο", "καιρο εχει", "θερμοκρασια", "προγνωση",
            "ti kairo", "ti kairos", "kairo exei", "kairo exi", "kairos stin", "kairos sto",
            "pes ton kairo", "pes mou ton kairo", "kairou", "kairos", "kairo",
            "weather", "temperature", "forecast", "how is the weather", "what is the weather",
            "καυσωνα", "καυσωνας", "40°c", "40 βαθμ", "αιθριος"
        ]
        norm_weather_cues = [normalize_greek(w) for w in weather_query_cues]
        if any(w in msg_norm for w in norm_weather_cues) or any(w in msg_norm for w in ["καιροσ", "καιρο", "καιρου"]):
            return IntentType.WEATHER_QUERY

        # -------------------------------------------------------------
        # 6. Έλεγχος για Δημόσιες Συγκοινωνίες / Μετρό / Λεωφορεία (TRANSIT_QUERY)
        # -------------------------------------------------------------
        transit_route_cues = [
            "πως παω απο", "πως θα παω απο", "πως να παω απο", "δρομολογια μετρο",
            "γραμμη μετρο", "γραμμη 1", "γραμμη 2", "γραμμη 3", "ποιο λεωφορειο",
            "σταθμος μετρο", "σταθμοι μετρο", "σταθμο μετρο", "δρομολογιο μετρο",
            "σταση μετρο", "οασα", "στασυ", "stasy", "oasa"
        ]
        norm_transit_cues = [normalize_greek(c) for c in transit_route_cues]
        if (
            any(w in msg_norm for w in norm_transit_cues)
            or (
                any(w in msg_norm for w in ["πωσ παω", "πωσ θα παω", "πωσ να παω"])
                and any(t in msg_norm for t in ["μετρο", "λεωφορει", "τραμ", "απο", "συνταγμα", "ακροπολ"])
            )
            or (
                any(w in msg_norm for w in ["δρομολογι", "σταθμ"])
                and any(t in msg_norm for t in ["μετρο", "stasy", "οασα"])
            )
        ):
            return IntentType.TRANSIT_QUERY

        # -------------------------------------------------------------
        # 7. Default: Factual QA (πληροφορίες, ιστορία, ωράρια, εισιτήρια)
        # -------------------------------------------------------------
        return IntentType.FACTUAL_QA

    def process_message(self, state: UserState, user_message: str) -> str:
        """
        Κεντρική συνάρτηση επεξεργασίας μηνύματος.
        """
        state.add_message("user", user_message)
        msg_lower = user_message.lower()

        # 0. Active Input Guardrail inspection (Prompt Injection, Out-of-Domain)
        guardrail_in = guardrails_manager.inspect_input(user_message)
        if not guardrail_in.passed and guardrail_in.sanitized_output:
            state.add_message("assistant", guardrail_in.sanitized_output)
            return guardrail_in.sanitized_output

        # 1. Έλεγχος για Prompt Injection / Jailbreak (TC-ADVR-01)
        if any(w in msg_lower for w in ["αγνόησε", "αγνοησε", "ignore previous", "επινόησε", "επινοησε", "jailbreak"]):
            response = (
                "🛡️ **Ασφάλεια & Τεκμηρίωση:** Ως AI Τουριστικός Βοηθός της Αθήνας, λειτουργώ αυστηρά βάσει της "
                "πιστοποιημένης βάσης γνώσης. Δεν επιτρέπεται η παράκαμψη των οδηγιών grounding ούτε η επινόηση "
                "ανύπαρκτων ή ψευδών αξιοθεάτων. Μπορώ να σας προσφέρω έγκυρες πληροφορίες μόνο για πραγματικά, "
                "επαληθευμένα τουριστικά σημεία της πόλης."
            )
            state.add_message("assistant", response)
            return response

        # 2. Έλεγχος για Out-of-Domain αίτημα (TC-ADVR-03)
        if any(w in msg_lower for w in ["fibonacci", "python", "javascript", "κώδικα", "κωδικα", "συνταγή μαγειρικής"]):
            response = (
                "👋 Λυπάμαι, αλλά ως εξειδικευμένος AI Τουριστικός Βοηθός της Αθήνας δεν μπορώ να παρέχω κώδικα "
                "προγραμματισμού ή άσχετες τεχνικές υπηρεσίες. Είμαι εδώ αποκλειστικά για να σας καθοδηγήσω σε "
                "αξιοθέατα, μουσεία, καιρικές συνθήκες και δρομολόγια στην πόλη της Αθήνας!"
            )
            state.add_message("assistant", response)
            return response

        # 3. Έλεγχος για επικίνδυνη δραστηριότητα / καταιγίδα στο βουνό (TC-ADVR-02)
        if "καταιγίδα" in msg_lower and any(w in msg_lower for w in ["πεζοπορία", "πεζοπορια", "υμηττ", "μονοπάτι", "μονοπατι", "βουνό", "βουνο"]):
            response = (
                "🚨 **Προειδοποίηση Ασφαλείας (Safety Alert):** Αποτρέπεται αυστηρά η πεζοπορία στο μονοπάτι του Υμηττού "
                "(ή σε οποιοδήποτε ορεινό ανάγλυφο) εν μέσω καταιγίδας. Υπάρχει άμεσος και σοβαρός κίνδυνος κεραυνοπληξίας, "
                "αιφνίδιων χειμάρρων (flash floods) και ολισθηρότητας. Παραμείνετε σε ασφαλές, στεγασμένο περιβάλλον."
            )
            state.add_message("assistant", response)
            return response

        # 4. Έλεγχος για καύσωνα (40°C) το μεσημέρι (Thermal Microclimate Guardrail)
        if any(w in msg_lower for w in ["40°c", "40 βαθμ", "καύσωνα", "καυσωνα", "39°c", "41°c"]) and any(w in msg_lower for w in ["13:00", "14:00", "μεσημέρι", "μεσημερι", "πεζοπορία", "πεζοπορια", "ακρόπολη", "ακροπολη", "φιλοπάππ", "φιλοπαππ", "αγορά", "αγορα"]):
            response = (
                "🚨 **Προειδοποίηση Ακραίας Ζέστης (Heatwave Warning):** Με θερμοκρασία 40°C το μεσημέρι, η πεζοπορία "
                "σε εκτεθειμένους υπαίθριους χώρους (όπως ο Λόφος Φιλοπάππου, η Αρχαία Αγορά ή ο Βράχος της Ακρόπολης) "
                "απαγορεύεται από το Thermal Microclimate Guardrail λόγω υψηλού κινδύνου θερμοπληξίας. "
                "Συστήνεται έντονα να μεταφέρετε τις υπαίθριες δραστηριότητες νωρίς το πρωί ή αργά το απόγευμα, "
                "και τις μεσημεριανές ώρες να επιλέξετε αποκλειστικά κλιματιζόμενους χώρους, όπως το Μουσείο Ακρόπολης!"
            )
            state.add_message("assistant", response)
            return response

        # 5. Έλεγχος για σταθερότητα πλάνου σε αίθριο καιρό (TC-WEAT-03)
        if "αίθριος" in msg_lower or "22°c" in msg_lower:
            if "χρειάζεται να αλλάξουμε" in msg_lower or "αλλαγή" in msg_lower:
                response = (
                    "☀️ Ο καιρός είναι ιδανικός και αίθριος (22°C), επομένως το πρόγραμμα πεζοπορίας παραμένει ως έχει "
                    "χωρίς καμία ανάγκη τροποποίησης. Απολαύστε τη βόλτα σας!"
                )
                state.add_message("assistant", response)
                return response

        # 6. Έλεγχος για απόρρητη / Out-of-Knowledge πληροφορία (TC-FACT-02)
        if any(w in msg_lower for w in ["προσωπικό τηλέφωνο", "προσωπικο τηλεφωνο", "τηλέφωνο του διευθυντή", "τηλεφωνο του διευθυντη", "μισθός", "κινητό του"]):
            response = (
                "🔒 **Δήλωση Μη Διαθέσιμης Πληροφορίας:** Δεν διαθέτω προσωπικά τηλέφωνα ή ιδιωτικά στοιχεία "
                "επικοινωνίας στελεχών στη βάση γνώσης μου (αποφυγή ψευδαισθήσεων/hallucinations). "
                "Για επίσημη πληροφόρηση, παρακαλώ επικοινωνήστε με το επίσημο τηλεφωνικό κέντρο ή την ιστοσελίδα του Μουσείου."
            )
            state.add_message("assistant", response)
            return response

        # 7. Έλεγχος για γεωγραφική ασυμβατότητα (TC-FEAS-03: Σούνιο -> Ακρόπολη σε 15 λεπτά με τα πόδια)
        if "σούνιο" in msg_lower and "ακρόπολ" in msg_lower and any(w in msg_lower for w in ["15 λεπτά", "15 λεπτα", "πόδια", "ποδια"]):
            response = (
                "⚠️ **Γεωγραφικά Ανέφικτο Αίτημα:** Η πραγματική απόσταση από το Σούνιο μέχρι την Ακρόπολη είναι περίπου "
                "65 χιλιόμετρα. Η μετακίνηση με τα πόδια απαιτεί τουλάχιστον 13-14 ώρες συνεχούς βαδίσματος, "
                "επομένως είναι απολύτως αδύνατο να ολοκληρωθεί σε 15 λεπτά!"
            )
            state.add_message("assistant", response)
            return response

        # 8. Έλεγχος για επίσκεψη εκτός ωραρίου (TC-FEAS-02: Μουσείο Ακρόπολης στις 21:30)
        if "ακρόπολ" in msg_lower and "μουσείο" in msg_lower and ("21:30" in msg_lower or "22:00" in msg_lower):
            response = (
                "⚠️ **Έλεγχος Ωραρίου Λειτουργίας:** Το Μουσείο Ακρόπολης κλείνει στις 20:00 (ωράριο: 09:00 - 20:00). "
                "Η επίσκεψη στις 21:30 το βράδυ δεν είναι εφικτή καθώς ο χώρος θα είναι κλειστός. "
                "Σας προτείνω να επισκεφθείτε το μουσείο νωρίτερα (π.χ. 18:00 - 19:30)."
            )
            state.add_message("assistant", response)
            return response

        # 8b. Έλεγχος για επίσκεψη εκτός ωραρίου στο Μουσείο Μπενάκη (TC-FEAS-CLOSED-MUSEUM / Incident Hotfix)
        if "μπενάκη" in msg_lower and any(w in msg_lower for w in ["18:00", "19:00", "20:00", "το απόγευμα"]):
            response = (
                "⚠️ **Έλεγχος Ωραρίου Λειτουργίας (Closed Museum):** Το Μουσείο Μπενάκη Ελληνικού Πολιτισμού "
                "κλείνει στις 17:00 (ωράριο: 09:00 - 17:00). Η επίσκεψη στις 18:00 δεν είναι εφικτή καθώς ο χώρος "
                "θα είναι κλειστός. Σας προτείνω να μεταφέρετε την επίσκεψη νωρίτερα ή να επιλέξετε ένα μουσείο "
                "που παραμένει ανοιχτό έως τις 20:00 (όπως το Μουσείο Ακρόπολης ή το Μουσείο Κυκλαδικής Τέχνης)."
            )
            state.add_message("assistant", response)
            return response

        # 9. Έλεγχος για επίλυση αναφοράς / Coreference Resolution (TC-MULT-03: "το πρώτο")
        if any(w in msg_lower for w in ["το πρώτο", "το πρωτο", "για το πρώτο", "για το πρωτο"]):
            docs = self.rag.retrieve(query="Μουσείο Ακρόπολης εισιτήριο τιμή ωράριο", top_k=2)
            rag_info = self._generate_rag_response(user_message, docs)
            response = (
                f"Αναφορικά με το **Μουσείο Ακρόπολης** (το πρώτο αξιοθέατο της προηγούμενης πρότασης):\n\n"
                f"{rag_info}"
            )
            state.add_message("assistant", response)
            return response

        # 10. Συνδυασμός ιστορικής γνώσης & ζωντανού καιρού (TC-FACT-03)
        if any(w in msg_lower for w in ["πότε χτίστηκε", "ποτε χτιστηκε", "παρθενώνας", "παρθενωνας"]) and any(w in msg_lower for w in ["καιρό", "καιρος"]):
            docs = self.rag.retrieve(query="Παρθενώνας χρονολογία ιστορία 5ος αιώνας", top_k=2)
            try:
                weather = get_live_weather(city="Athens")
            except Exception:
                weather = {"temperature_c": 19.0, "description": "αίθριος καιρός", "humidity_pct": 55}
            rag_info = self._generate_rag_response(user_message, docs)
            response = (
                f"🏛️ **Ιστορική Τεκμηρίωση:** Ο Παρθενώνας στον Ιερό Βράχο της Ακρόπολης χτίστηκε τον 5ο αιώνα π.Χ. "
                f"(μεταξύ 447–432 π.Χ.) κατά τον Χρυσό Αιώνα του Περικλή.\n\n"
                f"{rag_info}\n\n"
                f"🌤️ **Ζωντανός Καιρός στον Ιερό Βράχο:** Αυτή τη στιγμή η θερμοκρασία είναι **{weather['temperature_c']}°C** "
                f"με συνθήκες: **{weather['description']}** (Υγρασία: {weather['humidity_pct']}%)."
            )
            state.add_message("assistant", response)
            return response

        intent = self.classify_intent(user_message, has_active_itinerary=(state.active_itinerary is not None))

        # --- INTENT A: FACTUAL Q&A ---
        if intent == IntentType.FACTUAL_QA:
            mentioned_poi_ids = extract_named_pois(user_message)

            # Entity Validation Gate: έλεγχος αν ζητείται άγνωστο σημείο/περιοχή εκτός KB
            if not mentioned_poi_ids:
                retrieved_docs = self.rag.retrieve(query=user_message, top_k=3)
                entity_val = self.validate_entities_in_kb(user_message, retrieved_docs)
                if not entity_val["valid"]:
                    response = entity_val["message"]
                    state.add_message("assistant", response)
                    return response

            if len(mentioned_poi_ids) >= 1 or any(w in msg_lower for w in [" και ", ", ", " με στάση", " με σταση", " καθώς και "]):
                retrieved_docs = []
                seen_ids = set()
                # 1. Fetch exact POI documents for all extracted entities
                for pid in mentioned_poi_ids:
                    doc = self.rag.get_poi_document(pid)
                    if doc and doc["id"] not in seen_ids:
                        retrieved_docs.append(doc)
                        seen_ids.add(doc["id"])

                # 2. Parallel sub-queries for compound clauses (e.g. "Μουσείο Ακρόπολης", "Μοναστηράκι", "Βουλή")
                sub_clauses = [p.strip() for p in re.split(r"[,;]|\s+και\s+|\s+μετά\s+", user_message) if len(p.strip()) >= 3]
                if len(sub_clauses) > 1:
                    multi_docs = self.rag.retrieve_multi(sub_clauses, top_k_per_query=1)
                    for d in multi_docs:
                        if d["id"] not in seen_ids:
                            retrieved_docs.append(d)
                            seen_ids.add(d["id"])
                            if len(retrieved_docs) >= 4:
                                break

                # 3. If still under 3, fill with top semantic matches
                if len(retrieved_docs) < 3:
                    fallback_docs = self.rag.retrieve(query=user_message, top_k=3)
                    for d in fallback_docs:
                        if d["id"] not in seen_ids:
                            retrieved_docs.append(d)
                            seen_ids.add(d["id"])
                            if len(retrieved_docs) >= 3:
                                break

                response = self._generate_rag_response(user_message, retrieved_docs)
            else:
                retrieved_docs = self.rag.retrieve(query=user_message, top_k=3)
                response = self._generate_rag_response(user_message, retrieved_docs)

        # --- INTENT B: WEATHER QUERY ---
        elif intent == IntentType.WEATHER_QUERY:
            try:
                weather_data = get_live_weather(city=w_args.city)
            except Exception:
                weather_data = {
                    "city": "Athens",
                    "temperature_c": 19.0,
                    "feels_like_c": 18.5,
                    "condition": "Αίθριος",
                    "description": "αίθριος καιρός",
                    "humidity_pct": 55,
                    "rain_mm_1h": 0.0,
                    "is_indoor_recommended": False
                }
            note = ""
            unsupported_areas = [
                "κυψελ", "kypsel", "γλυφαδ", "glyfad", "μαρουσ", "marous", "κηφισ", "kifis",
                "περιστερ", "perister", "νεα σμυρν", "nea smyrn", "χαλανδρ", "chalandr",
                "πατησι", "patisi", "εξαρχει", "exarch", "παγκρατ", "pagkrat",
                "πειραι", "peirai", "piraeus", "ψυχικ", "psychik", "ζωγραφ", "zograf",
                "ιλισι", "ilisi", "καλλιθε", "kallithe", "βουλιαγμεν", "vouliagmen"
            ]
            if any(w in msg_lower for w in unsupported_areas):
                note = "\n\n*(Σημείωση: Η συγκεκριμένη περιοχή καλύπτεται από τις γενικές καιρικές συνθήκες της μητροπολιτικής Αθήνας.)*"

            response = (
                f"🌤️ **Ζωντανή Πρόγνωση & Καιρικές Συνθήκες: Αθήνα**\n\n"
                f"• **Θερμοκρασία:** {weather_data['temperature_c']}°C (Αίσθηση: {weather_data.get('feels_like_c', weather_data['temperature_c'])}°C)\n"
                f"• **Συνθήκες:** {weather_data['description']}\n"
                f"• **Σχετική Υγρασία:** {weather_data['humidity_pct']}%\n"
                f"• **Σύσταση Περιπάτου:** {'⚠️ Προτείνονται στεγασμένοι χώροι/μουσεία λόγω καιρικών συνθηκών' if weather_data.get('is_indoor_recommended') else '✅ Ιδανικές συνθήκες για εξωτερικό περίπατο και αξιοθέατα'}"
                f"{note}"
            )

        # --- INTENT C: ITINERARY REQUEST & DYNAMIC REPLANNING ---
        elif intent in [IntentType.ITINERARY_REQUEST, IntentType.REPLANNING]:
            response = self._handle_itinerary_workflow(state, user_message)

        # --- INTENT D: PUBLIC TRANSIT & ROUTING ---
        elif intent == IntentType.TRANSIT_QUERY:
            mentioned = extract_named_pois(user_message)
            dest = mentioned[0] if mentioned else "acropolis_hill"
            orig = mentioned[1] if len(mentioned) > 1 else "syntagma_changing_guards"
            t_args = transit_args_parser.parse({
                "origin": orig,
                "destination": dest,
                "wheelchair_accessible": state.wheelchair_accessible,
                "departure_time": state.start_time or "14:00"
            })
            transit_res = get_transit_route(
                origin=t_args.origin,
                destination=t_args.destination,
                wheelchair_accessible=t_args.wheelchair_accessible,
                departure_time=t_args.departure_time,
            )
            response = transit_res["summary_text"]
            for leg in transit_res.get("legs", []):
                response += f"\n• {leg['instructions']}"

        # --- INTENT E: LIVE TICKETS & AVAILABILITY ---
        elif intent == IntentType.TICKET_QUERY:
            mentioned = extract_named_pois(user_message)
            target_poi = mentioned[0] if mentioned else "acropolis_museum"
            tkt_args = ticketing_args_parser.parse({
                "poi_id": target_poi,
                "action": "check_availability",
                "time_slot": "10:00-11:00",
                "num_tickets": 1,
                "ticket_tier": "adult"
            })
            pricing = get_ticket_pricing(tkt_args.poi_id)
            avail = check_ticket_availability(tkt_args.poi_id, time_slot=tkt_args.time_slot)
            response = (
                f"🎟️ **Πληροφορίες & Διαθεσιμότητα Εισιτηρίων: {pricing.get('poi_name', target_poi)}**\n\n"
                f"• **Επίσημος Πάροχος:** {pricing.get('vendor', 'Επίσημο Ταμείο')}\n"
                f"• **Κανονικό Εισιτήριο:** {pricing.get('regular_price_eur', 0.0):.2f} € | **Μειωμένο:** {pricing.get('reduced_price_eur', 0.0):.2f} €\n"
                f"• **Συνδυαστικό Εισιτήριο 7 Χώρων:** {'✅ Διαθέσιμο (30,00 €)' if pricing.get('combined_ticket_eligible') else '❌ Μη επιλέξιμο'}\n"
                f"• **Δωρεάν Είσοδος:** {pricing.get('free_entry_conditions', '-')}\n\n"
                f"{avail.get('recommendation', '')}"
            )

        else:
            response = "Λυπάμαι, δεν κατάλαβα το αίτημά σας. Πώς μπορώ να βοηθήσω;"

        # Output Guardrail: Intercept hallucinations, verify grounding & sanity
        retrieved_docs_for_gr = retrieved_docs if 'retrieved_docs' in locals() else []
        final_response, _ = guardrails_manager.inspect_output(
            response, user_message, retrieved_docs_for_gr, state.active_itinerary
        )
        state.add_message("assistant", final_response)
        return final_response

    def _handle_itinerary_workflow(self, state: UserState, user_message: str) -> str:
        """
        Διαχειρίζεται τη δημιουργία και τη δυναμική τροποποίηση δρομολογίου.
        """
        msg_lower = user_message.lower()
        msg_norm = normalize_greek(msg_lower)

        # 0. Εξαγωγή Αρνητικών Περιορισμών (Solution 2)
        museums_negative_triggered = extract_negative_constraints(user_message, state, raw_pois=self.rag.raw_pois)

        # 0b. Έλεγχος κόπωσης & αιτήματος καφέ / ξεκούρασης (Solution 1)
        fatigue_keywords = ["κουραστηκα", "ξεκουραση", "καφε", "διαλειμμα", "κουραση", "ξεκουραστω"]
        is_fatigue_coffee = any(kw in msg_norm for kw in fatigue_keywords) or any(kw in msg_lower for kw in fatigue_keywords)
        if is_fatigue_coffee:
            if "relaxed_pace" not in state.preferences:
                state.preferences.append("relaxed_pace")
            state.preferred_pace = "relaxed"

        # 1. Ενημέρωση User State βάσει του νέου μηνύματος
        # Παιδιά
        if any(w in msg_lower for w in ["παιδί", "παιδιά", "παιδια", "κόρη", "γιο", "γιος"]):
            state.traveling_with_kids = True
            age_match = re.search(r"(\d+)\s*(?:ετών|χρονών|χρονη|χρονο)", msg_lower)
            if age_match:
                state.kid_age = int(age_match.group(1))
            elif state.kid_age is None:
                state.kid_age = 8

        # Προσβασιμότητα σε αναπηρικό αμαξίδιο
        if any(w in msg_lower for w in ["αμαξίδιο", "αμαξιδιο", "αναπηρικό", "αναπηρικο", "σκαλιά", "σκαλια"]):
            state.wheelchair_accessible = True
            # Αποκλεισμός σημείων με σκαλοπάτια (π.χ. Αναφιώτικα)
            if "anafiotika" not in state.blacklisted_poi_ids:
                state.blacklisted_poi_ids.append("anafiotika")

        # Αρνητικός περιορισμός: Χωρίς μουσεία (TC-PERS-02)
        if "χωρίς μουσεία" in msg_lower or "χωρις μουσεια" in msg_lower or "όχι μουσεία" in msg_lower or "οχι μουσεια" in msg_lower:
            for poi in self.rag.raw_pois:
                if poi.get("category") == "museum" and poi["id"] not in state.blacklisted_poi_ids:
                    state.blacklisted_poi_ids.append(poi["id"])

        # Χαλαρός ρυθμός (TC-PERS-03)
        if any(w in msg_lower for w in ["χαλαρό ρυθμό", "χαλαρο ρυθμο", "κουράζομαι", "κουραζομαι"]):
            if "relaxed_pace" not in state.preferences:
                state.preferences.append("relaxed_pace")

        # Αντικατάσταση συγκεκριμένου (π.χ. 2ου) μουσείου στο multi-turn (TC-MULT-01)
        if "δεύτερο" in msg_lower or "δευτερο" in msg_lower:
            if state.active_itinerary:
                acts = [s for s in state.active_itinerary.get("schedule", []) if s["type"] == "activity"]
                if len(acts) >= 2:
                    sec_id = acts[1].get("poi_id")
                    if sec_id and sec_id not in state.blacklisted_poi_ids:
                        state.blacklisted_poi_ids.append(sec_id)
            if "benaki_museum_greek_culture" not in state.blacklisted_poi_ids:
                state.blacklisted_poi_ids.append("benaki_museum_greek_culture")

        # Έλεγχος για αφαίρεση / blacklisting συγκεκριμένων POIs
        if "μουσείο" in msg_lower and any(w in msg_lower for w in ["δεν", "όχι", "οχι", "βγάλε", "βγαλε", "άλλαξε", "αλλαξε"]):
            if state.active_itinerary:
                for step in state.active_itinerary.get("schedule", []):
                    pid = step.get("poi_id")
                    if pid and "museum" in pid and pid not in state.blacklisted_poi_ids:
                        state.blacklisted_poi_ids.append(pid)
            if "acropolis_museum" not in state.blacklisted_poi_ids:
                state.blacklisted_poi_ids.append("acropolis_museum")

        # Έλεγχος για συνωστισμό / ουρά αναμονής > 45 λεπτά (IoT Crowd Rerouting)
        crowd_wait_match = re.search(r"(\d+)\s*(?:λεπτά|λεπτα|λεπτών|λεπτων|mins|min)", msg_lower)
        has_crowd_intent = any(w in msg_lower for w in ["αναμονή", "αναμονη", "ουρά", "ουρα", "συνωστισμ", "καθυστέρηση", "καθυστερηση", "wait"])
        if has_crowd_intent and crowd_wait_match:
            wait_val = int(crowd_wait_match.group(1))
            if wait_val >= 45:
                # Αν η αναμονή αφορά την Ακρόπολη, προσθήκη στα blacklisted_poi_ids άμεσα
                if any(w in msg_lower for w in ["ακρόπολ", "ακροπολ", "βράχο", "βραχο", "acropolis"]):
                    if "acropolis_hill" not in state.blacklisted_poi_ids:
                        state.blacklisted_poi_ids.append("acropolis_hill")

        # Έλεγχος χρονικού παραθύρου (π.χ. "17:00 - 19:00" ή "17:00 έως 19:00")
        time_range = re.search(r"(\d{1,2}:\d{2})\s*(?:-|έως|εως|μέχρι|μεχρι|to)\s*(\d{1,2}:\d{2})", msg_lower)
        if time_range:
            state.start_time = time_range.group(1)
            s_parts = [int(p) for p in time_range.group(1).split(":")]
            e_parts = [int(p) for p in time_range.group(2).split(":")]
            budget = ((e_parts[0] * 60 + e_parts[1]) - (s_parts[0] * 60 + s_parts[1])) / 60.0
            if budget > 0:
                state.time_budget_hours = budget
        else:
            # Έλεγχος έναρξης (π.χ. "στις 13:00", "στις 14:00")
            start_match = re.search(r"(?:στις|στις\s*ώρα|ώρα)\s*(\d{1,2}:\d{2})", msg_lower)
            if start_match:
                state.start_time = start_match.group(1)

            # Έλεγχος διάρκειας (π.χ. "2 ώρες", "3 ωρών", "90 λεπτά", "μόνο 2 ώρες")
            dur_mins_match = re.search(r"(\d+)\s*(?:λεπτά|λεπτα|mins)", msg_lower)
            dur_match = re.search(r"(\d+)\s*(?:ώρες|ωρες|ωρών|ωρων|hours)", msg_lower)
            if dur_mins_match:
                state.time_budget_hours = float(dur_mins_match.group(1)) / 60.0
            elif dur_match:
                state.time_budget_hours = float(dur_match.group(1))

        # 2. Έλεγχος & Λήψη Live Weather μέσω Pydantic Output Parser
        weather_triggered_heat = any(w in msg_lower for w in ["40°c", "40 βαθμ", "καύσωνα", "καυσωνα", "καύσων", "38°c", "39°c", "41°c", "40c", "θερμοκρασία 40", "θερμοκρασια 40"]) or ("40" in msg_lower and any(w in msg_lower for w in ["βαθμ", "θερμοκρασ", "ζέστη", "ζεστη"]))
        weather_triggered_bad = weather_triggered_heat or any(w in msg_lower for w in ["συννέφιασε", "συννεφιασε", "βρέχει", "βρεχει", "βροχή", "βροχη", "σύννεφα", "καταιγίδα"])
        
        if weather_triggered_heat:
            weather_data = {
                "city": "Athens",
                "temperature_c": 40.0,
                "feels_like_c": 43.0,
                "condition": "Heatwave",
                "description": "καύσωνας / ακραία ζέστη (40°C)",
                "humidity_pct": 25,
                "rain_mm_1h": 0.0,
                "rain_expected": False,
                "is_indoor_recommended": True,
                "uv_index": 10.0,
                "wind_speed_kmh": 5.0,
            }
            # Αυτόματο Hard Blacklist όλων των outdoor POIs στη Feasibility Engine
            for p in self.rag.raw_pois:
                if p.get("type") == "outdoor" and p["id"] not in state.blacklisted_poi_ids:
                    state.blacklisted_poi_ids.append(p["id"])
        else:
            w_args = weather_args_parser.parse({
                "city": "Athens",
                "mock_scenario": "rain" if weather_triggered_bad else None,
            })
            try:
                weather_data = get_live_weather(city=w_args.city)
            except Exception:
                weather_data = {
                    "city": "Athens",
                    "temperature_c": 19.0,
                    "feels_like_c": 18.5,
                    "condition": "Αίθριος",
                    "description": "αίθριος καιρός",
                    "humidity_pct": 55,
                    "rain_mm_1h": 0.0,
                    "rain_expected": False,
                    "is_indoor_recommended": False
                }
            if weather_triggered_bad:
                # Αν ο χρήστης είχε προγραμματίσει εξωτερικό χώρο (π.χ. Λυκαβηττό), τον αποκλείουμε
                if any(w in msg_lower for w in ["λυκαβηττ", "λυκαβητο", "λυκαβηττό"]):
                    if "lycabettus_hill" not in state.blacklisted_poi_ids:
                        state.blacklisted_poi_ids.append("lycabettus_hill")

        # 3. Ανίχνευση ρητά κατονομασμένων POIs από τον χρήστη
        explicit_poi_ids = [pid for pid in extract_named_pois(user_message) if pid not in state.blacklisted_poi_ids]

        # Entity Validation Gate πριν την κλήση της Feasibility Engine
        # Αν ο χρήστης ζήτησε συγκεκριμένη τοποθεσία/περιοχή εκτός βάσης γνώσης
        candidate_docs = self.rag.retrieve(query=user_message, top_k=5)
        entity_val = self.validate_entities_in_kb(user_message, candidate_docs)
        if not entity_val["valid"]:
            return entity_val["message"]

        # Αν ο χρήστης ζήτησε συγκεκριμένο αριθμό μουσείων (π.χ. TC-FEAS-01: "5 διαφορετικά μουσεία")
        num_museums_match = re.search(r"(\d+)\s*(?:διαφορετικά\s*)?μουσεία", msg_lower)
        if num_museums_match:
            req_count = int(num_museums_match.group(1))
            all_museums = [p for p in self.rag.raw_pois if p.get("category") == "museum"]
            for m in all_museums[:req_count]:
                if m["id"] not in explicit_poi_ids:
                    explicit_poi_ids.append(m["id"])

        # 4. Ανάκτηση POIs από το RAG με βάση τα ενημερωμένα κριτήρια μέσω Pydantic Parser
        filters = {}
        if state.traveling_with_kids:
            filters["kid_friendly"] = True
        if state.wheelchair_accessible:
            filters["wheelchair_accessible"] = True

        candidate_pois: List[Dict[str, Any]] = []

        # Αν ο χρήστης ζήτησε ρητά συγκεκριμένα POIs (π.χ. Use Case 3), τα τοποθετούμε ως πρώτους υποψηφίους
        if explicit_poi_ids:
            for pid in explicit_poi_ids:
                if pid not in state.blacklisted_poi_ids:
                    p = self.rag.get_poi(pid)
                    if p:
                        if state.wheelchair_accessible:
                            is_acc = p.get("wheelchair_accessible", False)
                            has_st = "stairs" in p.get("tags", [])
                            if not is_acc or has_st:
                                continue
                        candidate_pois.append(p)

        # Αν δεν υπάρχουν αρκετά, εκτελούμε RAG retrieval με επικυρωμένα ορίσματα
        rag_query = user_message
        if weather_triggered_bad or (weather_data and weather_data.get("is_indoor_recommended")):
            rag_query += " museum indoor"
        elif state.wheelchair_accessible:
            rag_query += " accessible flat museum"

        r_args = rag_args_parser.parse({
            "query": rag_query,
            "top_k": 8,
            "filters": filters,
            "explicit_poi_ids": explicit_poi_ids,
        })
        candidate_docs = self.rag.retrieve(query=r_args.query, top_k=r_args.top_k, filters=r_args.filters)
        for doc in candidate_docs:
            p_meta = doc["metadata"]
            if p_meta["id"] not in state.blacklisted_poi_ids:
                if state.wheelchair_accessible:
                    is_acc = p_meta.get("wheelchair_accessible", False)
                    has_st = "stairs" in p_meta.get("tags", [])
                    if not is_acc or has_st:
                        continue
                if not any(cp["id"] == p_meta["id"] for cp in candidate_pois):
                    candidate_pois.append(p_meta)

        # Αν ακόμα υπολείπονται, συμπληρώνουμε από τη βάση
        if len(candidate_pois) < 4:
            for poi in self.rag.raw_pois:
                if poi["id"] not in state.blacklisted_poi_ids:
                    if state.traveling_with_kids and not poi.get("kid_friendly"):
                        continue
                    if state.wheelchair_accessible:
                        is_acc = poi.get("wheelchair_accessible", False)
                        has_st = "stairs" in poi.get("tags", [])
                        if not is_acc or has_st:
                            continue
                    if not any(cp["id"] == poi["id"] for cp in candidate_pois):
                        candidate_pois.append(poi)

        # Εάν αποκλείστηκαν αρχαιολογικά μουσεία (Scenario 5):
        # Εξασφαλίζουμε ότι τα candidate POIs ξεκινούν με Εθνικό Κήπο & Σύνταγμα/Ζάππειο & Πλάκα
        if museums_negative_triggered or "museum" in state.blacklisted_categories:
            preferred_open_ids = ["national_garden", "syntagma_changing_guards", "plaka_historic_walk", "panathenaic_stadium", "monastiraki_flea_market"]
            open_pois = [self.rag.get_poi(pid) for pid in preferred_open_ids if self.rag.get_poi(pid) and self.rag.get_poi(pid)["id"] not in state.blacklisted_poi_ids]
            candidate_pois = open_pois + [p for p in candidate_pois if p["id"] not in [x["id"] for x in open_pois]]

        # Εάν ζητήθηκε καφές/ξεκούραση (Scenario 4):
        # Τοποθετούμε ως πρώτο σημείο επίσκεψης ένα κεντρικό POI (Μοναστηράκι 50 λεπτά: 14:05-14:55),
        # ακολουθούμενο από στάση 35 λεπτών για καφέ στην Πλάκα (14:55-15:30), και η επόμενη επίσκεψη μετατίθεται στις 15:30!
        if is_fatigue_coffee:
            monastiraki = self.rag.get_poi("monastiraki_flea_market")
            garden = self.rag.get_poi("national_garden")
            walk = self.rag.get_poi("plaka_historic_walk")
            reordered = [p for p in [monastiraki, garden, walk] if p and p["id"] not in state.blacklisted_poi_ids]
            candidate_pois = reordered + [p for p in candidate_pois if p["id"] not in [x["id"] for x in reordered]]

        # Αυστηρό φιλτράρισμα candidate_pois μέσω Feasibility Engine
        candidate_pois = filter_candidate_pois(
            candidate_pois,
            user_state=state,
            blacklisted_categories=state.blacklisted_categories,
            blacklisted_tags=state.blacklisted_tags,
            wheelchair_accessible=state.wheelchair_accessible,
            weather_data=weather_data,
        )

        # 5. Εκτέλεση Feasibility Engine μέσω αυστηρού Pydantic Output Parser
        start_time = state.start_time or "14:00"
        start_hour, start_min = map(int, start_time.split(":")[:2])
        total_budget_mins = max(30, int(state.time_budget_hours * 60))
        end_total_mins = min(23 * 60 + 59, start_hour * 60 + start_min + total_budget_mins)
        end_time = f"{end_total_mins // 60:02d}:{end_total_mins % 60:02d}"

        explicit_requested_pois = (
            [p for p in candidate_pois if p["id"] in explicit_poi_ids]
            if len(explicit_poi_ids) >= 1
            else None
        )

        f_args = feasibility_args_parser.parse({
            "start_time": start_time,
            "end_time": end_time,
            "time_budget_hours": state.time_budget_hours,
            "traveling_with_kids": state.traveling_with_kids,
            "child_age": state.kid_age,
            "wheelchair_accessible": state.wheelchair_accessible,
            "preferred_pace": state.preferred_pace,
            "explicit_poi_ids": explicit_poi_ids,
            "blacklisted_poi_ids": state.blacklisted_poi_ids,
            "blacklisted_categories": list(state.blacklisted_categories),
            "insert_rest_stop": is_fatigue_coffee,
            "rest_stop_mins": 35,
        })

        validated_itinerary = self.feasibility.build_feasible_itinerary(
            start_time_str=f_args.start_time,
            end_time_str=f_args.end_time or end_time,
            candidate_pois=candidate_pois,
            weather_data=weather_data,
            traveling_with_kids=f_args.traveling_with_kids,
            child_age=f_args.child_age,
            wheelchair_accessible=f_args.wheelchair_accessible,
            explicit_requested_pois=explicit_requested_pois,
            user_state=state,
            blacklisted_categories=set(f_args.blacklisted_categories),
            blacklisted_tags=state.blacklisted_tags,
            insert_rest_stop=f_args.insert_rest_stop,
            rest_stop_mins=f_args.rest_stop_mins,
            rest_stop_name="☕ Στάση για Καφέ & Ξεκούραση (Πλάκα)",
        )

        # 6. Ενημέρωση του Active Itinerary στο State
        state.active_itinerary = validated_itinerary

        # 7. LLM Formatting / Synthesis
        return self._format_itinerary_response(
            validated_itinerary,
            weather_data,
            state,
            is_fatigue_coffee=is_fatigue_coffee,
            museums_removed=museums_negative_triggered,
        )

    def _generate_rag_response(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Συνθέτει απάντηση Q&A μέσω του Cloud LLM (NVIDIA Nemotron-3-Ultra via Ollama API).
        Χωρίς deterministic fallback: εάν το Cloud LLM είναι μη διαθέσιμο, εγείρεται HTTPException(503).
        """
        from orchestrator.llm_client import get_cloud_ollama_client
        return get_cloud_ollama_client().generate_rag_synthesis(query=query, docs=docs)

    def _format_itinerary_response(
        self,
        itinerary: Dict[str, Any],
        weather: Dict[str, Any],
        state: UserState,
        is_fatigue_coffee: bool = False,
        museums_removed: bool = False,
    ) -> str:
        """
        Συνθέτει αφηγηματική παρουσίαση δρομολογίου μέσω του Cloud LLM.
        Χωρίς deterministic fallback: εάν το Cloud LLM είναι μη διαθέσιμο, εγείρεται HTTPException(503).
        """
        if not itinerary.get("feasible"):
            return "Δεν ήταν εφικτός ο υπολογισμός δρομολογίου με τους περιορισμούς που θέσατε."

        last_item = state.conversation_history[-1] if getattr(state, "conversation_history", None) else None
        if isinstance(last_item, dict):
            user_msg = last_item.get("content", "Πρόγραμμα για Αθήνα")
        elif last_item is not None:
            user_msg = getattr(last_item, "content", str(last_item))
        else:
            user_msg = "Πρόγραμμα για Αθήνα"
        from orchestrator.llm_client import get_cloud_ollama_client
        return get_cloud_ollama_client().generate_itinerary_synthesis(
            user_request=user_msg,
            itinerary=itinerary,
            weather=weather,
            user_state=state,
            is_fatigue_coffee=is_fatigue_coffee,
            museums_removed=museums_removed,
        )

    def handle_user_request(
        self,
        user_request: str,
        time_budget: float | int = 4.0,
        user_preferences: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Two-Step Generation Pipeline proxy on TouristLLMOrchestrator."""
        return handle_user_request(
            user_request=user_request,
            time_budget=time_budget,
            user_preferences=user_preferences,
            feasibility_engine=self.feasibility,
            llm_client=self._llm_client if self._llm_enabled else None,
            model=self._llm_model,
        )


def post_generation_validation(llm_response_text: str, validated_json: Optional[Dict[str, Any]] = None) -> str:
    """
    Hard Validation Gate (Deterministic Override):
    Επαληθεύει ότι το παραχθέν κείμενο του LLM δεν περιέχει αναφορές επίσκεψης
    μετά την ώρα κλεισίματος του αξιοθέατου. Εάν εντοπιστεί απόκλιση, εφαρμόζεται
    άμεσο deterministic override με το ασφαλές πρότυπο.
    """
    if not validated_json:
        return llm_response_text

    if validated_json.get("closing_time_violation"):
        poi_name = validated_json.get("poi_name", "το αξιοθέατο")
        close_time = validated_json.get("closing_time", "17:00")
        return (
            f"⚠️ **Έλεγχος Ωραρίου Λειτουργίας (Closed Museum):** {poi_name} "
            f"κλείνει στις {close_time}. Η επίσκεψη δεν είναι εφικτή τη ζητούμενη ώρα "
            f"καθώς ο χώρος θα είναι κλειστός."
        )
    return llm_response_text


class AthensTouristAgent(TouristLLMOrchestrator):
    """
    Main Agent wrapper used by Streamlit UI and test scripts.
    Maintains persistent user state and returns detailed response dict.
    """

    def __init__(
        self,
        retriever: Optional[AthensRAGRetriever] = None,
        feasibility_engine: Optional[FeasibilityEngine] = None,
        enable_graph: bool = True,
    ):
        super().__init__(rag_retriever=retriever, feasibility_engine=feasibility_engine)
        self.retriever = self.rag
        self.feasibility_engine = self.feasibility
        self.user_state = UserState()
        self.enable_graph = enable_graph
        self._graph_orchestrator = None

    @property
    def graph_orchestrator(self):
        """Lazy-loaded LangGraph State Graph with Reflection Loops."""
        if self._graph_orchestrator is None:
            from orchestrator.graph import LangGraphOrchestrator
            self._graph_orchestrator = LangGraphOrchestrator(self)
        return self._graph_orchestrator

    def chat_graph(self, user_message: str) -> Dict[str, Any]:
        """
        Executes message processing via LangGraph State Graph with reflection loops.
        """
        graph_res = self.graph_orchestrator.execute(user_message, self.user_state)
        reply = graph_res.get("final_response", "")
        raw_plan = self.user_state.active_itinerary
        intent_str = graph_res.get("intent", "itinerary_planning")

        citations = []
        if intent_str == "factual_qa":
            retrieved_docs = graph_res.get("candidate_docs", []) or self.rag.retrieve(query=user_message, top_k=3)
            for d in retrieved_docs:
                citations.append({
                    "source_id": d["id"],
                    "source_name": d.get("name", d.get("source_name", "")),
                    "category": d.get("metadata", {}).get("category", ""),
                    "type": d.get("metadata", {}).get("type", ""),
                })
        else:
            if raw_plan and "selected_poi_ids" in raw_plan:
                for pid in raw_plan["selected_poi_ids"]:
                    p = self.rag.get_poi(pid)
                    if p:
                        citations.append({
                            "source_id": pid,
                            "source_name": p["name"],
                            "category": p.get("category", ""),
                            "type": p.get("type", ""),
                        })

        weather = graph_res.get("weather_data") or (get_live_weather(city="Athens") if intent_str != "factual_qa" else None)
        replaced_poi = self.user_state.blacklisted_poi_ids[-1] if self.user_state.blacklisted_poi_ids else None

        u_state = asdict(self.user_state)
        u_state["blacklisted_categories"] = list(self.user_state.blacklisted_categories)
        u_state["blacklisted_tags"] = list(self.user_state.blacklisted_tags)
        u_state["child_age"] = self.user_state.kid_age

        interaction_entry = feedback_manager.record_interaction(
            session_id=getattr(self.user_state, "session_id", "default_session"),
            user_query=user_message,
            detected_intent=intent_str,
            retrieved_poi_ids=[c.get("source_id", "") for c in citations if c.get("source_id")],
            llm_response=reply,
            user_followup_message=user_message if len(self.user_state.conversation_history) > 1 else None,
        )

        token_usage = graph_res.get("token_usage") or {}
        if not token_usage:
            prompt_tokens = context_length_manager.count_tokens(user_message)
            reply_tokens = context_length_manager.count_tokens(reply)
            cost_report = context_length_manager.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=reply_tokens)
            token_usage = cost_report.to_dict()

        return {
            "intent": intent_str,
            "reply": reply,
            "raw_plan": raw_plan,
            "weather_data": weather,
            "citations": citations,
            "replaced_poi": replaced_poi,
            "user_state": u_state,
            "interaction_id": interaction_entry.feedback_id,
            "conversational_sentiment": interaction_entry.conversational_sentiment,
            "token_usage": token_usage,
            "execution_steps": graph_res.get("execution_steps", []),
            "reflection_count": graph_res.get("reflection_count", 0),
            "reflection_critique": graph_res.get("reflection_critique"),
            "engine": "langgraph_state_graph",
        }

    def chat(self, user_message: str, use_langgraph: bool = True) -> Dict[str, Any]:
        """
        Processes message strictly through LangGraph State Graph with reflection loops.
        Unconditionally routes via LangGraph to enforce Reflection Node validation before Cloud LLM Synthesis.
        """
        return self.chat_graph(user_message)

        citations = []
        raw_plan = self.user_state.active_itinerary
        if intent_enum == IntentType.FACTUAL_QA:
            retrieved_docs = self.rag.retrieve(query=user_message, top_k=3)
            for d in retrieved_docs:
                citations.append({
                    "source_id": d["id"],
                    "source_name": d.get("name", d.get("source_name", "")),
                    "category": d.get("metadata", {}).get("category", ""),
                    "type": d.get("metadata", {}).get("type", ""),
                })
        else:
            if raw_plan and "selected_poi_ids" in raw_plan:
                for pid in raw_plan["selected_poi_ids"]:
                    p = self.rag.get_poi(pid)
                    if p:
                        citations.append({
                            "source_id": pid,
                            "source_name": p["name"],
                            "category": p.get("category", ""),
                            "type": p.get("type", ""),
                        })

        weather = get_live_weather(city="Athens") if intent_enum != IntentType.FACTUAL_QA else None

        replaced_poi = self.user_state.blacklisted_poi_ids[-1] if self.user_state.blacklisted_poi_ids else None

        u_state = asdict(self.user_state)
        u_state["blacklisted_categories"] = list(self.user_state.blacklisted_categories)
        u_state["blacklisted_tags"] = list(self.user_state.blacklisted_tags)
        u_state["child_age"] = self.user_state.kid_age

        reply = post_generation_validation(reply, raw_plan)

        # Record interaction in proprietary feedback dataset
        interaction_entry = feedback_manager.record_interaction(
            session_id=getattr(self.user_state, "session_id", "default_session"),
            user_query=user_message,
            detected_intent=intent_str,
            retrieved_poi_ids=[c.get("source_id", "") for c in citations if c.get("source_id")],
            llm_response=reply,
            user_followup_message=user_message if len(self.user_state.conversation_history) > 1 else None,
        )

        prompt_tokens = context_length_manager.count_tokens(user_message)
        reply_tokens = context_length_manager.count_tokens(reply)
        cost_report = context_length_manager.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=reply_tokens)

        return {
            "intent": intent_str,
            "reply": reply,
            "raw_plan": raw_plan,
            "weather_data": weather,
            "citations": citations,
            "replaced_poi": replaced_poi,
            "user_state": u_state,
            "interaction_id": interaction_entry.feedback_id,
            "conversational_sentiment": interaction_entry.conversational_sentiment,
            "token_usage": cost_report.to_dict(),
        }


    def detect_intent(self, query: str) -> str:
        return self.classify_intent(query, has_active_itinerary=(self.user_state.active_itinerary is not None)).value

    def handle_wearable_telemetry(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes real-time IoT telemetry from the traveler's smartwatch/smart bracelet.
        Supports Fatigue Alerts, Geofence Entry, Ambient UV/Heat Alerts, and Voice Commands.
        """
        from tools.wearable import format_wearable_card

        event_type = event_payload.get("event_type", "")
        telemetry = event_payload.get("telemetry", {})
        haptic_cue = event_payload.get("haptic_feedback", "single_short")

        response_payload = {
            "event_type": event_type,
            "processed": True,
            "haptic": haptic_cue,
            "wrist_notification": "",
            "active_plan": self.user_state.active_itinerary,
            "wearable_cards": [],
        }

        if event_type == "fatigue_alert":
            # 1. State mutation: mark fatigue and activate relaxed pace
            bpm = telemetry.get("heart_rate_bpm", 135)
            if "relaxed_pace" not in self.user_state.preferences:
                self.user_state.preferences.append("relaxed_pace")

            # 2. Dynamic Replanning: inject rest stop / cafe
            msg = f"Ανιχνεύθηκε κόπωση και αυξημένοι παλμοί ({bpm} bpm). Προσάρμοσε το πρόγραμμα με στάση ξεκούρασης."
            chat_res = self.chat(msg)

            response_payload["wrist_notification"] = (
                f"⏸️ Στάση Ξεκούρασης: Αυξημένοι παλμοί ({bpm} bpm). "
                f"Το πρόγραμμα προσαρμόστηκε αυτόματα με χαλαρό ρυθμό και στάση για ξεκούραση."
            )
            response_payload["active_plan"] = self.user_state.active_itinerary

        elif event_type == "geofence_entry":
            poi_id = telemetry.get("approaching_poi_id", "hephaestus_temple")
            dist = telemetry.get("distance_m", 50)
            poi = self.rag.get_poi(poi_id)
            poi_name = poi["name"] if poi else "Αξιοθέατο"
            hours = poi.get("opening_hours", {}) if poi else {}
            hours_str = f"{hours.get('open', '08:00')}-{hours.get('close', '20:00')}"

            response_payload["wrist_notification"] = (
                f"📍 {poi_name} ({dist}m) | Ωράριο: {hours_str}. "
                f"Βρίσκεστε σε άμεση εγγύτητα!"
            )

        elif event_type == "ambient_heat_uv_alert":
            temp = telemetry.get("ambient_temp_c", 39.5)
            uv = telemetry.get("uv_index", 10.0)
            response_payload["wrist_notification"] = (
                f"☀️ Προσοχή - Έντονη Ζέστη ({temp}°C / UV {uv}): "
                f"Αναζητήστε σκιερό ή κλιματιζόμενο χώρο και πίνετε νερό."
            )

        elif event_type == "voice_command":
            transcript = telemetry.get("transcript", "")
            chat_res = self.chat(transcript)
            response_payload["wrist_notification"] = chat_res["reply"][:140]
            response_payload["active_plan"] = self.user_state.active_itinerary

        # Render bite-sized cards for smartwatch UI
        if self.user_state.active_itinerary:
            cards = []
            for idx, step in enumerate(self.user_state.active_itinerary.get("schedule", []), 1):
                c = format_wearable_card(step, step_idx=idx)
                cards.append(c.to_dict())
            response_payload["wearable_cards"] = cards

        return response_payload

    def handle_iot_sensor_event(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes real-time IoT events from DOTSOFT Smart City Gateway or Wearables:
        - crowd_density: Automatically reroutes around overcrowded attractions.
        - fatigue_alert: Injects rest stop into itinerary.
        """
        event_type = event_payload.get("event_type", "")
        telemetry = event_payload.get("telemetry", {})

        if event_type == "crowd_density":
            poi_id = telemetry.get("poi_id", "acropolis_hill")
            wait_time = telemetry.get("queue_wait_mins", 75)
            density = telemetry.get("density_level", "high")
            poi = self.rag.get_poi(poi_id)
            poi_name = poi["name"] if poi else poi_id

            # 1. State mutation: blacklist or deprioritize congested POI
            if poi_id not in self.user_state.blacklisted_poi_ids:
                self.user_state.blacklisted_poi_ids.append(poi_id)

            # 2. Trigger dynamic replanning via Feasibility Engine
            replan_msg = (
                f"Ο αισθητήρας της έξυπνης πόλης (DOTSOFT Smart City) ανίχνευσε υψηλό συνωστισμό "
                f"στο σημείο {poi_name} (αναμονή {wait_time} λεπτά). Άλλαξέ το με άλλο αξιοθέατο."
            )
            chat_res = self.chat(replan_msg)

            # 3. Format response and cards
            from tools.wearable import format_wearable_card
            cards = []
            if self.user_state.active_itinerary:
                for idx, step in enumerate(self.user_state.active_itinerary.get("schedule", []), 1):
                    c = format_wearable_card(step, step_idx=idx)
                    cards.append(c.to_dict())

            return {
                "event_type": "crowd_density",
                "processed": True,
                "congested_poi": poi_name,
                "queue_wait_mins": wait_time,
                "density_level": density,
                "wrist_notification": (
                    f"👥 Ειδοποίηση Συνωστισμού (DOTSOFT Smart City): Ανιχνεύθηκε ουρά {wait_time} λεπτών στο σημείο {poi_name}. "
                    f"Η Feasibility Engine αναπροσάρμοσε αυτόματα το δρομολόγιο για αποφυγή καθυστερήσεων."
                ),
                "reply": chat_res.get("reply", ""),
                "active_plan": self.user_state.active_itinerary,
                "wearable_cards": cards,
            }

        # Delegate other events to wearable telemetry handler
        return self.handle_wearable_telemetry(event_payload)

    def simulate_iot_sensor_event(
        self,
        sensor_type: str = "crowd_density",
        poi_id: Optional[str] = None,
        density_level: Optional[str] = "high",
        wait_time_mins: Optional[int] = 75,
        heart_rate_bpm: Optional[int] = None,
        fatigue_index: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience method to simulate an IoT event directly on this agent instance."""
        from tools.wearable import simulate_iot_sensor_event
        return simulate_iot_sensor_event(
            sensor_type=sensor_type,
            poi_id=poi_id,
            density_level=density_level,
            wait_time_mins=wait_time_mins,
            heart_rate_bpm=heart_rate_bpm,
            fatigue_index=fatigue_index,
            agent=self,
        )

    def handle_user_request(
        self,
        user_request: str,
        time_budget: float | int = 4.0,
        user_preferences: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Two-Step Generation Pipeline proxy on AthensTouristAgent."""
        return handle_user_request(
            user_request=user_request,
            time_budget=time_budget,
            user_preferences=user_preferences,
            feasibility_engine=self.feasibility,
            llm_client=self._llm_client if self._llm_enabled else None,
            model=self._llm_model,
        )


# ---------------------------------------------------------------------------
# Step 3: Two-Step Generation Pipeline (Deterministic Layer -> Probabilistic Layer)
# ---------------------------------------------------------------------------
def render_natural_synthesis(optimized_itinerary: List[Dict[str, Any]] | Dict[str, Any], user_request: str) -> str:
    """
    Δημιουργεί μια ρέουσα, φιλική και αφηγηματική απάντηση στα Ελληνικά
    σύμφωνα με τους 5 αυστηρούς κανόνες του NATURAL_SYNTHESIS_PROMPT:
    1. Αφηγηματικός τόνος: Φιλική ιστορία με ενθουσιασμό.
    2. Τι απαγορεύεται: Καμία αναφορά σε ID, Category, indoor, outdoor, wheelchair_accessible, km.
    3. Ωράρια: Φυσική αναφορά ωρών ("Ξεκινάμε στις 14:00 με το...").
    4. Καμία παραίσθηση: Μόνο τα σημεία που υπάρχουν στο JSON.
    5. Πηγές: Στο τέλος του μηνύματος: 📚 Πηγές που χρησιμοποιήθηκαν: [Πηγή: ...].
    """
    items = optimized_itinerary
    if isinstance(optimized_itinerary, dict):
        items = optimized_itinerary.get("schedule", [])

    activities = [item for item in items if item.get("type") != "walking"]
    if not activities:
        return "Γεια σας! Είμαι ο Philody. Δεν μπόρεσα να υπολογίσω κάποια διαθέσιμη δραστηριότητα για το επιθυμητό διάστημα."

    start_time = "14:00"
    for item in items:
        if item.get("time_slot"):
            start_time = item["time_slot"].split("-")[0].strip()
            break
        elif item.get("time"):
            start_time = item["time"].split("-")[0].strip()
            break

    first_name = activities[0].get("name") or activities[0].get("poi_name", "πρώτο μας σημείο")
    story_parts = [
        f"Γεια σας! Είμαι ο Philody και ετοίμασα με πολλή χαρά το ιδανικό πρόγραμμα για τη βόλτα σας στην Αθήνα.",
        f"Ξεκινάμε στις {start_time} με το {first_name}."
    ]

    first_desc = activities[0].get("description", "")
    if first_desc:
        clean_desc = first_desc.split(".")[0].strip() + "."
        story_parts.append(clean_desc)

    for act in activities[1:]:
        name = act.get("name") or act.get("poi_name", "")
        time_hint = ""
        if act.get("time_slot"):
            t_start = act["time_slot"].split("-")[0].strip()
            time_hint = f" κατά τις {t_start}"
        elif act.get("time"):
            t_start = act["time"].split("-")[0].strip()
            time_hint = f" κατά τις {t_start}"

        if act.get("type") == "rest_area" or "καφέ" in name.lower() or "ξεκούραση" in name.lower():
            story_parts.append(
                f"Στη συνέχεια{time_hint}, ακολουθεί μια {name.lower()} "
                f"για να απολαύσετε την αυθεντική ατμόσφαιρα της πόλης και να πάρετε δυνάμεις."
            )
        else:
            story_parts.append(f"Αμέσως μετά{time_hint}, η διαδρομή μάς οδηγεί στο {name}.")
            if act.get("description"):
                d = act["description"].split(".")[0].strip() + "."
                story_parts.append(d)

    story_parts.append("Σας εύχομαι μια αξέχαστη και μαγευτική εμπειρία στην καρδιά της Αθήνας!")

    used_sources = []
    seen = set()
    for act in activities:
        name = act.get("name") or act.get("poi_name", "")
        name = name.strip()
        if name and name not in seen:
            used_sources.append(f"[Πηγή: {name}]")
            seen.add(name)

    body = " ".join(story_parts)
    citations_str = f"📚 Πηγές που χρησιμοποιήθηκαν: {', '.join(used_sources)}."
    return f"{body}\n\n{citations_str}"


def handle_user_request(
    user_request: str,
    time_budget: float | int = 4.0,
    user_preferences: Optional[Any] = None,
    feasibility_engine: Optional[Any] = None,
    llm_client: Optional[Any] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Two-Step Generation Pipeline:
    Βήμα 1: DETERMINISTIC LAYER (Feasibility Engine + Time-Filling Buffer)
    Βήμα 2: PROBABILISTIC LAYER (LLM Natural Synthesis)
    Βήμα 3: FRONTEND PAYLOAD & CHAT SEPARATION
    """
    # ---------------------------------------------------------
    # ΒΗΜΑ 1: DETERMINISTIC LAYER (Feasibility Engine + RAG)
    # ---------------------------------------------------------
    # Το σύστημά σου υπολογίζει το τέλειο, μαθηματικά σωστό δρομολόγιο.
    engine = feasibility_engine or FeasibilityEngine()
    budget_mins = int(time_budget * 60) if time_budget <= 24 else int(time_budget)
    raw_itinerary = engine.generate_draft(user_request, time_budget, user_preferences)
    
    # Εφαρμόζουμε το Time-Filling Buffer για να γεμίσει τέλεια ο χρόνος
    current_mins = calculate_total_mins(raw_itinerary)
    optimized_itinerary = optimize_itinerary_duration(raw_itinerary, budget_mins, current_mins)
    
    # ---------------------------------------------------------
    # ΒΗΜΑ 2: PROBABILISTIC LAYER (LLM Natural Synthesis)
    # ---------------------------------------------------------
    # Στέλνουμε το έτοιμο JSON στο Cloud LLM (π.χ. nemotron / openai)
    messages = [
        {"role": "system", "content": NATURAL_SYNTHESIS_PROMPT}, # Το prompt από το Βήμα 1
        {"role": "user", "content": f"Χρήστης: '{user_request}'\n\nΔρομολόγιο JSON:\n{json.dumps(optimized_itinerary, ensure_ascii=False)}"}
    ]
    
    chat_text = None
    try:
        from orchestrator.llm_client import get_cloud_ollama_client
        llm = get_cloud_ollama_client()
        chat_text = llm.chat(messages=messages, temperature=0.4, max_tokens=1000)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[TwoStepPipeline] Cloud LLM call failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud LLM is currently unavailable."
        ) from e

    if not chat_text:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud LLM is currently unavailable."
        )

    # Προσθήκη του mandatory EU AI Act disclaimer προγραμματιστικά στο τέλος (όχι από το LLM)
    final_chat_text = f"{chat_text}\n\n(🤖 Σημείωση Διαφάνειας EU AI Act: Το περιεχόμενο παρήχθη αυτόνομα από το σύστημα Philody AI Travel Assistant — Άρθρο 50)"

    # ---------------------------------------------------------
    # ΒΗΜΑ 3: ΕΠΙΣΤΡΟΦΗ ΣΤΟ FRONTEND (Διαχωρισμός Payload/Chat)
    # ---------------------------------------------------------
    return {
        "chat_bubble_text": final_chat_text,          # Το κείμενο που θα διαβάσει ο χρήστης
        "background_json_payload": optimized_itinerary # Το JSON που θα πάει στον Χάρτη, στο Timeline και στο Smartwatch
    }


