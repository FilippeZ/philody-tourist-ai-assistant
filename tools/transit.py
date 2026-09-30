"""
Public Transit & Real-Time Mobility Mock API for Athens (tools/transit.py).

Simulates the Athens Urban Transport Organization (OASA / STASY / OSY):
- Metro Line 1 (Green), Line 2 (Red), Line 3 (Blue)
- Tram lines (T6, T7) & Key Central Bus lines (040, 230, 025)
- Real-time station arrival times & schedules
- Point-to-point transit routing between Athens POIs & Landmarks
- Wheelchair accessibility & step-free elevator status
- Live transport alerts & disruption notices
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas for Transit Tool
# ---------------------------------------------------------------------------
class TransitRouteRequest(BaseModel):
    origin: str = Field(..., description="Starting station, address, or POI in Athens")
    destination: str = Field(..., description="Target station, address, or POI in Athens")
    transit_mode: str = Field(default="all", description="Mode preference: all, metro, bus, tram, walking")
    wheelchair_accessible: bool = Field(default=False, description="Require 100% step-free / elevator access")
    departure_time: str = Field(default="14:00", description="Departure time HH:MM")


class TransitLeg(BaseModel):
    mode: str  # walking, metro, bus, tram
    line: Optional[str] = None  # e.g. "Line 2 (Red)", "Bus 230"
    direction: Optional[str] = None
    from_stop: str
    to_stop: str
    duration_mins: int
    distance_km: float
    instructions: str
    wheelchair_accessible: bool = True


class TransitRouteResponse(BaseModel):
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    total_duration_mins: int
    total_distance_km: float
    fare_eur: float
    ticket_type: str
    wheelchair_accessible: bool
    legs: List[TransitLeg]
    summary_text: str
    status: str = "on_time"


# ---------------------------------------------------------------------------
# Athens Stations & Network Knowledge
# ---------------------------------------------------------------------------
ATHENS_STATIONS = {
    "syntagma": {
        "name": "Σύνταγμα (Syntagma)",
        "lines": ["Metro Line 2 (Red)", "Metro Line 3 (Blue)", "Tram T6"],
        "lat": 37.9753,
        "lon": 23.7350,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["acropolis_hill", "syntagma_changing_guards", "national_garden", "benaki_museum_greek_culture", "cycladic_art_museum"],
    },
    "acropolis": {
        "name": "Ακρόπολη (Akropoli)",
        "lines": ["Metro Line 2 (Red)"],
        "lat": 37.9686,
        "lon": 23.7297,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["acropolis_museum", "acropolis_hill", "plaka_historic_walk", "anafiotika"],
    },
    "monastiraki": {
        "name": "Μοναστηράκι (Monastiraki)",
        "lines": ["Metro Line 1 (Green)", "Metro Line 3 (Blue)"],
        "lat": 37.9765,
        "lon": 23.7258,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["monastiraki_flea_market", "hephaestus_temple", "ancient_agora", "plaka_historic_walk"],
    },
    "thissio": {
        "name": "Θησείο (Thissio)",
        "lines": ["Metro Line 1 (Green)"],
        "lat": 37.9768,
        "lon": 23.7208,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["hephaestus_temple", "ancient_agora"],
    },
    "panepistimio": {
        "name": "Πανεπιστήμιο (Panepistimio)",
        "lines": ["Metro Line 2 (Red)"],
        "lat": 37.9800,
        "lon": 23.7330,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["national_archaeological_museum"],
    },
    "omonia": {
        "name": "Ομόνοια (Omonia)",
        "lines": ["Metro Line 1 (Green)", "Metro Line 2 (Red)"],
        "lat": 37.9840,
        "lon": 23.7280,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["national_archaeological_museum"],
    },
    "evangelismos": {
        "name": "Ευαγγελισμός (Evangelismos)",
        "lines": ["Metro Line 3 (Blue)"],
        "lat": 37.9760,
        "lon": 23.7460,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["cycladic_art_museum", "benaki_museum_greek_culture", "goulandris_modern_art", "lycabettus_hill"],
    },
    "victoria": {
        "name": "Βικτώρια (Victoria)",
        "lines": ["Metro Line 1 (Green)"],
        "lat": 37.9930,
        "lon": 23.7300,
        "wheelchair_elevators": True,
        "step_free": True,
        "connects_to": ["national_archaeological_museum"],
    },
}

POI_TO_STATION_MAP = {
    "acropolis_museum": "acropolis",
    "acropolis_hill": "acropolis",
    "parthenon": "acropolis",
    "plaka_historic_walk": "acropolis",
    "anafiotika": "acropolis",
    "syntagma_changing_guards": "syntagma",
    "national_garden": "syntagma",
    "monastiraki_flea_market": "monastiraki",
    "hephaestus_temple": "thissio",
    "ancient_agora": "thissio",
    "national_archaeological_museum": "omonia",
    "cycladic_art_museum": "evangelismos",
    "benaki_museum_greek_culture": "syntagma",
    "goulandris_modern_art": "evangelismos",
    "lycabettus_hill": "evangelismos",
    "panathenaic_stadium": "syntagma",
    "hellenic_children_museum": "evangelismos",
    "eugenides_planetarium": "syntagma",
}


def _resolve_station_key(query: str) -> str:
    """Finds matching station key from natural language input."""
    q = query.lower().strip()
    if any(w in q for w in ["ακρόπολ", "ακροπολ", "acropol"]):
        return "acropolis"
    if any(w in q for w in ["σύνταγμα", "συνταγμα", "syntagma"]):
        return "syntagma"
    if any(w in q for w in ["μοναστηράκι", "μοναστηρακι", "monastiraki"]):
        return "monastiraki"
    if any(w in q for w in ["θησείο", "θησειο", "thissio"]):
        return "thissio"
    if any(w in q for w in ["ομόνοια", "ομονοια", "omonia"]):
        return "omonia"
    if any(w in q for w in ["πανεπιστήμιο", "πανεπιστημιο", "panepistimio"]):
        return "panepistimio"
    if any(w in q for w in ["ευαγγελισμ", "evangelismos"]):
        return "evangelismos"
    if any(w in q for w in ["βικτώρια", "βικτωρια", "victoria"]):
        return "victoria"

    # POI lookup
    for poi_key, st_key in POI_TO_STATION_MAP.items():
        if poi_key in q or poi_key.replace("_", " ") in q:
            return st_key

    return "syntagma"


def get_transit_route(
    origin: str,
    destination: str,
    wheelchair_accessible: bool = False,
    departure_time: str = "14:00",
) -> Dict[str, Any]:
    """
    Computes a realistic public transit route across the Athens Metro/Bus network.
    Includes walking legs, transfers, line details, and wheelchair accessibility notes.
    """
    st_orig_key = _resolve_station_key(origin)
    st_dest_key = _resolve_station_key(destination)

    st_orig = ATHENS_STATIONS.get(st_orig_key, ATHENS_STATIONS["syntagma"])
    st_dest = ATHENS_STATIONS.get(st_dest_key, ATHENS_STATIONS["acropolis"])

    # Calculate hours / minutes
    try:
        sh, sm = map(int, departure_time.split(":")[:2])
    except Exception:
        sh, sm = 14, 0

    legs: List[TransitLeg] = []
    total_duration = 0
    total_dist = 0.0

    # Same station / walking distance
    if st_orig_key == st_dest_key:
        legs.append(TransitLeg(
            mode="walking",
            from_stop=origin,
            to_stop=destination,
            duration_mins=8,
            distance_km=0.6,
            instructions=f"🚶 Περπάτημα από '{origin}' προς '{destination}' μέσω πεζοδρόμου.",
            wheelchair_accessible=True,
        ))
        total_duration = 8
        total_dist = 0.6
    else:
        # 1. Walk to origin station
        legs.append(TransitLeg(
            mode="walking",
            from_stop=origin,
            to_stop=st_orig["name"],
            duration_mins=4,
            distance_km=0.3,
            instructions=f"🚶 Περπάτημα 300μ. προς το σταθμό Μετρό {st_orig['name']}.",
            wheelchair_accessible=True,
        ))

        # Check direct metro connection
        common_lines = [l for l in st_orig["lines"] if l in st_dest["lines"] and "Metro" in l]
        if common_lines:
            metro_line = common_lines[0]
            # direct ride
            ride_mins = 4
            legs.append(TransitLeg(
                mode="metro",
                line=metro_line,
                direction=st_dest["name"],
                from_stop=st_orig["name"],
                to_stop=st_dest["name"],
                duration_mins=ride_mins,
                distance_km=1.8,
                instructions=f"🚇 Επιβίβαση στη {metro_line} με κατεύθυνση {st_dest['name']} (αποβίβαση σε 1-2 στάσεις).",
                wheelchair_accessible=st_orig["wheelchair_elevators"] and st_dest["wheelchair_elevators"],
            ))
            total_duration = 4 + ride_mins
            total_dist = 0.3 + 1.8
        else:
            # Transfer at Syntagma or Monastiraki
            transfer_station = "syntagma" if st_orig_key != "syntagma" else "monastiraki"
            st_trans = ATHENS_STATIONS[transfer_station]

            line1 = [l for l in st_orig["lines"] if "Metro" in l][0]
            line2 = [l for l in st_dest["lines"] if "Metro" in l][0]

            legs.append(TransitLeg(
                mode="metro",
                line=line1,
                direction=st_trans["name"],
                from_stop=st_orig["name"],
                to_stop=st_trans["name"],
                duration_mins=5,
                distance_km=2.2,
                instructions=f"🚇 Επιβίβαση στη {line1} μέχρι το σταθμό μετεπιβίβασης {st_trans['name']}.",
                wheelchair_accessible=True,
            ))
            legs.append(TransitLeg(
                mode="metro",
                line=line2,
                direction=st_dest["name"],
                from_stop=st_trans["name"],
                to_stop=st_dest["name"],
                duration_mins=5,
                distance_km=2.0,
                instructions=f"🔄 Μετεπιβίβαση στη {line2} και αποβίβαση στο σταθμό {st_dest['name']}.",
                wheelchair_accessible=True,
            ))
            total_duration = 4 + 5 + 5
            total_dist = 0.3 + 2.2 + 2.0

        # 3. Walk to final destination
        legs.append(TransitLeg(
            mode="walking",
            from_stop=st_dest["name"],
            to_stop=destination,
            duration_mins=4,
            distance_km=0.3,
            instructions=f"🚶 Έξοδος από το σταθμό {st_dest['name']} και σύντομο περπάτημα προς '{destination}'.",
            wheelchair_accessible=True,
        ))
        total_duration += 4
        total_dist += 0.3

    arr_total_mins = (sh * 60 + sm + total_duration) % (24 * 60)
    arr_h = arr_total_mins // 60
    arr_m = arr_total_mins % 60
    arrival_time = f"{arr_h:02d}:{arr_m:02d}"

    summary = (
        f"🚇 **Διαδρομή Μέσων Μαζικής Μεταφοράς Αθήνας (ΟΑΣΑ / ΣΤΑΣΥ)**\n"
        f"• **Αφετηρία:** {origin} ({departure_time})\n"
        f"• **Προορισμός:** {destination} ({arrival_time})\n"
        f"• **Συνολικός Χρόνος:** {total_duration} λεπτά | Απόσταση: {total_dist:.1f} km\n"
        f"• **Ενιαίο Εισιτήριο ΟΑΣΑ:** 1,20 € (ισχύει για 90 λεπτά σε όλα τα ΜΜΜ)\n"
        f"• **Προσβασιμότητα ΑμεΑ:** {'✅ Πλήρως προσβάσιμο (ανελκυστήρες σε όλους τους σταθμούς)' if not wheelchair_accessible or True else '⚠️ Μερικώς προσβάσιμο'}\n"
    )

    resp = TransitRouteResponse(
        origin=origin,
        destination=destination,
        departure_time=departure_time,
        arrival_time=arrival_time,
        total_duration_mins=total_duration,
        total_distance_km=round(total_dist, 2),
        fare_eur=1.20,
        ticket_type="Ενιαίο Εισιτήριο 90 λεπτών (ΟΑΣΑ)",
        wheelchair_accessible=True,
        legs=legs,
        summary_text=summary,
        status="on_time",
    )
    return resp.model_dump()


def get_station_schedule(station_name: str, line: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns real-time arrivals for an Athens metro/tram station.
    """
    st_key = _resolve_station_key(station_name)
    st = ATHENS_STATIONS.get(st_key, ATHENS_STATIONS["syntagma"])

    arrivals = []
    for l in st["lines"]:
        arrivals.append({
            "line": l,
            "destination": "Ελληνικό" if "Line 2" in l else ("Αεροδρόμιο" if "Line 3" in l else "Κηφισιά"),
            "arrival_in_mins": 3,
            "next_arrival_in_mins": 8,
            "status": "on_time",
            "crowd_level": "moderate",
        })
        arrivals.append({
            "line": l,
            "destination": "Ανθούπολη" if "Line 2" in l else ("Δημοτικό Θέατρο Πειραιά" if "Line 3" in l else "Πειραιάς"),
            "arrival_in_mins": 5,
            "next_arrival_in_mins": 11,
            "status": "on_time",
            "crowd_level": "low",
        })

    return {
        "station_name": st["name"],
        "lines": st["lines"],
        "wheelchair_accessible": st["wheelchair_elevators"],
        "step_free_access": st["step_free"],
        "live_arrivals": arrivals,
        "disruptions": [],
        "ticket_machines": "Διαθέσιμα Αυτόματα Μηχανήματα ATH.ENA Ticket (POS & Μετρητά)",
    }


def get_transit_alerts() -> Dict[str, Any]:
    """Returns live network operation status & alerts for Athens public transport."""
    return {
        "network": "ΟΑΣΑ / ΣΤΑΣΥ (Μετρό, Τραμ, Λεωφορεία)",
        "status": "normal_operation",
        "alerts": [
            {
                "type": "info",
                "message": "Κανονικά δρομολόγια σε όλες τις γραμμές Μετρό 1, 2, 3 και Τραμ.",
                "valid_until": "23:59",
            },
            {
                "type": "tip",
                "message": "Τουριστικό Εισιτήριο 3 Ημερών (20,00 €) περιλαμβάνει απεριόριστες μετακινήσεις και διαδρομή από/προς Αεροδρόμιο.",
                "valid_until": "ongoing",
            }
        ],
    }


# Tool schema for OpenAI / LLM function calling
TRANSIT_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_transit_route",
        "description": (
            "Αναζητά δρομολόγια δημόσιας συγκοινωνίας (Μετρό, Τραμ, Λεωφορεία ΟΑΣΑ) στην Αθήνα μεταξύ "
            "δύο σημείων ή αξιοθέατων. Επιστρέφει στάσεις, χρόνο, γραμμές μετεπιβίβασης, κόστος εισιτηρίου "
            "και πληροφορίες προσβασιμότητας για αμαξίδια."
        ),
        "parameters": TransitRouteRequest.model_json_schema(),
    },
}
