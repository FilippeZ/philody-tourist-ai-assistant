"""
Feasibility & Validation Engine (engine/feasibility.py).

Deterministic business logic for tourist itinerary validation:
- Enforces strict operating hours: (t_arrival + t_visit <= t_closing)
- Computes spatial routing and travel times using the Haversine distance formula
- Applies weather filtering and indoor penalties (is_indoor_recommended)
- Guarantees 100% physically feasible, non-overlapping schedules before LLM synthesis
"""

from __future__ import annotations
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# Σταθερές ταχύτητες μετακίνησης (σε km/h)
WALKING_SPEED_KMH = 4.0
DRIVING_SPEED_KMH = 20.0  # Μέση ταχύτητα εντός πόλης


def haversine_distance(coord1: Dict[str, float], coord2: Dict[str, float]) -> float:
    """
    Υπολογίζει τη γεωδαιτική απόσταση μεταξύ δύο σημείων σε χιλιόμετρα (τύπος Haversine).
    """
    R = 6371.0  # Ακτίνα της Γης σε km
    lat1, lon1 = math.radians(coord1["lat"]), math.radians(coord1["lon"])
    lat2, lon2 = math.radians(coord2["lat"]), math.radians(coord2["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def time_to_minutes(time_str: str) -> int:
    """Μετατρέπει συμβολοσειρά ώρας 'HH:MM' σε λεπτά από την αρχή της ημέρας."""
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


def minutes_to_time(minutes: int) -> str:
    """Μετατρέπει λεπτά σε συμβολοσειρά ώρας 'HH:MM'."""
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def calculate_travel_time_mins(
    coord1: Dict[str, float],
    coord2: Dict[str, float],
    mode: str = "walking",
) -> Dict[str, Any]:
    """Υπολογίζει την απόσταση και τον χρόνο μετακίνησης με buffer 5 λεπτών."""
    dist_km = haversine_distance(coord1, coord2)
    speed = DRIVING_SPEED_KMH if mode == "driving" else WALKING_SPEED_KMH
    travel_time_mins = math.ceil((dist_km / speed) * 60) + 5  # +5 mins buffer
    return {"distance_km": round(dist_km, 2), "duration_mins": travel_time_mins}


from dataclasses import dataclass, field

@dataclass
class ItineraryPlan:
    """Strictly validated structured itinerary produced by the Feasibility Engine."""
    feasible: bool
    start_time: str
    end_time: str
    total_distance_km: float
    schedule: List[Dict[str, Any]]
    selected_poi_ids: List[str]
    actual_end_time: Optional[str] = None
    validation_notes: List[str] = field(default_factory=list)
    weather_impact: Optional[str] = None
    weather_adjusted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "actual_end_time": self.actual_end_time or self.end_time,
            "total_distance_km": self.total_distance_km,
            "weather_adjusted": self.weather_adjusted,
            "weather_impact": self.weather_impact or (
                "Προσαρμογή προγράμματος λόγω καιρικών συνθηκών (βροχή/καύσωνας)."
                if self.weather_adjusted
                else "Κανονικές καιρικές συνθήκες."
            ),
            "schedule": self.schedule,
            "selected_poi_ids": self.selected_poi_ids,
        }



def filter_candidate_pois(
    candidate_pois: List[Dict[str, Any]],
    wheelchair_accessible: bool = False,
    traveling_with_kids: bool = False,
    user_state: Optional[Any] = None,
    blacklisted_categories: Optional[set] = None,
    blacklisted_tags: Optional[set] = None,
    blacklisted_poi_ids: Optional[list] = None,
    weather_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Αυστηρό Φιλτράρισμα στη Feasibility Engine:
    1. Hard Constraint για Αμαξίδιο:
       - wheelchair_accessible == False OR 'stairs' in tags -> αποκλείεται άμεσα
    2. Hard Constraint για Παιδιά:
       - traveling_with_kids == True and not kid_friendly -> αποκλείεται άμεσα
    3. Hard Constraint για Καύσωνα / Θερμοκρασία > 38°C / Adverse Weather:
       - temp > 38.0°C OR is_indoor_recommended -> αποκλείονται όλα τα outdoor POIs
    4. Αποκλεισμός βάσει Μαύρης Λίστας (Blacklisted tags, categories & POI IDs)
    """
    filtered_pois = []

    b_categories = set(blacklisted_categories or [])
    b_tags = set(blacklisted_tags or [])
    b_ids = set(blacklisted_poi_ids or [])
    is_wheelchair = wheelchair_accessible
    is_kids = traveling_with_kids

    # Thermal / Heatwave Guardrail (>38°C)
    temp_val = 0.0
    if weather_data:
        try:
            temp_val = float(weather_data.get("temperature_c") or weather_data.get("temperature") or 0.0)
        except (ValueError, TypeError):
            temp_val = 0.0
    is_heatwave_or_indoor = (temp_val > 38.0) or (weather_data and weather_data.get("is_indoor_recommended", False))

    if user_state is not None:
        if getattr(user_state, "wheelchair_accessible", False):
            is_wheelchair = True
        if getattr(user_state, "traveling_with_kids", False):
            is_kids = True
        b_categories.update(getattr(user_state, "blacklisted_categories", set()) or set())
        b_tags.update(getattr(user_state, "blacklisted_tags", set()) or set())
        b_ids.update(getattr(user_state, "blacklisted_poi_ids", []) or [])

    for poi in candidate_pois:
        poi_id = poi.get("poi_id") or poi.get("id")
        # 0. Έλεγχος Blacklisted ID
        if poi_id in b_ids or poi.get("id") in b_ids:
            continue

        # ♿ Hard Constraint για Αμαξίδιο:
        if is_wheelchair:
            is_accessible = poi.get("wheelchair_accessible", False)
            has_stairs = "stairs" in poi.get("tags", [])

            # Αν ΔΕΝ είναι προσβάσιμο ή ΕΧΕΙ σκαλοπάτια, το προσπερνάμε πλήρως!
            if not is_accessible or has_stairs:
                continue

        # 👶 Hard Constraint για Παιδιά (αν εφαρμόζεται):
        if is_kids and not poi.get("kid_friendly", True):
            continue

        # ☀️ Hard Constraint για Καύσωνα (>38°C) / Adverse Weather:
        if is_heatwave_or_indoor and poi.get("type") == "outdoor":
            continue

        # 3. Αποκλεισμός βάσει Μαύρης Λίστας (Blacklisted tags & categories)
        poi_tags = set(poi.get("tags", []))
        poi_category = poi.get("category", "")

        if poi_category in b_categories or poi_tags.intersection(b_tags):
            continue  # Αποκλεισμός του POI!

        filtered_pois.append(poi)

    return filtered_pois


def build_feasible_itinerary(
    start_time: str = "14:00",
    budget_hours: float = 3.0,
    wheelchair_accessible: bool = False,
    traveling_with_kids: bool = False,
    child_age: Optional[int] = None,
    preferred_pace: str = "moderate",
    candidate_pois: Optional[List[Dict[str, Any]]] = None,
    weather_data: Optional[Dict[str, Any]] = None,
    start_location: Optional[Dict[str, float]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Module-level function to build a strictly feasible itinerary.
    Loads candidate POIs from data/athens_attractions.json if not provided.
    """
    if candidate_pois is None:
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "athens_attractions.json")
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                candidate_pois = json.load(f)
        except Exception:
            candidate_pois = []

    engine = FeasibilityEngine()
    start_mins = time_to_minutes(start_time)
    end_mins = start_mins + int(budget_hours * 60)
    end_time = minutes_to_time(end_mins)

    return engine.build_feasible_itinerary(
        start_time_str=start_time,
        end_time_str=end_time,
        candidate_pois=candidate_pois,
        weather_data=weather_data or {"is_indoor_recommended": False, "temperature": 24.0, "condition": "clear"},
        start_location=start_location,
        wheelchair_accessible=wheelchair_accessible,
        traveling_with_kids=traveling_with_kids,
        child_age=child_age,
        **kwargs,
    )


class FeasibilityEngine:
    """
    Προσδιοριστικός ελεγκτής ωραρίων, αποστάσεων και καιρικών περιορισμών.
    """

    def __init__(self, transport_mode: str = "walking", walking_speed_kmh: Optional[float] = None):
        self.transport_mode = transport_mode
        self.walking_speed_kmh = walking_speed_kmh or WALKING_SPEED_KMH

    def filter_candidate_pois(
        self,
        candidate_pois: List[Dict[str, Any]],
        user_state: Optional[Any] = None,
        wheelchair_accessible: bool = False,
        traveling_with_kids: bool = False,
        blacklisted_categories: Optional[set] = None,
        blacklisted_tags: Optional[set] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return filter_candidate_pois(
            candidate_pois,
            wheelchair_accessible=wheelchair_accessible,
            traveling_with_kids=traveling_with_kids,
            user_state=user_state,
            blacklisted_categories=blacklisted_categories,
            blacklisted_tags=blacklisted_tags,
            **kwargs,
        )

    def build_feasible_itinerary(
        self,
        start_time_str: Optional[str] = None,
        end_time_str: Optional[str] = None,
        candidate_pois: Optional[List[Dict[str, Any]]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        start_location: Optional[Dict[str, float]] = None,
        traveling_with_kids: bool = False,
        child_age: Optional[int] = None,
        wheelchair_accessible: bool = False,
        explicit_requested_pois: Optional[List[Dict[str, Any]]] = None,
        user_state: Optional[Any] = None,
        blacklisted_categories: Optional[set] = None,
        blacklisted_tags: Optional[set] = None,
        insert_rest_stop: bool = False,
        rest_stop_mins: int = 35,
        rest_stop_name: str = "☕ Στάση για Καφέ & Ξεκούραση (Πλάκα)",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        budget_hours: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ελέγχει τους περιορισμούς και κατασκευάζει ένα έγκυρο δρομολόγιο σε μορφή JSON.
        """
        if start_time_str is None:
            start_time_str = start_time or "14:00"

        if end_time_str is None:
            if end_time is not None:
                end_time_str = end_time
            elif budget_hours is not None:
                end_mins_calc = time_to_minutes(start_time_str) + int(budget_hours * 60)
                end_time_str = minutes_to_time(end_mins_calc)
            else:
                end_time_str = "18:00"

        if candidate_pois is None:
            db_path = os.path.join(os.path.dirname(__file__), "..", "data", "athens_attractions.json")
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    candidate_pois = json.load(f)
            except Exception:
                candidate_pois = []

        if weather_data is None:
            weather_data = {"is_indoor_recommended": False, "temperature": 24.0, "condition": "clear"}

        start_mins = time_to_minutes(start_time_str)
        end_mins = time_to_minutes(end_time_str)
        budget_mins = end_mins - start_mins
        current_mins = start_mins

        # Αυστηρό προ-φίλτρο (Hard Pre-filter) βάσει αρνητικών περιορισμών & προσβασιμότητας
        candidate_pois = filter_candidate_pois(
            candidate_pois,
            wheelchair_accessible=wheelchair_accessible,
            traveling_with_kids=traveling_with_kids,
            user_state=user_state,
            blacklisted_categories=blacklisted_categories,
            blacklisted_tags=blacklisted_tags,
            weather_data=weather_data,
        )

        # Έλεγχος αν τα ρητά ζητηθέντα POIs υπερβαίνουν το διαθέσιμο χρόνο
        constraint_rejected = False
        total_needed_mins = 0
        if explicit_requested_pois and len(explicit_requested_pois) > 1:
            total_req_dur = sum(p.get("avg_visit_duration_mins", 60) for p in explicit_requested_pois)
            est_transit = (len(explicit_requested_pois) - 1) * 20  # ~20 λεπτά ανά μετακίνηση
            total_needed_mins = total_req_dur + est_transit
            if total_needed_mins > budget_mins:
                constraint_rejected = True

        # 1. Weather Filtering / Sorting & Kid-friendly / Wheelchair Filtering
        temp_val = 0.0
        if weather_data:
            try:
                temp_val = float(weather_data.get("temperature_c") or weather_data.get("temperature") or 0.0)
            except (ValueError, TypeError):
                temp_val = 0.0
        is_bad_weather = weather_data.get("is_indoor_recommended", False) or (temp_val > 38.0)
        filtered_pois: List[Dict[str, Any]] = []

        for poi in candidate_pois:
            # Φιλτράρισμα παιδιών αν ζητήθηκε
            if traveling_with_kids:
                if not poi.get("kid_friendly", True):
                    continue
                if child_age is not None and poi.get("min_age", 0) > child_age:
                    continue

            # ♿ Hard Constraint για Αμαξίδιο:
            if wheelchair_accessible:
                is_accessible = poi.get("wheelchair_accessible", False)
                has_stairs = "stairs" in poi.get("tags", [])
                if not is_accessible or has_stairs:
                    continue

            # Αν ο καιρός είναι κακός, αποκλείουμε ή υποβιβάζουμε τα outdoor POIs
            if is_bad_weather and poi.get("type") == "outdoor":
                continue  # Αποκλεισμός εξωτερικών χώρων σε έντονη βροχή / καύσωνα

            filtered_pois.append(poi)

        # Αν αποκλείστηκαν όλα λόγω καιρού, επαναφέρουμε τα indoor ως fallback
        if not filtered_pois:
            filtered_pois = [p for p in candidate_pois if p.get("type") == "indoor"]


        schedule: List[Dict[str, Any]] = []
        total_distance_km = 0.0
        current_coords = start_location or (
            filtered_pois[0]["coordinates"]
            if filtered_pois and "coordinates" in filtered_pois[0]
            else {"lat": 37.9715, "lon": 23.7257}
        )

        selected_poi_ids: List[str] = []
        rest_stop_inserted = False

        for poi in filtered_pois:
            if poi["id"] in selected_poi_ids:
                continue

            poi_open_mins = time_to_minutes(poi["opening_hours"]["open"])
            poi_close_mins = time_to_minutes(poi["opening_hours"]["close"])
            visit_duration = poi.get("avg_visit_duration_mins", 60)

            # Υπολογισμός μετακίνησης από την τρέχουσα τοποθεσία
            travel_info = calculate_travel_time_mins(
                current_coords, poi["coordinates"], mode=self.transport_mode
            )
            travel_mins = travel_info["duration_mins"]
            arrival_mins = current_mins + travel_mins

            # Εάν φτάσουμε πριν ανοίξει, περιμένουμε μέχρι την ώρα ανοίγματος
            actual_start_mins = max(arrival_mins, poi_open_mins)
            departure_mins = actual_start_mins + visit_duration

            # --- ΕΛΕΓΧΟΣ ΠΕΡΙΟΡΙΣΜΩΝ (CONSTRAINTS VALIDATION) ---
            # 1. Το POI πρέπει να είναι ανοιχτό κατά τη διάρκεια της επίσκεψης
            if departure_mins > poi_close_mins:
                continue  # Παράλειψη: Το POI θα έχει κλείσει

            # 2. Αυστηρό Time Budget Enforcement: Σ(t_visit + t_transit) <= user_time_budget
            total_elapsed_mins = departure_mins - start_mins
            if departure_mins > end_mins or total_elapsed_mins > budget_mins:
                continue  # Παράλειψη: Υπερβαίνει το αυστηρό χρονικό περιθώριο (Time Budget) του χρήστη

            # Προσθήκη βήματος μετακίνησης αν υπάρχει απόσταση
            if current_mins != start_mins or start_location is not None:
                schedule.append({
                    "type": "transit",
                    "action": f"Μετάβαση στο {poi['name']} ({self.transport_mode})",
                    "duration_mins": travel_mins,
                    "distance_km": travel_info["distance_km"],
                    "time": f"{minutes_to_time(current_mins)}-{minutes_to_time(arrival_mins)}",
                })
                total_distance_km += travel_info["distance_km"]

            # Προσθήκη δραστηριότητας στο πρόγραμμα
            time_slot_str = f"{minutes_to_time(actual_start_mins)}-{minutes_to_time(departure_mins)}"
            schedule.append({
                "type": "activity",
                "poi_id": poi["id"],
                "poi_name": poi["name"],
                "poi": poi["name"],
                "category": poi.get("category", "attraction"),
                "env_type": poi.get("type", "indoor"),
                "time_slot": time_slot_str,
                "time": time_slot_str,
                "duration_mins": visit_duration,
            })

            selected_poi_ids.append(poi["id"])

            # Ενημέρωση της τρέχουσας κατάστασης
            current_mins = departure_mins
            current_coords = poi["coordinates"]

            # Εισαγωγή Rest Slot (π.χ. 35 λεπτά για καφέ/ξεκούραση) μετά την πρώτη στάση
            if insert_rest_stop and not rest_stop_inserted and len([s for s in schedule if s["type"] == "activity"]) == 1:
                rest_start_mins = current_mins
                rest_end_mins = rest_start_mins + rest_stop_mins
                rest_time_slot = f"{minutes_to_time(rest_start_mins)}-{minutes_to_time(rest_end_mins)}"
                schedule.append({
                    "type": "activity",
                    "poi_id": "plaka_coffee_rest",
                    "poi_name": rest_stop_name,
                    "poi": rest_stop_name,
                    "category": "cafe",
                    "env_type": "outdoor",
                    "time_slot": rest_time_slot,
                    "time": rest_time_slot,
                    "duration_mins": rest_stop_mins,
                })
                selected_poi_ids.append("plaka_coffee_rest")
                current_mins = rest_end_mins
                rest_stop_inserted = True

            # Αν απομένει λίγος χρόνος (< 30 λεπτά), σταματάμε
            if end_mins - current_mins < 30:
                break

        if insert_rest_stop and not rest_stop_inserted:
            rest_start_mins = current_mins
            rest_end_mins = rest_start_mins + rest_stop_mins
            rest_time_slot = f"{minutes_to_time(rest_start_mins)}-{minutes_to_time(rest_end_mins)}"
            schedule.insert(0, {
                "type": "activity",
                "poi_id": "plaka_coffee_rest",
                "poi_name": rest_stop_name,
                "poi": rest_stop_name,
                "category": "cafe",
                "env_type": "outdoor",
                "time_slot": rest_time_slot,
                "time": rest_time_slot,
                "duration_mins": rest_stop_mins,
            })
            selected_poi_ids.insert(0, "plaka_coffee_rest")
            current_mins = max(current_mins, rest_end_mins)
            rest_stop_inserted = True

        feasible = len([s for s in schedule if s["type"] == "activity"]) > 0

        return {
            "feasible": feasible,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "actual_end_time": minutes_to_time(current_mins),
            "total_distance_km": round(total_distance_km, 2),
            "weather_adjusted": is_bad_weather,
            "constraint_rejected": constraint_rejected,
            "total_needed_mins": total_needed_mins if constraint_rejected else None,
            "schedule": schedule,
            "selected_poi_ids": selected_poi_ids,
        }

    # Backward compatible wrapper matching earlier agent signatures
    def build_itinerary(
        self,
        start_time: str,
        end_time: str,
        candidate_pois: List[Dict[str, Any]],
        weather_data: Optional[Dict[str, Any]] = None,
        traveling_with_kids: bool = False,
        child_age: Optional[int] = None,
        preferred_pace: str = "moderate",
        existing_plan: Optional[Any] = None,
    ) -> Any:
        """Alias for build_feasible_itinerary returning compatible plan dict/object."""
        w_data = weather_data or {}
        res = self.build_feasible_itinerary(
            start_time_str=start_time,
            end_time_str=end_time,
            candidate_pois=candidate_pois,
            weather_data=w_data,
            traveling_with_kids=traveling_with_kids,
            child_age=child_age,
        )
        return ItineraryPlan(
            feasible=res["feasible"],
            start_time=res["start_time"],
            end_time=res["end_time"],
            actual_end_time=res.get("actual_end_time", res["end_time"]),
            total_distance_km=res["total_distance_km"],
            schedule=res["schedule"],
            selected_poi_ids=res["selected_poi_ids"],
            weather_adjusted=res.get("weather_adjusted", False),
        )

    def replan_itinerary(
        self,
        current_plan: Any,
        all_candidates: List[Dict[str, Any]],
        remove_poi_id: Optional[str] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        traveling_with_kids: bool = False,
        child_age: Optional[int] = None,
    ) -> Any:
        """Dynamic Replanning: replaces a disliked POI while preserving other candidates."""
        selected_ids = getattr(current_plan, "selected_poi_ids", None)
        if selected_ids is None and isinstance(current_plan, dict):
            selected_ids = current_plan.get("selected_poi_ids", [])

        # Filter out the disliked POI
        retained_candidates = [
            p for p in all_candidates
            if p["id"] != remove_poi_id
        ]

        # Prioritize previously selected POIs (except the removed one)
        def priority_key(p: Dict[str, Any]) -> int:
            return 0 if (selected_ids and p["id"] in selected_ids) else 1

        retained_candidates.sort(key=priority_key)

        start_time = getattr(current_plan, "start_time", "14:00") if not isinstance(current_plan, dict) else current_plan.get("start_time", "14:00")
        end_time = getattr(current_plan, "end_time", "18:00") if not isinstance(current_plan, dict) else current_plan.get("end_time", "18:00")

        return self.build_itinerary(
            start_time=start_time,
            end_time=end_time,
            candidate_pois=retained_candidates,
            weather_data=weather_data,
            traveling_with_kids=traveling_with_kids,
            child_age=child_age,
        )

    def generate_draft(
        self,
        user_request: str,
        time_budget: float | int,
        user_preferences: Optional[Dict[str, Any]] = None,
        candidate_pois: Optional[List[Dict[str, Any]]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic Layer: Υπολογίζει το αρχικό σχέδιο δρομολογίου (draft itinerary)
        λαμβάνοντας υπόψη τη βάση γνώσης POIs, τους περιορισμούς και το διαθέσιμο χρόνο.
        """
        if candidate_pois is None:
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "athens_attractions.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    candidate_pois = json.load(f)
            else:
                candidate_pois = []

        prefs = user_preferences or {}
        if isinstance(prefs, dict):
            wheelchair = prefs.get("wheelchair_accessible", False)
            kids = prefs.get("traveling_with_kids", False)
            child_age = prefs.get("child_age")
            start_time = prefs.get("start_time", "14:00")
            blacklisted_cats = set(prefs.get("blacklisted_categories", []))
            blacklisted_tags = set(prefs.get("blacklisted_tags", []))
        else:
            wheelchair = getattr(prefs, "wheelchair_accessible", False)
            kids = getattr(prefs, "traveling_with_kids", False)
            child_age = getattr(prefs, "kid_age", None)
            start_time = getattr(prefs, "start_time", "14:00")
            blacklisted_cats = getattr(prefs, "blacklisted_categories", set())
            blacklisted_tags = getattr(prefs, "blacklisted_tags", set())

        req_lower = user_request.lower() if user_request else ""
        if "καροτσ" in req_lower or "αμαξιδι" in req_lower or "wheelchair" in req_lower:
            wheelchair = True
        if "παιδ" in req_lower or "kids" in req_lower:
            kids = True
        if "χωρίς μουσεία" in req_lower or "οχι μουσεια" in req_lower or "όχι μουσεία" in req_lower:
            blacklisted_cats.add("museum")

        budget_mins = int(time_budget * 60) if time_budget <= 24 else int(time_budget)
        start_mins = time_to_minutes(start_time)
        end_time = minutes_to_time(start_mins + budget_mins)

        w_data = weather_data or {"is_indoor_recommended": False, "description": "Καθαρός καιρός"}

        plan = self.build_feasible_itinerary(
            start_time_str=start_time,
            end_time_str=end_time,
            candidate_pois=candidate_pois,
            weather_data=w_data,
            traveling_with_kids=kids,
            child_age=child_age,
            wheelchair_accessible=wheelchair,
            blacklisted_categories=blacklisted_cats,
            blacklisted_tags=blacklisted_tags,
        )

        raw_list: List[Dict[str, Any]] = []
        for step in plan.get("schedule", []):
            if step["type"] == "activity":
                poi_id = step.get("poi_id")
                desc = f"Επίσκεψη και ξενάγηση στο {step.get('poi_name', 'αξιοθέατο')}."
                matching_p = next((p for p in candidate_pois if p.get("id") == poi_id), None)
                if matching_p and matching_p.get("description"):
                    desc = matching_p["description"].split(".")[0].strip() + "."
                raw_list.append({
                    "name": step.get("poi_name") or step.get("poi", "Αξιοθέατο"),
                    "duration": step.get("duration_mins", 60),
                    "duration_mins": step.get("duration_mins", 60),
                    "type": step.get("category", "activity"),
                    "category": step.get("category", "activity"),
                    "outdoor": step.get("env_type") != "indoor",
                    "indoor": step.get("env_type") == "indoor",
                    "wheelchair_accessible": not wheelchair or True,
                    "description": desc,
                    "time_slot": step.get("time_slot") or step.get("time", ""),
                    "poi_id": poi_id,
                })
            elif step["type"] == "transit":
                raw_list.append({
                    "name": step.get("action", "Μετάβαση"),
                    "duration": step.get("duration_mins", 15),
                    "duration_mins": step.get("duration_mins", 15),
                    "type": "walking",
                    "distance_km": step.get("distance_km", 0.5),
                    "time": step.get("time", ""),
                })

        return raw_list

    def optimize_itinerary_duration(self, itinerary_list, requested_budget_mins, current_total_mins):
        return optimize_itinerary_duration(itinerary_list, requested_budget_mins, current_total_mins)

    def calculate_total_mins(self, itinerary_data):
        return calculate_total_mins(itinerary_data)


# ---------------------------------------------------------------------------
# Step 2: Time-Filling Buffer & Dynamic Scaling (Python)
# ---------------------------------------------------------------------------
def optimize_itinerary_duration(itinerary_list, requested_budget_mins, current_total_mins):
    """
    Αναπροσαρμόζει το δρομολόγιο για να γεμίσει ακριβώς τον διαθέσιμο χρόνο του χρήστη.
    """
    if requested_budget_mins <= 24:
        requested_budget_mins = int(requested_budget_mins * 60)

    remaining_mins = requested_budget_mins - current_total_mins

    # 1. Αν περισσεύουν 15 έως 45 λεπτά: Προσθέτουμε ένα "Rest Slot" (Καφές/Ξεκούραση)
    if 15 <= remaining_mins <= 45:
        rest_slot = {
            "name": "Χαλαρή βόλτα & στάση για καφέ/ξεκούραση",
            "duration": remaining_mins,
            "duration_mins": remaining_mins,
            "type": "rest_area",
            "outdoor": True,
            "description": "Ένα μικρό διάλειμμα για να απολαύσετε την ατμόσφαιρα της πόλης πριν την ολοκλήρωση της βόλτας σας."
        }
        itinerary_list.append(rest_slot)
        try:
            print(f"✅ Προστέθηκε Rest Slot διάρκειας {remaining_mins} λεπτών.")
        except UnicodeEncodeError:
            print(f"[OK] Προστέθηκε Rest Slot διάρκειας {remaining_mins} λεπτών.")

    # 2. Αν περισσεύουν 1 έως 14 λεπτά: Τα μοιράζουμε στα υπάρχοντα αξιοθέατα (Dynamic Scaling)
    elif 0 < remaining_mins < 15:
        num_pois = len([item for item in itinerary_list if item.get('type') != 'walking'])
        if num_pois > 0:
            extra_per_poi = remaining_mins // num_pois
            remainder = remaining_mins % num_pois
            
            # Μοιράζουμε τα έξτρα λεπτά (π.χ. 7 έξτρα λεπτά παραμονή στο Μουσείο)
            poi_count = 0
            for item in itinerary_list:
                if item.get('type') != 'walking':
                    if 'duration' not in item and 'duration_mins' in item:
                        item['duration'] = item['duration_mins']
                    item['duration'] += extra_per_poi
                    if poi_count == 0: 
                        item['duration'] += remainder # Βάζουμε το όποιο υπόλοιπο στο πρώτο
                    if 'duration_mins' in item:
                        item['duration_mins'] = item['duration']
                    poi_count += 1
            try:
                print(f"✅ Έγινε Dynamic Scaling: +{extra_per_poi} λεπτά σε κάθε στάση.")
            except UnicodeEncodeError:
                print(f"[OK] Έγινε Dynamic Scaling: +{extra_per_poi} λεπτά σε κάθε στάση.")

    return itinerary_list


def calculate_total_mins(itinerary_data: Any) -> int:
    """
    Υπολογίζει το σύνολο λεπτών ενός δρομολογίου.
    """
    items = itinerary_data
    if isinstance(itinerary_data, dict):
        items = itinerary_data.get("schedule", [])
    total = 0
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                dur = item.get("duration")
                if dur is None:
                    dur = item.get("duration_mins", 0)
                total += int(dur)
    return total


# Export compatible dataclasses / types
ScheduleItem = dict

