"""
Document Ingestion & Document Construction for Athens Tourism Knowledge Base.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_athens_pois(json_path: str = "data/athens_attractions.json") -> List[Dict[str, Any]]:
    """Φορτώνει τη βάση δεδομένων των POIs."""
    p = Path(json_path)
    if not p.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent
        p = base_dir / json_path

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def format_poi_to_document(poi: Dict[str, Any]) -> Dict[str, Any]:
    """
    Μετατρέπει ένα POI σε κείμενο κατάλληλο για Embedding,
    διατηρώντας παράλληλα τα metadata.
    """
    hours = poi.get("opening_hours", {})
    hours_str = f"{hours.get('open', '08:00')} - {hours.get('close', '20:00')}" if hours else "Δεν ορίζεται"
    text_content = (
        f"Όνομα: {poi['name']}\n"
        f"Κατηγορία: {poi['category']} ({poi['type']})\n"
        f"Ωράριο λειτουργίας: {hours_str}\n"
        f"Μέση διάρκεια επίσκεψης: {poi.get('avg_visit_duration_mins', 60)} λεπτά\n"
        f"Ετικέτες: {', '.join(poi.get('tags', []))}\n"
        f"Κατάλληλο για παιδιά: {'Ναι' if poi.get('kid_friendly') else 'Όχι'}\n"
        f"Προσβάσιμο σε αμαξίδιο: {'Ναι' if poi.get('wheelchair_accessible') else 'Όχι'}\n"
        f"Περιγραφή: {poi.get('description', '')}"
    )

    metadata = {
        "id": poi["id"],
        "name": poi["name"],
        "category": poi["category"],
        "type": poi["type"],
        "kid_friendly": poi.get("kid_friendly", False),
        "min_age": poi.get("min_age", 0),
        "wheelchair_accessible": poi.get("wheelchair_accessible", True),
        "opening_hours": poi.get("opening_hours", {}),
        "avg_visit_duration_mins": poi.get("avg_visit_duration_mins", 60),
        "coordinates": poi.get("coordinates", {}),
        "tags": poi.get("tags", []),
        "description": poi.get("description", ""),
    }

    return {
        "id": poi["id"],
        "text": text_content,
        "metadata": metadata,
    }


# Backwards compatible alias functions and classes
def load_attractions(file_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Alias for load_athens_pois."""
    path_str = str(file_path) if file_path else "data/athens_attractions.json"
    return load_athens_pois(path_str)


class AthensKnowledgeBase:
    """Class wrapper managing the POI documents and fast lookups."""

    def __init__(self, data_path: Optional[str | Path] = None):
        path_str = str(data_path) if data_path else "data/athens_attractions.json"
        self.raw_data: List[Dict[str, Any]] = load_athens_pois(path_str)
        self.documents: List[Dict[str, Any]] = [
            format_poi_to_document(poi) for poi in self.raw_data
        ]
        self._poi_map: Dict[str, Dict[str, Any]] = {
            poi["id"]: poi for poi in self.raw_data
        }

    def get_all_pois(self) -> List[Dict[str, Any]]:
        return list(self._poi_map.values())

    def get_poi_by_id(self, poi_id: str) -> Optional[Dict[str, Any]]:
        return self._poi_map.get(poi_id)
