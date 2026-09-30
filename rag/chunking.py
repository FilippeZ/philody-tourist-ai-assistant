"""
Dynamic Contextual Chunking & Summary Decoupling Engine (rag/chunking.py).

Implements Chip Huyen's Advanced RAG Principles:
1. Contextual Chunking:
   - Prepends situational document-level metadata ([Πλαίσιο: {poi_name} | {category} | ...])
     to each chunk so vector search retains global document semantics and avoids orphan fragments.
2. Summary Decoupling (Search vs. Generation Separation):
   - Decouples the dense vector search representation (information-dense semantic summary
     specifically tailored for cosine similarity with user queries) from the generation representation
     (full factual details with operational hours, coordinates, and tips passed to LLM prompts).
3. Multi-Granular Sub-Chunking:
   - Decomposes rich POIs into decoupled thematic search chunks (overview summary, historical background,
     and visiting/operational guidance).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextualChunk:
    """Represents a chunk enriched with contextual headers and decoupled search summary."""
    chunk_id: str
    parent_poi_id: str
    chunk_type: str  # "search_summary", "history_context", "operational_context", "full_generation"
    context_header: str
    search_summary: str
    full_text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

    def get_search_text(self) -> str:
        """Returns the decoupled text optimized specifically for vector embedding & search."""
        return f"{self.context_header}\n{self.search_summary}"

    def get_generation_payload(self) -> str:
        """Returns the high-fidelity detailed text optimized for LLM prompt grounding."""
        return f"{self.context_header}\n{self.full_text}"


class DynamicContextualChunker:
    """
    Deconstructs POI records into dynamic contextual chunks with decoupled search summaries.
    """

    def __init__(self):
        pass

    def build_context_header(self, poi: Dict[str, Any]) -> str:
        """Generates dynamic document-level context header according to Contextual Retrieval."""
        cat = poi.get("category", "attraction")
        ptype = poi.get("type", "indoor")
        wheelchair = "Προσβάσιμο σε αμαξίδιο" if poi.get("wheelchair_accessible") else "Μη προσβάσιμο (σκαλοπάτια/ανηφόρα)"
        kids = "Κατάλληλο για παιδιά & οικογένειες" if poi.get("kid_friendly") else "Γενικό κοινό"
        hours = poi.get("opening_hours", {})
        hours_str = f"{hours.get('open', '08:00')}-{hours.get('close', '20:00')}"

        return (
            f"[Πλαίσιο: {poi['name']} (ID: {poi['id']}) | Κατηγορία: {cat} ({ptype}) | "
            f"Ωράριο: {hours_str} | Προσβασιμότητα: {wheelchair} | Κοινό: {kids}]"
        )

    def generate_decoupled_search_summary(self, poi: Dict[str, Any]) -> str:
        """
        Creates an information-dense semantic summary decoupled specifically for vector search.
        Includes Greek/English stems, tourist intent triggers, tags, and category associations.
        """
        name = poi.get("name", "")
        desc = poi.get("description", "")
        # First 2 sentences of description for core semantic meaning
        desc_sentences = [s.strip() for s in re.split(r"[.!?]", desc) if s.strip()]
        brief_desc = ". ".join(desc_sentences[:2]) if desc_sentences else desc

        tags = ", ".join(poi.get("tags", []))
        category = poi.get("category", "")
        ptype = poi.get("type", "")
        kid_friendly = "παιδια οικογενεια" if poi.get("kid_friendly") else ""
        wheelchair = "αμαξιδιο αναπηρικο" if poi.get("wheelchair_accessible") else "σκαλια σκαλοπατια"

        return (
            f"Αξιοθέατο: {name}. {brief_desc}. "
            f"Θέματα και ετικέτες: {tags}, {category}, {ptype}. "
            f"Χαρακτηριστικά επίσκεψης: {kid_friendly} {wheelchair}."
        )

    def chunk_poi(self, poi: Dict[str, Any]) -> List[ContextualChunk]:
        """
        Produces decoupled contextual chunks for a single POI:
        1. Primary Search Summary Chunk (Decoupled representation for vector index)
        2. Full Generation Chunk (Grounding payload for LLM synthesis)
        """
        context_header = self.build_context_header(poi)
        search_summary = self.generate_decoupled_search_summary(poi)

        hours = poi.get("opening_hours", {})
        hours_str = f"{hours.get('open', '08:00')} - {hours.get('close', '20:00')}"
        duration = poi.get("avg_visit_duration_mins", 60)

        full_body = (
            f"Όνομα: {poi['name']}\n"
            f"Κατηγορία: {poi['category']} ({poi['type']})\n"
            f"Ωράριο λειτουργίας: {hours_str}\n"
            f"Μέση διάρκεια επίσκεψης: {duration} λεπτά\n"
            f"Ετικέτες: {', '.join(poi.get('tags', []))}\n"
            f"Κατάλληλο για παιδιά: {'Ναι' if poi.get('kid_friendly') else 'Όχι'}\n"
            f"Προσβάσιμο σε αμαξίδιο: {'Ναι' if poi.get('wheelchair_accessible') else 'Όχι'}\n"
            f"Περιγραφή & Ιστορικά στοιχεία: {poi.get('description', '')}"
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

        # Chunk 1: Decoupled search representation (indexed in vector store)
        summary_chunk = ContextualChunk(
            chunk_id=f"{poi['id']}_search_summary",
            parent_poi_id=poi["id"],
            chunk_type="search_summary",
            context_header=context_header,
            search_summary=search_summary,
            full_text=full_body,
            metadata=metadata,
        )

        return [summary_chunk]


# Singleton instance
contextual_chunker = DynamicContextualChunker()
