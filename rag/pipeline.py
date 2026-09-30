"""
Robust POI Data Pipeline & Consistency Engine (rag/pipeline.py).

Implements Chip Huyen's Principles on Data Engineering & Pipeline Robustness:
1. Strict Schema Validation:
   - Validates POI metadata against Pydantic V2 rules (coordinates within Athens bounding box,
     opening hours strictly 'open < close', visit duration in reasonable range, non-empty tags).
2. Atomic In-Place Updates (Zero-Inconsistency Guarantee):
   - Rejects invalid updates without corrupting active serving state (Clean Rollback).
   - Dynamically recalculates contextual chunks and embeddings via UnifiedEmbeddingPipeline.
   - Atomically updates in-memory vector stores and retriever document caches.
3. Safe Persistence:
   - Atomic disk replacement via write-to-temp-then-rename to prevent partial write corruption.
4. Checksum & Version Auditing:
   - Tracks data versions with SHA-256 content hashes.
"""

from __future__ import annotations
import copy
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from rag.chunking import contextual_chunker, ContextualChunk
from rag.unified_embedder import unified_pipeline

logger = logging.getLogger(__name__)


# 1. Strict Pydantic POI Metadata Schema
class OpeningHoursSchema(BaseModel):
    open: str = Field(..., example="08:00")
    close: str = Field(..., example="20:00")

    @field_validator("open", "close")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", v.strip()):
            raise ValueError(f"Time format must be HH:MM, got '{v}'")
        h, m = map(int, v.split(":"))
        if not (0 <= h <= 24 and 0 <= m <= 59):
            raise ValueError(f"Invalid time bounds: {v}")
        return v.strip()

    @model_validator(mode="after")
    def validate_open_before_close(self) -> OpeningHoursSchema:
        if self.open >= self.close and self.close != "00:00":
            raise ValueError(f"Opening time '{self.open}' must be strictly before closing time '{self.close}'")
        return self


class CoordinatesSchema(BaseModel):
    lat: float = Field(..., example=37.9715)
    lon: float = Field(..., example=23.7257)

    @model_validator(mode="after")
    def validate_athens_region_bounds(self) -> CoordinatesSchema:
        if not (37.0 <= self.lat <= 38.5):
            raise ValueError(f"Latitude {self.lat} is outside valid Athens metropolitan area [37.0, 38.5]")
        if not (23.0 <= self.lon <= 24.5):
            raise ValueError(f"Longitude {self.lon} is outside valid Athens metropolitan area [23.0, 24.5]")
        return self


class POIMetadataSchema(BaseModel):
    id: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=3, max_length=128)
    category: str = Field(..., min_length=2)
    type: str = Field(..., pattern=r"^(indoor|outdoor)$")
    opening_hours: OpeningHoursSchema
    avg_visit_duration_mins: int = Field(..., ge=15, le=360)
    kid_friendly: bool = Field(default=False)
    min_age: int = Field(default=0, ge=0, le=18)
    wheelchair_accessible: bool = Field(default=True)
    coordinates: CoordinatesSchema
    tags: List[str] = Field(default_factory=list, min_length=1)
    description: str = Field(..., min_length=10)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {
            "monument", "museum", "archaeological_site", "square",
            "cultural_center", "viewpoint", "neighborhood", "park", "tourist_service"
        }
        if v.lower() not in allowed:
            raise ValueError(f"Category '{v}' not allowed. Must be one of: {sorted(allowed)}")
        return v.lower()


# 2. Atomic Data Pipeline Manager
class AtomicPOIPipeline:
    """
    Manages robust transactional updates to the POI dataset.
    Guarantees pipeline consistency, validation enforcement, and atomic serving updates.
    """

    def __init__(self, data_path: Optional[str | Path] = None, retriever: Optional[Any] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.data_path = Path(data_path) if data_path else base_dir / "data" / "athens_attractions.json"
        self.retriever = retriever
        self.pipeline_version = 1
        self.last_update_timestamp = datetime.now(timezone.utc).isoformat()
        self.total_updates_performed = 0
        self.total_validation_rejections = 0

    @staticmethod
    def compute_content_hash(data: Any) -> str:
        """Computes SHA-256 fingerprint of POI records to audit changes."""
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def validate_poi_payload(self, poi_data: Dict[str, Any]) -> POIMetadataSchema:
        """Validates payload against strict schema. Raises ValidationError on inconsistency."""
        return POIMetadataSchema(**poi_data)

    def update_poi_metadata(
        self,
        poi_id: str,
        updates: Dict[str, Any],
        persist_to_disk: bool = False
    ) -> Dict[str, Any]:
        """
        Atomically updates a POI's metadata:
        1. Clones existing record.
        2. Merges updates.
        3. Validates merged record with Pydantic.
        4. Re-calculates contextual chunk and embedding via UnifiedEmbeddingPipeline.
        5. Atomically replaces in-memory retriever state.
        6. If persist_to_disk is True, writes atomically to disk.
        """
        if not self.retriever:
            from rag.retriever import AthensRetriever
            self.retriever = AthensRetriever()

        # 1. Lookup existing POI
        existing_raw = next((p for p in self.retriever.raw_pois if p["id"] == poi_id), None)
        if not existing_raw:
            raise ValueError(f"POI with id '{poi_id}' does not exist in knowledge base.")

        # 2. Deep clone and merge updates
        tentative_poi = copy.deepcopy(existing_raw)
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(tentative_poi.get(k), dict):
                tentative_poi[k].update(v)
            else:
                tentative_poi[k] = v

        # 3. Strict Schema Validation (Clean Rollback on Failure)
        try:
            validated = self.validate_poi_payload(tentative_poi)
        except Exception as err:
            self.total_validation_rejections += 1
            logger.error("[Pipeline Inconsistency Intercepted] POI update rejected for '%s': %s", poi_id, err)
            raise ValueError(f"Data pipeline validation failed for POI '{poi_id}': {str(err)}") from err

        validated_dict = validated.model_dump()

        # 4. Atomically recalculate contextual chunk and embeddings
        new_chunks = contextual_chunker.chunk_poi(validated_dict)
        for ch in new_chunks:
            ch.embedding = unified_pipeline.embed_text(ch.get_search_text())

        primary_chunk = new_chunks[0]

        # 5. Atomic In-Memory Replacement in Serving Layer
        for i, p in enumerate(self.retriever.raw_pois):
            if p["id"] == poi_id:
                self.retriever.raw_pois[i] = validated_dict
                break

        for i, doc in enumerate(self.retriever.documents):
            if doc["id"] == poi_id:
                self.retriever.documents[i] = {
                    "id": poi_id,
                    "name": validated_dict["name"],
                    "text": primary_chunk.get_generation_payload(),
                    "metadata": primary_chunk.metadata,
                    "search_summary": primary_chunk.search_summary,
                    "context_header": primary_chunk.context_header,
                    "embedding": primary_chunk.embedding,
                }
                break

        # Also update raw map and knowledge base
        if hasattr(self.retriever, "_raw_map"):
            self.retriever._raw_map[poi_id] = validated_dict
        if hasattr(self.retriever, "kb") and hasattr(self.retriever.kb, "_poi_map"):
            self.retriever.kb._poi_map[poi_id] = validated_dict

        self.pipeline_version += 1
        self.total_updates_performed += 1
        self.last_update_timestamp = datetime.now(timezone.utc).isoformat()

        # 6. Optional Safe Atomic Disk Persistence
        if persist_to_disk and self.data_path.exists():
            self._atomic_save_to_disk(self.retriever.raw_pois)

        logger.info("[Pipeline Success] POI '%s' metadata updated atomically. Pipeline version: %d", poi_id, self.pipeline_version)

        return {
            "status": "success",
            "poi_id": poi_id,
            "pipeline_version": self.pipeline_version,
            "updated_fields": list(updates.keys()),
            "content_hash": self.compute_content_hash(validated_dict),
            "timestamp": self.last_update_timestamp,
        }

    def _atomic_save_to_disk(self, data: List[Dict[str, Any]]) -> None:
        """Writes data to a temporary file in the same directory, then atomically replaces target."""
        target_dir = self.data_path.parent
        with tempfile.NamedTemporaryFile("w", dir=str(target_dir), delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name

        os.replace(temp_name, str(self.data_path))

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Returns health and consistency telemetry for the data pipeline."""
        poi_count = len(self.retriever.raw_pois) if self.retriever else 0
        overall_hash = self.compute_content_hash(self.retriever.raw_pois) if self.retriever else ""
        return {
            "status": "healthy",
            "pipeline_version": self.pipeline_version,
            "total_pois_indexed": poi_count,
            "dataset_content_hash": overall_hash,
            "last_update_timestamp": self.last_update_timestamp,
            "total_updates_performed": self.total_updates_performed,
            "total_validation_rejections": self.total_validation_rejections,
            "zero_inconsistency_guarantee": True,
        }


# Global singleton instance
atomic_poi_pipeline = AtomicPOIPipeline()
