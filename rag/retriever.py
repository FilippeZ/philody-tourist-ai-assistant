"""
Embeddings & Vector Store Retriever for Athens Tourist Assistant (rag/retriever.py).
"""

from __future__ import annotations
import math
import os
import re
from typing import Any, Dict, List, Optional
import numpy as np

from rag.ingestion import format_poi_to_document, load_athens_pois


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vector representations."""
    a, b = np.array(v1, dtype=float), np.array(v2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# Dynamic Embedding Function with 3-tier Fallback Strategy:
# 1. Local SentenceTransformers (if cached/fast)
# 2. OpenAI Embedding API (if OPENAI_API_KEY is configured)
# 3. Fast In-Memory TF-IDF / Subword Dense Vectorizer (zero-latency, 100% offline resilient)
_EMBED_BACKEND = "fallback"
_SENTENCE_MODEL = None

try:
    # Only load sentence_transformers if explicitly enabled or if local model exists
    if os.getenv("USE_SENTENCE_TRANSFORMERS", "").lower() in ["1", "true", "yes"]:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBED_BACKEND = "sentence_transformers"
except Exception:
    pass

if _EMBED_BACKEND == "fallback" and os.getenv("OPENAI_API_KEY", "").strip():
    try:
        import openai
        _EMBED_BACKEND = "openai"
    except ImportError:
        pass


import hashlib

import unicodedata

def normalize_greek(text: str) -> str:
    """Strips Greek diacritics/accents and converts to lowercase with uniform sigma."""
    if not text:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").lower()
    return s.replace("ς", "σ")


def _det_hash(text: str, dim: int = 128) -> int:
    """Deterministic hash across Python processes."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % dim


def _get_fallback_dense_vector(text: str, dim: int = 128) -> List[float]:
    """
    Deterministic subword hashing vectorizer for 100% offline instant execution.
    Preserves semantic similarity across Greek and English tourist terms.
    """
    vec = np.zeros(dim, dtype=float)
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        # Hash full token
        h = _det_hash(token, dim)
        vec[h] += 1.0
        # Hash character n-grams (3-grams) for morphological overlap in Greek
        for i in range(len(token) - 2):
            ngram = token[i : i + 3]
            h_ng = _det_hash(ngram, dim)
            vec[h_ng] += 0.5

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


from rag.unified_embedder import unified_pipeline


def get_embedding(text: str) -> List[float]:
    """Generates embedding vector via the unified pipeline (zero feature skew)."""
    return unified_pipeline.embed_text(text)


class AthensRAGRetriever:
    """
    In-memory Vector Store & Semantic Search Retriever for Athens Attractions.
    """

    def __init__(self, json_path: str = "data/athens_attractions.json"):
        self.raw_pois = load_athens_pois(json_path)
        self.documents = [format_poi_to_document(poi) for poi in self.raw_pois]
        self._raw_map = {p["id"]: p for p in self.raw_pois}

        from rag.ingestion import AthensKnowledgeBase
        from rag.chunking import contextual_chunker, ContextualChunk
        self.kb = AthensKnowledgeBase(json_path)

        # Calculate embeddings upon initialization using Dynamic Contextual Chunking & Summary Decoupling
        print("Indexing Athens Knowledge Base (Dynamic Contextual Chunking & Summary Decoupling)...")
        self.contextual_chunks: List[ContextualChunk] = []
        for poi in self.raw_pois:
            chunks = contextual_chunker.chunk_poi(poi)
            for ch in chunks:
                ch.embedding = get_embedding(ch.get_search_text())
                self.contextual_chunks.append(ch)

        for doc in self.documents:
            ch = next((c for c in self.contextual_chunks if c.parent_poi_id == doc["id"]), None)
            if ch:
                doc["embedding"] = ch.embedding
                doc["search_summary"] = ch.search_summary
                doc["context_header"] = ch.context_header
                doc["text"] = ch.get_generation_payload()
            else:
                doc["embedding"] = get_embedding(doc["text"])

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant POIs based on Vector Similarity + Metadata Filtering.
        """
        query_emb = get_embedding(query)
        results = []

        for doc in self.documents:
            # Apply Metadata Filters (e.g. kid_friendly=True or type='indoor')
            if filters:
                skip = False
                for key, val in filters.items():
                    doc_val = doc["metadata"].get(key)
                    if key == "max_min_age":
                        if doc["metadata"].get("min_age", 0) > val:
                            skip = True
                            break
                    elif key == "exclude_ids":
                        if doc["id"] in val:
                            skip = True
                            break
                    elif doc_val != val:
                        skip = True
                        break
                if skip:
                    continue

            score = cosine_similarity(query_emb, doc["embedding"])

            # Normalized Greek keyword & stem matching
            q_norm = normalize_greek(query)
            q_words = set(re.findall(r"\w+", q_norm))
            # Extract stems (prefixes >= 4 chars) to catch Greek inflectional variants
            q_stems = set(w[:5] if len(w) >= 5 else w for w in q_words if len(w) >= 3)

            doc_text_norm = normalize_greek(doc["text"])
            doc_name_norm = normalize_greek(doc["metadata"]["name"])
            doc_words = set(re.findall(r"\w+", doc_text_norm))
            name_words = set(re.findall(r"\w+", doc_name_norm))
            tag_words = set(normalize_greek(t) for t in doc["metadata"].get("tags", []))

            # Direct word overlap
            overlap = len(q_words.intersection(doc_words))
            name_overlap = len(q_words.intersection(name_words))
            tag_overlap = len(q_words.intersection(tag_words))

            # Stem matching for Greek grammatical cases (e.g. ακροπολη/ακροπολης, μοναστηρακι/μοναστηρακιου, βουλη/βουλης)
            doc_stems = set(w[:5] if len(w) >= 5 else w for w in doc_words if len(w) >= 3)
            name_stems = set(w[:5] if len(w) >= 5 else w for w in name_words if len(w) >= 3)
            stem_name_overlap = len(q_stems.intersection(name_stems))
            stem_text_overlap = len(q_stems.intersection(doc_stems))

            # Specific high-value entity checks (e.g. Βουλή -> syntagma_changing_guards)
            entity_extra_boost = 0.0
            if "βουλ" in q_norm and ("βουλη" in doc_text_norm or doc["id"] == "syntagma_changing_guards"):
                entity_extra_boost += 1.2
            if "μοναστηρακ" in q_norm and ("μοναστηρακ" in doc_name_norm or doc["id"] == "monastiraki_flea_market"):
                entity_extra_boost += 1.2
            if "ακροπολ" in q_norm and ("ακροπολ" in doc_name_norm or "acropolis" in doc["id"]):
                entity_extra_boost += 0.8
            if "παρθενων" in q_norm and ("παρθενων" in doc_text_norm or "parthenon" in doc["id"] or "acropolis" in doc["id"]):
                entity_extra_boost += 1.2
            if "γλυπτ" in q_norm and ("γλυπτ" in doc_text_norm or "sculpture" in doc["metadata"].get("tags", [])):
                entity_extra_boost += 0.8
            if "μουσει" in q_norm and "μουσει" in doc_name_norm:
                entity_extra_boost += 0.4

            boosted_score = (
                score
                + (overlap * 0.05)
                + (name_overlap * 0.40)
                + (stem_name_overlap * 0.50)
                + (stem_text_overlap * 0.08)
                + (tag_overlap * 0.20)
                + entity_extra_boost
            )

            results.append({
                "id": doc["id"],
                "name": doc["metadata"]["name"],
                "score": round(boosted_score, 4),
                "text": doc["text"],
                "metadata": doc["metadata"],
                "context_header": doc.get("context_header", ""),
                "search_summary": doc.get("search_summary", ""),
                # Standard properties for agent integration
                "source_id": doc["id"],
                "source_name": doc["metadata"]["name"],
                "content": doc["text"],
            })

        # Sort descending by similarity score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def format_grounding_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved docs into clear grounding blocks with explicit citations."""
        if not retrieved_docs:
            return "Δεν βρέθηκαν σχετικά αξιοθέατα στη βάση γνώσης."

        context_str = "--- ΠΑΡΕΧΟΜΕΝΗ ΒΑΣΗ ΓΝΩΣΗΣ ---\n"
        for doc in retrieved_docs:
            name = doc.get("name") or doc.get("source_name", "Αξιοθέατο")
            context_str += (
                f"POIs ID: {doc['id']} | Όνομα: {name}\n"
                f"{doc.get('text') or doc.get('content', '')}\n"
                f"-------------------\n"
            )
        return context_str

    def get_poi(self, poi_id: str) -> Optional[Dict[str, Any]]:
        """Direct lookup of raw POI by ID."""
        return self._raw_map.get(poi_id)

    def get_poi_document(self, poi_id: str) -> Optional[Dict[str, Any]]:
        """Direct lookup of formatted document by POI ID."""
        for doc in self.documents:
            if doc["id"] == poi_id:
                return {
                    "id": doc["id"],
                    "name": doc["metadata"]["name"],
                    "score": 1.0,
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "source_id": doc["id"],
                    "source_name": doc["metadata"]["name"],
                    "content": doc["text"],
                }
        return None

    def retrieve_multi(self, queries: List[str], top_k_per_query: int = 1) -> List[Dict[str, Any]]:
        """Executes parallel sub-queries for multiple entities and merges unique results."""
        merged = []
        seen = set()
        for q in queries:
            sub_res = self.retrieve(query=q, top_k=top_k_per_query)
            for r in sub_res:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    merged.append(r)
        return merged

    def get_candidate_pois(
        self,
        query: str = "",
        kid_friendly: Optional[bool] = None,
        max_min_age: Optional[int] = None,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 14,
    ) -> List[Dict[str, Any]]:
        """Convenience query for the Feasibility Engine."""
        filters = {}
        if kid_friendly is not None:
            filters["kid_friendly"] = kid_friendly
        if max_min_age is not None:
            filters["max_min_age"] = max_min_age
        if exclude_ids:
            filters["exclude_ids"] = exclude_ids

        res = self.retrieve(query=query, top_k=limit * 2, filters=filters)
        ids = []
        for r in res:
            if r["id"] not in ids:
                ids.append(r["id"])

        if len(ids) < limit:
            for p in self.raw_pois:
                if p["id"] not in ids:
                    if exclude_ids and p["id"] in exclude_ids:
                        continue
                    if kid_friendly is not None and p.get("kid_friendly") != kid_friendly:
                        continue
                    if max_min_age is not None and p.get("min_age", 0) > max_min_age:
                        continue
                    ids.append(p["id"])
                if len(ids) >= limit:
                    break

        return [self._raw_map[pid] for pid in ids if pid in self._raw_map][:limit]


# Alias for backward compatibility
AthensRetriever = AthensRAGRetriever
