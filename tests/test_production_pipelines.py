"""
Production Data Pipelines, Zero-Skew & Observability Verification Suite (test_production_pipelines.py).

Validates:
1. 🔧 Pipeline Inconsistency Prevention on POI Metadata Updates (Strict validation & atomic rollback)
2. 🔄 Unified Batch & Streaming Embedding Pipeline (Zero Feature / Train-Serving Skew)
3. ➕ Operational Metrics & Observability (Prometheus / Grafana, HTTP 5xx, p99 Latency, CPU/GPU Telemetry)
"""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import sys
from pathlib import Path


import io
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows safely
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rag.pipeline import atomic_poi_pipeline, POIMetadataSchema, OpeningHoursSchema, CoordinatesSchema
from rag.unified_embedder import unified_pipeline
from rag.retriever import AthensRetriever
from monitoring.observability import observability_registry, LatencyQuantiles
from fastapi.testclient import TestClient
from api import app


def test_data_pipeline_inconsistency_prevention():
    print("=" * 80)
    print("🧪 1. DATA PIPELINE INCONSISTENCY PREVENTION & ATOMIC METADATA UPDATES")
    print("=" * 80)

    retriever = AthensRetriever()
    atomic_poi_pipeline.retriever = retriever

    # 1.1 Valid POI Metadata Update
    poi_id = "acropolis_museum"
    original_poi = next(p for p in retriever.raw_pois if p["id"] == poi_id)
    original_duration = original_poi.get("avg_visit_duration_mins", 90)

    print(f"Original duration for {poi_id}: {original_duration} mins")
    update_res = atomic_poi_pipeline.update_poi_metadata(
        poi_id=poi_id,
        updates={"avg_visit_duration_mins": 105, "opening_hours": {"open": "08:30", "close": "20:30"}},
        persist_to_disk=False
    )
    print(f"Valid Update Success: {update_res['status']}, Content Hash: {update_res['content_hash']}")
    assert update_res["status"] == "success"
    assert update_res["pipeline_version"] >= 2

    # Verify atomic update in raw_pois and vector document text
    updated_raw = next(p for p in retriever.raw_pois if p["id"] == poi_id)
    assert updated_raw["avg_visit_duration_mins"] == 105
    assert updated_raw["opening_hours"]["open"] == "08:30"

    updated_doc = next(d for d in retriever.documents if d["id"] == poi_id)
    assert "105 λεπτά" in updated_doc["text"]
    assert "08:30" in updated_doc["text"]

    # 1.2 Invalid Update: Closing hour before opening hour (Must Fail & Rollback)
    print("\nTesting Invalid Update: Closing hour before opening hour ('20:00' - '08:00')...")
    invalid_caught = False
    try:
        atomic_poi_pipeline.update_poi_metadata(
            poi_id=poi_id,
            updates={"opening_hours": {"open": "20:00", "close": "08:00"}}
        )
    except ValueError as e:
        invalid_caught = True
        print(f"  ✓ Correctly rejected invalid hours: {e}")

    assert invalid_caught is True, "Pipeline failed to reject invalid opening hours"

    # 1.3 Invalid Update: Coordinates outside Athens Metropolitan Area (Must Fail & Rollback)
    print("\nTesting Invalid Update: Coordinates in Paris (lat: 48.85, lon: 2.35)...")
    invalid_coords_caught = False
    try:
        atomic_poi_pipeline.update_poi_metadata(
            poi_id=poi_id,
            updates={"coordinates": {"lat": 48.8566, "lon": 2.3522}}
        )
    except ValueError as e:
        invalid_coords_caught = True
        print(f"  ✓ Correctly rejected out-of-bounds coordinates: {e}")

    assert invalid_coords_caught is True, "Pipeline failed to reject out-of-bounds coordinates"

    # Verify that the active POI state remains completely clean and uncorrupted!
    current_raw = next(p for p in retriever.raw_pois if p["id"] == poi_id)
    assert current_raw["opening_hours"]["open"] == "08:30"
    assert current_raw["opening_hours"]["close"] == "20:30"
    print("  ✓ Active serving layer remains completely uncorrupted (Clean Rollback Verified)")

    status = atomic_poi_pipeline.get_pipeline_status()
    print(f"\nPipeline Health Status: {status}")
    assert status["total_validation_rejections"] >= 2
    assert status["zero_inconsistency_guarantee"] is True

    # Revert duration back to original
    atomic_poi_pipeline.update_poi_metadata(
        poi_id=poi_id,
        updates={"avg_visit_duration_mins": original_duration, "opening_hours": {"open": "09:00", "close": "20:00"}}
    )
    print("✅ Data Pipeline Consistency & Atomic Rollbacks: 100% SUCCESS!\n")


def test_unified_embeddings_zero_feature_skew():
    print("=" * 80)
    print("🧪 2. UNIFIED EMBEDDINGS PIPELINE & ZERO FEATURE SKEW VERIFICATION")
    print("=" * 80)

    test_queries = [
        "Παρθενώνας Ακρόπολη ιστορία",
        "Parthenon Acropolis ancient temple",
        "Μουσείο Ακρόπολης εισιτήρια και ωράριο",
        "Ιστορική βόλτα στα σοκάκια της Πλάκας με τα πόδια",
    ]

    for q in test_queries:
        # Preprocessing audit
        feats = unified_pipeline.extract_features(q)
        print(f"Query: '{q}'\n  -> Tokens: {feats.tokens[:4]} | Stems: {feats.stems[:4]} | Hash: {feats.feature_hash}")
        assert len(feats.tokens) >= 3
        assert len(feats.stems) >= 3

        # Zero-skew validation between batch call and streaming call
        v_stream = unified_pipeline.embed_text(q)
        v_batch = unified_pipeline.embed_batch([q])[0]

        skew_audit = unified_pipeline.verify_zero_skew(q, q)
        print(f"  -> Cosine Similarity: {skew_audit['cosine_similarity']:.6f} | Max Diff: {skew_audit['max_absolute_feature_difference']:.8f}")
        assert skew_audit["zero_skew_guaranteed"] is True
        assert skew_audit["cosine_similarity"] >= 0.999999
        assert skew_audit["max_absolute_feature_difference"] < 1e-6

    # Verify that RAG retriever query embedding matches unified pipeline output identically
    sample_text = "Ναός του Ηφαίστου στην Αρχαία Αγορά"
    v1 = unified_pipeline.embed_text(sample_text)
    from rag.retriever import get_embedding
    v2 = get_embedding(sample_text)

    diff = max(abs(a - b) for a, b in zip(v1, v2))
    print(f"\nRetriever get_embedding vs Unified Pipeline Diff: {diff:.8f}")
    assert diff < 1e-6, "Ingestion and streaming embedding paths deviated!"

    print("✅ Unified Ingestion & Streaming Feature Pipeline: ZERO FEATURE SKEW CONFIRMED!\n")


def test_operational_metrics_and_observability():
    print("=" * 80)
    print("🧪 3. OPERATIONAL METRICS & OBSERVABILITY (PROMETHEUS / GRAFANA)")
    print("=" * 80)

    # 3.1 Record synthetic requests across percentiles
    simulated_durations = [3.2, 4.5, 5.1, 7.8, 12.4, 15.0, 18.2, 22.0, 35.5, 48.0, 95.0]
    for d in simulated_durations:
        observability_registry.record_request("POST", "/api/v1/chat", 200, d)

    # Simulate 5xx incident
    observability_registry.record_request("POST", "/api/v1/itinerary/generate", 500, 120.5)

    stats = observability_registry.get_dashboard_stats()
    print(f"Traffic Stats: {stats['traffic']}")
    print(f"Latency Quantiles: {stats['latency']}")
    assert stats["traffic"]["total_requests"] >= 12
    assert stats["traffic"]["http_5xx_errors_total"] >= 1
    assert stats["traffic"]["error_rate_5xx_pct"] > 0.0

    # Verify quantile ordering: p50 <= p90 <= p95 <= p99
    lat = stats["latency"]
    print(f"Percentiles: p50={lat['p50_ms']}ms, p90={lat['p90_ms']}ms, p95={lat['p95_ms']}ms, p99={lat['p99_ms']}ms")
    assert lat["p50_ms"] <= lat["p90_ms"] <= lat["p95_ms"] <= lat["p99_ms"]

    # 3.2 Hardware Telemetry
    hw = observability_registry.get_hardware_telemetry()
    print(f"\nHardware Telemetry:\n  CPU: {hw['cpu_utilization_pct']}% (Ratio: {hw['cpu_ratio']}) | Memory RSS: {hw['memory_process_rss_mb']} MB | GPU: {hw['gpu_device']}")
    assert 0.0 <= hw["cpu_ratio"] <= 1.0
    assert hw["memory_process_rss_mb"] > 10.0

    # 3.3 Prometheus Exporter Format Verification
    prom_text = observability_registry.export_prometheus_metrics()
    print(f"\nPrometheus Export Preview (first 300 chars):\n{prom_text[:300]}...")
    assert "# HELP http_requests_total" in prom_text
    assert "# TYPE http_requests_total counter" in prom_text
    assert "# HELP http_server_errors_5xx_total" in prom_text
    assert "http_request_duration_seconds{quantile=\"0.99\"}" in prom_text
    assert "system_cpu_utilization_ratio" in prom_text

    # 3.4 FastAPI Endpoints Verification via TestClient
    client = TestClient(app)

    # GET /metrics
    resp_prom = client.get("/metrics")
    print(f"\nGET /metrics -> Status: {resp_prom.status_code}, Content-Type: {resp_prom.headers.get('content-type')}")
    assert resp_prom.status_code == 200
    assert "http_requests_total" in resp_prom.text

    # GET /api/v1/observability/stats
    resp_stats = client.get("/api/v1/observability/stats")
    print(f"GET /api/v1/observability/stats -> Status: {resp_stats.status_code}")
    assert resp_stats.status_code == 200
    data_stats = resp_stats.json()
    assert "sla_availability_pct" in data_stats["traffic"]
    assert "p99_ms" in data_stats["latency"]

    # GET /api/v1/pipeline/status
    resp_pipe = client.get("/api/v1/pipeline/status")
    print(f"GET /api/v1/pipeline/status -> Status: {resp_pipe.status_code}")
    assert resp_pipe.status_code == 200
    assert resp_pipe.json()["zero_inconsistency_guarantee"] is True

    print("✅ Operational Metrics & Observability: 100% SUCCESS — Prometheus & Grafana Ready!\n")


if __name__ == "__main__":
    test_data_pipeline_inconsistency_prevention()
    test_unified_embeddings_zero_feature_skew()
    test_operational_metrics_and_observability()
    print("=" * 80)
    print("🎉 ALL PRODUCTION PIPELINE & OBSERVABILITY TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)
