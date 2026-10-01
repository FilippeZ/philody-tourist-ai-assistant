"""test_eu_ai_act_compliance.py - Verification Suite for EU AI Act Compliance (Articles 50 & 12).

Validates:
1. 🔧 Limited Risk Classification & Pre-Exposure Transparency Notice (Article 50(1))
2. 🔄 Cryptographically-Chained Immutable Audit Logging & Tamper Detection (Article 12 Recordkeeping)
3. ➕ Machine-Readable Watermarking / Content Marking for Text & Audio (Article 50(2))
4. 🌐 FastAPI Compliance Endpoints Integrity (/api/v1/compliance/*, /api/v1/audio-tour/*, /api/v1/watermark/*)
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
import os
import sys
import json
import base64
import tempfile
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ["OPENAI_API_KEY"] = ""

from fastapi.testclient import TestClient
from api import app
from orchestrator import agent
agent._LLM_ENABLED = False


from orchestrator.audit_logger import ImmutableAuditLogger, audit_logger
from tools.watermarking import (
    SyntheticContentWatermarker,
    detect_text_watermark,
    EU_AI_ACT_ARTICLE_50_NOTICE,
)
from tools.audio_tour import (
    SyntheticAudioTourGenerator,
    detect_audio_watermark,
)



def test_pre_exposure_transparency_notice():
    print("=" * 80)
    print("🧪 1. PRE-EXPOSURE TRANSPARENCY NOTICE (EU AI ACT ARTICLE 50(1))")
    print("=" * 80)

    client = TestClient(app)
    session_id = f"test_compliance_session_{os.urandom(4).hex()}"

    # First turn: MUST deliver transparency notice before exposure to synthetic advice
    res1 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "Γεια σου! Ποιες είναι οι ώρες λειτουργίας του Μουσείου Ακρόπολης;"
    })
    assert res1.status_code == 200, f"Chat turn 1 failed: {res1.text}"
    data1 = res1.json()

    print(f"Turn 1 Transparency Notice Delivered: {data1.get('transparency_notice_delivered')}")
    assert data1["transparency_notice_delivered"] is True
    assert data1["transparency_notice"] is not None
    assert data1["transparency_notice"]["delivered_pre_exposure"] is True
    assert "Limited Risk AI System (Article 50)" in data1["transparency_notice"]["system_classification"]
    assert "Δήλωση Διαφάνειας" in data1["transparency_notice"]["message"]
    print(f"Notice Text Preview: {data1['transparency_notice']['message'][:70]}...")

    # Second turn: Should NOT re-deliver first-exposure notice, but maintain synthetic flag
    res2 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "Και ποιο είναι το κόστος εισιτηρίου;"
    })
    assert res2.status_code == 200, f"Chat turn 2 failed: {res2.text}"
    data2 = res2.json()

    print(f"Turn 2 Transparency Notice Delivered: {data2.get('transparency_notice_delivered')}")
    assert data2["transparency_notice_delivered"] is False
    assert data2["transparency_notice"] is None
    assert data2["is_synthetic_content"] is True
    print("✅ Pre-Exposure Transparency Notice (Article 50(1)) PASSED!\n")


def test_machine_readable_text_watermarking():
    print("=" * 80)
    print("🧪 2. MACHINE-READABLE TEXT WATERMARKING (EU AI ACT ARTICLE 50(2))")
    print("=" * 80)

    watermarker = SyntheticContentWatermarker()
    raw_text = "Η Ακρόπολη της Αθήνας είναι το σημαντικότερο μνημείο της κλασικής αρχαιότητας."
    
    watermarked = watermarker.watermark_text(
        raw_text=raw_text,
        session_id="session_art50_text",
        origin="chat_response"
    )

    print(f"Watermark present: {watermarked['has_machine_readable_watermark']}")
    print(f"Watermark Hash (SHA-256): {watermarked['watermark_hash']}")
    assert watermarked["has_machine_readable_watermark"] is True
    assert "watermarked_text" in watermarked
    assert len(watermarked["watermark_hash"]) == 64

    # Verify automated machine extraction and steganographic decoding
    detection = detect_text_watermark(watermarked["watermarked_text"])
    print(f"Detection result: {detection['is_watermarked']}, Standard: {detection['compliance_standard']}")
    assert detection["is_watermarked"] is True
    assert detection["compliance_standard"] == "EU_AI_ACT_ARTICLE_50"
    assert detection["metadata"]["session_id"] == "session_art50_text"
    assert detection["metadata"]["is_synthetic"] is True

    # Negative test on unwatermarked plain text
    plain_detection = detect_text_watermark("Απλό κείμενο χωρίς καμία σήμανση.")
    assert plain_detection["is_watermarked"] is False
    assert plain_detection["metadata"] is None
    print("✅ Machine-Readable Text Watermarking (Article 50(2)) PASSED!\n")


def test_synthetic_audio_tour_watermarking():
    print("=" * 80)
    print("🧪 3. SYNTHETIC AUDIO TOUR & RIFF/C2PA WATERMARK (EU AI ACT ARTICLE 50(2))")
    print("=" * 80)

    generator = SyntheticAudioTourGenerator()
    tour = generator.generate_tour(
        poi_name="Αρχαία Αγορά",
        topic="Δημοκρατία και Φιλοσοφία",
        session_id="session_audio_art50"
    )

    print(f"Tour Generation Compliant: {tour['article_50_compliant']}")
    print(f"Audio Size: {tour['audio_size_bytes']} bytes, Format: {tour['audio_format']}")
    assert tour["article_50_compliant"] is True
    assert tour["text_watermark_present"] is True
    assert tour["audio_format"] == "audio/wav"
    assert tour["audio_size_bytes"] > 1000

    # Decode audio bytes and inspect RIFF structure
    audio_bytes = base64.b64decode(tour["audio_base64"])
    assert audio_bytes.startswith(b"RIFF")

    audio_detection = detect_audio_watermark(audio_bytes)
    print(f"Audio Detection: Watermarked={audio_detection['is_watermarked']}, DisclosureFound={audio_detection['synthetic_disclosure_found']}")
    print(f"Tags found: {list(audio_detection['raw_tags'].keys())}")
    assert audio_detection["is_watermarked"] is True
    assert audio_detection["synthetic_disclosure_found"] is True
    assert audio_detection["compliance_standard"] == "EU_AI_ACT_ARTICLE_50"
    assert audio_detection["metadata"]["poi_name"] == "Αρχαία Αγορά"
    assert "Philody AI" in audio_detection["raw_tags"].get("ISFT", "")
    print("✅ Synthetic Audio Tour RIFF/C2PA Watermarking PASSED!\n")


def test_immutable_audit_logger_hash_chain():
    print("=" * 80)
    print("🧪 4. IMMUTABLE AUDIT LOG & CRYPTOGRAPHIC HASH CHAIN (EU AI ACT ARTICLE 12)")
    print("=" * 80)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        tmp_log_path = tmp_file.name

    try:
        logger = ImmutableAuditLogger(log_file_path=tmp_log_path)
        
        # Log series of chained events
        rec1 = logger.log_event(
            event_type="user_query",
            session_id="sess_chain_1",
            user_input="Πρόγραμμα για Ακρόπολη",
            system_output="Εδώ είναι το πρόγραμμα...",
            metadata={"step": 1}
        )
        assert rec1.sequence_id == 0
        assert rec1.previous_hash == "0" * 64
        assert len(rec1.record_hash) == 64

        rec2 = logger.log_event(
            event_type="itinerary_created",
            session_id="sess_chain_1",
            user_input="Έγκριση δρομολογίου",
            system_output="Δρομολόγιο εγκρίθηκε",
            metadata={"step": 2}
        )
        assert rec2.sequence_id == 1
        assert rec2.previous_hash == rec1.record_hash

        rec3 = logger.log_event(
            event_type="audio_tour_requested",
            session_id="sess_chain_1",
            user_input="Παραγωγή ηχητικής ξενάγησης",
            system_output="Audio tour ready",
            metadata={"step": 3}
        )
        assert rec3.sequence_id == 2
        assert rec3.previous_hash == rec2.record_hash

        print(f"Logged 3 chained records. Latest Hash: {rec3.record_hash[:16]}...")

        # Verify legitimate chain
        validity = logger.verify_chain_integrity()
        print(f"Legitimate Chain Integrity: is_valid={validity['is_valid']}, count={validity['verified_records_count']}")
        assert validity["is_valid"] is True
        assert validity["verified_records_count"] == 3
        assert validity["tamper_detected"] is False

        # Simulate adversarial tamper: Corrupt record 1 (second record) on disk
        with open(tmp_log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        lines[1]["user_query"] = "TAMPERED_QUERY_BY_ATTACKER"
        with open(tmp_log_path, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")


        # Re-verify tampered chain: MUST detect violation
        tampered_logger = ImmutableAuditLogger(log_file_path=tmp_log_path)
        tamper_check = tampered_logger.verify_chain_integrity()
        print(f"Tampered Chain Integrity: is_valid={tamper_check['is_valid']}, tamper_detected={tamper_check['tamper_detected']}, sequence={tamper_check.get('tampered_sequence_id')}")
        assert tamper_check["is_valid"] is False
        assert tamper_check["tamper_detected"] is True
        assert tamper_check["tampered_sequence_id"] == 1
        assert "Hash mismatch at sequence 1" in tamper_check["error"]

        print("✅ Immutable Audit Log & Tamper-Evident Hash Chain (Article 12) PASSED!\n")

    finally:
        if os.path.exists(tmp_log_path):
            os.remove(tmp_log_path)


def test_fastapi_compliance_endpoints():
    print("=" * 80)
    print("🧪 5. FASTAPI COMPLIANCE REST ENDPOINTS INTEGRATION")
    print("=" * 80)

    client = TestClient(app)

    # 1. Transparency Notice Endpoint
    resp_trans = client.get("/api/v1/compliance/transparency")
    assert resp_trans.status_code == 200
    d_trans = resp_trans.json()
    print(f"Transparency Endpoint Status: {d_trans['status']}, Classification: {d_trans['risk_classification']}")
    assert d_trans["status"] == "compliant"
    assert "Limited Risk" in d_trans["risk_classification"]
    assert d_trans["pre_exposure_enforced"] is True
    assert d_trans["machine_readable_watermarking_enabled"] is True
    assert d_trans["immutable_audit_logging_active"] is True

    # 2. Audit Verification Endpoint
    resp_verify = client.get("/api/v1/compliance/audit/verify")
    assert resp_verify.status_code == 200
    d_verify = resp_verify.json()
    print(f"Audit Verify Endpoint: is_valid={d_verify['is_valid']}, records={d_verify['verified_records_count']}")
    assert d_verify["is_valid"] is True
    assert d_verify["tamper_detected"] is False

    # 3. Audit Logs Endpoint
    resp_logs = client.get("/api/v1/compliance/audit/logs?limit=5")
    assert resp_logs.status_code == 200
    d_logs = resp_logs.json()
    print(f"Audit Logs Endpoint: retrieved {d_logs['total_records']} records")
    assert "total_records" in d_logs
    assert isinstance(d_logs["records"], list)

    # 4. Audio Tour Endpoint
    resp_audio = client.post("/api/v1/audio-tour/generate", json={
        "poi_name": "Μουσείο Ακρόπολης",
        "topic": "Αρχαϊκή Κόρη και Καρυάτιδες",
        "session_id": "test_api_audio"
    })
    assert resp_audio.status_code == 200
    d_audio = resp_audio.json()
    print(f"Audio Tour Generated: compliant={d_audio['article_50_compliant']}, size={d_audio['audio_size_bytes']} bytes")
    assert d_audio["article_50_compliant"] is True
    assert "audio_base64" in d_audio

    # 5. Watermark Verification Endpoint
    resp_wm = client.post("/api/v1/watermark/verify", json={
        "text": d_audio["script"],
        "audio_base64": d_audio["audio_base64"]
    })
    assert resp_wm.status_code == 200
    d_wm = resp_wm.json()
    print(f"Watermark Verification: Text={d_wm['text_watermark']['is_watermarked']}, Audio={d_wm['audio_watermark']['is_watermarked']}")
    assert d_wm["text_watermark"]["is_watermarked"] is True
    assert d_wm["audio_watermark"]["is_watermarked"] is True
    assert d_wm["audio_watermark"]["compliance_standard"] == "EU_AI_ACT_ARTICLE_50"

    print("✅ All EU AI Act FastAPI Endpoints PASSED!\n")


if __name__ == "__main__":
    test_pre_exposure_transparency_notice()
    test_machine_readable_text_watermarking()
    test_synthetic_audio_tour_watermarking()
    test_immutable_audit_logger_hash_chain()
    test_fastapi_compliance_endpoints()
    print("=" * 80)
    print("🎉 ALL 5/5 EU AI ACT COMPLIANCE TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 80)
