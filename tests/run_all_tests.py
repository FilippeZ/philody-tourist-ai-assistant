"""
Athens AI Tourist Assistant - Unified Automated Test Suite Runner (tests/run_all_tests.py).

Executes all 9 unit, integration, and compliance test suites:
1. test_system.py (Core Architecture & 4 Subsystems)
2. test_use_cases.py (4 End-to-End Persona Scenarios)
3. test_tools_and_parsers.py (Strict Pydantic Parsers & Mock APIs)
4. test_wearable.py (IoT Edge Telemetry & Smart City Crowd Rerouting)
5. test_context_management.py (Token-Aware Sliding Window & Delimiters)
6. test_eu_ai_act_compliance.py (EU AI Act Articles 50 & 12 Compliance)
7. test_parsers_graph_observability.py (LangGraph Reflection Loops & Fallbacks)
8. test_chip_huyen_advancements.py (Parallel Guardrails, Contextual Chunking & Feedback)
9. test_production_pipelines.py (Zero Feature Skew & Prometheus Observability)
"""

from __future__ import annotations
import io
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT_DIR / "tests"

TEST_SUITES = [
    ("Core Architecture & Unit Tests", "test_system.py"),
    ("End-to-End Use Cases (4 Scenarios)", "test_use_cases.py"),
    ("Pydantic Output Parsers & Mock APIs", "test_tools_and_parsers.py"),
    ("Wearable Edge IoT & Smart City", "test_wearable.py"),
    ("Context Management & Delimiter Tokens", "test_context_management.py"),
    ("EU AI Act Compliance (Articles 50 & 12)", "test_eu_ai_act_compliance.py"),
    ("LangGraph Reflection Loops & Observability", "test_parsers_graph_observability.py"),
    ("Chip Huyen ML Advancements (Guardrails & Feedback)", "test_chip_huyen_advancements.py"),
    ("Production Pipelines & Prometheus Telemetry", "test_production_pipelines.py"),
]


def run_suite(name: str, filename: str) -> dict:
    file_path = TESTS_DIR / filename
    if not file_path.exists():
        return {
            "name": name,
            "filename": filename,
            "passed": False,
            "duration_s": 0.0,
            "error": f"File not found: {file_path}",
        }

    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["OPENAI_API_KEY"] = ""  # Force fast deterministic mode

    proc = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    duration = time.time() - start
    passed = (proc.returncode == 0)

    return {
        "name": name,
        "filename": filename,
        "passed": passed,
        "duration_s": round(duration, 2),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    print("=" * 80)
    print("🚀 ATHENS AI TOURIST ASSISTANT — UNIFIED AUTOMATED TEST RUNNER")
    print("=" * 80)
    print(f"Working Directory: {ROOT_DIR}")
    print(f"Test Suites Scheduled: {len(TEST_SUITES)}\n")

    results = []
    total_start = time.time()

    for idx, (suite_name, suite_file) in enumerate(TEST_SUITES, 1):
        print(f"[{idx}/{len(TEST_SUITES)}] Running {suite_file} ({suite_name})...", end="", flush=True)
        res = run_suite(suite_name, suite_file)
        results.append(res)
        if res["passed"]:
            print(f" -> ✅ PASSED ({res['duration_s']}s)")
        else:
            print(f" -> ❌ FAILED ({res['duration_s']}s)")
            print("-" * 60)
            if res.get("stderr"):
                print("STDERR:\n" + res["stderr"][-800:])
            if res.get("stdout"):
                print("STDOUT:\n" + res["stdout"][-800:])
            print("-" * 60)

    total_duration = round(time.time() - total_start, 2)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count
    pass_rate = (passed_count / len(results)) * 100

    print("\n" + "=" * 80)
    print("📊 UNIFIED QUALITY ASSURANCE TEST SCORECARD")
    print("=" * 80)
    print(f"{'#':<3} | {'Test Suite':<45} | {'File':<32} | {'Result':<8} | {'Time'}")
    print("-" * 105)
    for idx, r in enumerate(results, 1):
        status_str = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{idx:<3} | {r['name'][:45]:<45} | {r['filename'][:32]:<32} | {status_str:<8} | {r['duration_s']}s")
    print("-" * 105)
    print(f"Total Suites: {len(results)} | Passed: {passed_count} | Failed: {failed_count} | Pass Rate: {pass_rate:.1f}%")
    print(f"Total Execution Time: {total_duration}s")
    print("=" * 80)

    if failed_count == 0:
        print("🎉 ALL TEST SUITES PASSED! REPOSITORY IS 100% PRODUCTION READY.")
        sys.exit(0)
    else:
        print(f"⚠️ {failed_count} TEST SUITE(S) FAILED. PLEASE REVIEW LOGS ABOVE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
