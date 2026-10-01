"""
Wearable IoT & Smart Bracelet Verification Test (Phase 3 Extension).
Verifies:
  1. Fatigue Alert (Heart rate telemetry -> Relaxed Pace mutation -> Rest break replanning)
  2. Proximity Geofencing (GPS 45m from POI -> Bite-sized card & Haptic notification)
  3. Ambient UV/Heat Alert (Wrist sensor 39.8°C -> Safety alert)
  4. Wearable Bite-sized Cards formatting for Smartwatch screens
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import sys
import io

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from orchestrator.agent import AthensTouristAgent
from tools.wearable import (
    HapticPattern,
    WearableEventType,
    format_wearable_card,
    simulate_wearable_event,
)


def run_wearable_tests():
    print("=" * 85)
    print("⌚ SMARTWATCH / SMART BRACELET (WEARABLE IoT EDGE) VERIFICATION")
    print("=" * 85)

    agent = AthensTouristAgent()

    # Step 0: Establish active itinerary
    init_res = agent.chat("Φτιάξε μου πρόγραμμα 3 ωρών στην Αθήνα.")
    assert agent.user_state.active_itinerary is not None, "Itinerary should be active"
    print("✅ Αρχικό δρομολόγιο δημιουργήθηκε στο UserState.")

    # 1. Test Fatigue Alert
    print("\n--- [1] Sensor Telemetry: Fatigue Alert & Heart Rate ---")
    fatigue_event = simulate_wearable_event(
        event_type=WearableEventType.FATIGUE_ALERT,
        heart_rate_bpm=142,
        fatigue_index=0.88,
    )
    res_fatigue = agent.handle_wearable_telemetry(fatigue_event)
    print(f"📡 IoT Payload: Heart Rate = 142 bpm | Fatigue Index = 0.88")
    print(f"📳 Haptic Pattern: {res_fatigue['haptic']}")
    print(f"💬 Wrist Notification: {res_fatigue['wrist_notification']}")

    assert "relaxed_pace" in agent.user_state.preferences, "relaxed_pace must be added to preferences"
    assert res_fatigue["haptic"] == HapticPattern.LONG_WARNING.value, "Expected long_warning haptic"
    assert "Στάση" in res_fatigue["wrist_notification"], "Should suggest a rest break"
    print("✓ Fatigue Telemetry & Dynamic Rest Injection: PASSED")

    # 2. Test Geofence Entry Proximity
    print("\n--- [2] Edge Sensor: GPS Proximity & Geofencing ---")
    geofence_event = simulate_wearable_event(
        event_type=WearableEventType.GEOFENCE_ENTRY,
        target_poi_id="hephaestus_temple",
    )
    res_geo = agent.handle_wearable_telemetry(geofence_event)
    print(f"📡 IoT Payload: Approaching hephaestus_temple at 45 meters")
    print(f"📳 Haptic Pattern: {res_geo['haptic']}")
    print(f"💬 Wrist Notification: {res_geo['wrist_notification']}")

    assert res_geo["haptic"] == HapticPattern.DOUBLE_PULSE.value, "Expected double_pulse haptic"
    assert "Ηφαίστου" in res_geo["wrist_notification"], "POI name should be displayed"
    assert "08:00" in res_geo["wrist_notification"], "Hours should be present"
    print("✓ GPS Geofencing Proximity Alert: PASSED")

    # 3. Test Ambient UV / Heat Alert
    print("\n--- [3] Microclimate Sensor: Ambient UV & Temperature ---")
    heat_event = simulate_wearable_event(
        event_type=WearableEventType.AMBIENT_HEAT_UV_ALERT,
        ambient_temp_c=40.2,
        uv_index=10.5,
    )
    res_heat = agent.handle_wearable_telemetry(heat_event)
    print(f"📡 IoT Payload: Ambient Temp = 40.2°C | UV Index = 10.5")
    print(f"💬 Wrist Notification: {res_heat['wrist_notification']}")

    assert "40.2" in res_heat["wrist_notification"] or "Ζέστη" in res_heat["wrist_notification"]
    print("✓ Wrist Microclimate Safety Alert: PASSED")

    # 4. Test Bite-Sized Cards for Smartwatch Display
    print("\n--- [4] Smartwatch UI: Bite-Sized Cards Generation ---")
    cards = res_fatigue.get("wearable_cards", [])
    print(f"📱 Παρήχθησαν {len(cards)} Bite-Sized κάρτες για την οθόνη του ρολογιού:")
    for c in cards[:3]:
        print(f"   [{c['icon']} Βήμα {c['step']}]: {c['title']} ({c['time_slot']}) | Haptic: {c['haptic']}")

    assert len(cards) > 0, "Wearable cards should be generated for the smartwatch screen"
    assert cards[0]["screen"] == "bite_sized_card", "Screen type must be bite_sized_card"
    print("✓ Smartwatch Compact Screen Cards: PASSED")

    # 5. Test Smart City IoT Sensor: Crowd Density & Replanning
    print("\n--- [5] Smart City IoT Sensor: Crowd Density & Congestion Replanning ---")
    crowd_res = agent.simulate_iot_sensor_event(
        sensor_type="crowd_density",
        poi_id="acropolis_hill",
        density_level="high",
        wait_time_mins=75,
    )
    print(f"📡 DOTSOFT Smart City Gateway Event: {crowd_res['event_type']} at {crowd_res.get('congested_poi')}")
    print(f"⏱️ Estimated Queue Wait: {crowd_res.get('queue_wait_mins')} mins")
    print(f"💬 Notification: {crowd_res.get('wrist_notification')}")

    assert crowd_res["processed"] is True
    assert "acropolis_hill" in agent.user_state.blacklisted_poi_ids, "Congested POI must be blacklisted"
    assert "Συνωστισμού" in crowd_res["wrist_notification"] or "DOTSOFT" in crowd_res["wrist_notification"]
    print("✓ Smart City Crowd Density Telemetry & Dynamic Rerouting: PASSED")

    print("\n" + "=" * 85)
    print("🎉 ΟΛΟΙ ΟΙ ΕΛΕΓΧΟΙ WEARABLE IoT & SMART CITY ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΜΕ 100% ΕΠΙΤΥΧΙΑ!")
    print("=" * 85)


if __name__ == "__main__":
    run_wearable_tests()

