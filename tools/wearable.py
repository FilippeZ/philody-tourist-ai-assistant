"""
Wearable IoT Edge Client & Telemetry Module (tools/wearable.py).

Provides smart device integration for Smartwatch / Smart Bracelet (Wearable IoT):
- Real-time Sensor Telemetry (Heart rate, Fatigue Index, Ambient UV/Temp, GPS Geofencing)
- Compact Screen Card Formatting (Bite-sized Cards for small round/square displays)
- Haptic Feedback Patterns & Proactive Context-Aware Alerts
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WearableEventType(str, Enum):
    FATIGUE_ALERT = "fatigue_alert"
    GEOFENCE_ENTRY = "geofence_entry"
    AMBIENT_HEAT_UV_ALERT = "ambient_heat_uv_alert"
    VOICE_COMMAND = "voice_command"
    CROWD_DENSITY = "crowd_density"


class HapticPattern(str, Enum):
    SINGLE_SHORT = "single_short"         # Simple confirmation
    DOUBLE_PULSE = "double_pulse"         # Proximity notification / turn navigation
    LONG_WARNING = "long_warning"         # Fatigue / Safety critical alert


@dataclass
class WearableCard:
    """Ultra-compact card tailored for round or rectangular smartwatch screens."""
    step_number: int
    title: str
    time_slot: str
    category_icon: str
    duration_mins: int
    distance_remaining_m: Optional[int] = None
    haptic_cue: HapticPattern = HapticPattern.SINGLE_SHORT
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen": "bite_sized_card",
            "step": self.step_number,
            "title": self.title,
            "time_slot": self.time_slot,
            "icon": self.category_icon,
            "duration_mins": self.duration_mins,
            "distance_m": self.distance_remaining_m,
            "haptic": self.haptic_cue.value,
            "compact_summary": self.summary_text,
        }


def format_wearable_card(step: Dict[str, Any], step_idx: int = 1) -> WearableCard:
    """Transforms a full itinerary step into a smartwatch bite-sized card."""
    is_transit = step.get("type") == "transit"
    icon = "🚶" if is_transit else ("🏛️" if step.get("env_type") == "indoor" else "🌿")
    title = step.get("action") if is_transit else step.get("poi_name", "Αξιοθέατο")
    time_slot = step.get("time_slot") or step.get("time", "")
    duration = step.get("duration_mins", 30)
    dist_km = step.get("distance_km", 0.0)
    dist_m = int(dist_km * 1000) if dist_km else None

    return WearableCard(
        step_number=step_idx,
        title=title,
        time_slot=time_slot,
        category_icon=icon,
        duration_mins=duration,
        distance_remaining_m=dist_m,
        haptic_cue=HapticPattern.DOUBLE_PULSE if is_transit else HapticPattern.SINGLE_SHORT,
        summary_text=f"{time_slot} ({duration}m)",
    )


def simulate_wearable_event(
    event_type: str | WearableEventType,
    heart_rate_bpm: Optional[int] = None,
    fatigue_index: Optional[float] = None,
    current_gps: Optional[Dict[str, float]] = None,
    target_poi_id: Optional[str] = None,
    ambient_temp_c: Optional[float] = None,
    uv_index: Optional[float] = None,
    voice_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulates real-time IoT telemetry received from the user's Smartwatch/Bracelet.
    Used for unit testing and demonstration of edge-driven replanning.
    """
    if isinstance(event_type, WearableEventType):
        event_str = event_type.value
    else:
        event_str = str(event_type).lower()

    payload: Dict[str, Any] = {
        "event_type": event_str,
        "timestamp_device": "15:42:00",
        "battery_pct": 78,
    }

    if event_str == WearableEventType.FATIGUE_ALERT.value:
        payload["telemetry"] = {
            "heart_rate_bpm": heart_rate_bpm or 138,
            "fatigue_index": fatigue_index or 0.86,  # > 0.7 triggers rest
            "consecutive_walking_mins": 95,
            "recommended_action": "inject_rest_stop",
        }
        payload["haptic_feedback"] = HapticPattern.LONG_WARNING.value

    elif event_str == WearableEventType.GEOFENCE_ENTRY.value:
        payload["telemetry"] = {
            "current_gps": current_gps or {"lat": 37.9750, "lon": 23.7224},
            "approaching_poi_id": target_poi_id or "hephaestus_temple",
            "distance_m": 45,
            "heading_degrees": 210,
        }
        payload["haptic_feedback"] = HapticPattern.DOUBLE_PULSE.value

    elif event_str == WearableEventType.AMBIENT_HEAT_UV_ALERT.value:
        payload["telemetry"] = {
            "ambient_temp_c": ambient_temp_c or 39.8,
            "uv_index": uv_index or 10.2,
            "exposure_duration_mins": 40,
            "warning": "High UV and extreme local heat detected at wrist sensor",
        }
        payload["haptic_feedback"] = HapticPattern.LONG_WARNING.value

    elif event_str == WearableEventType.CROWD_DENSITY.value or event_str == "crowd_density":
        payload["telemetry"] = {
            "gateway": "DOTSOFT_SmartCity_MQTT_v2",
            "poi_id": target_poi_id or "acropolis_hill",
            "density_level": "high",
            "wait_time_mins": 75,
            "recommended_action": "reroute_crowd_avoidance",
        }
        payload["haptic_feedback"] = HapticPattern.LONG_WARNING.value

    return payload


def simulate_iot_sensor_event(
    sensor_type: str = "crowd_density",
    poi_id: Optional[str] = None,
    density_level: Optional[str] = "high",
    wait_time_mins: Optional[int] = 75,
    heart_rate_bpm: Optional[int] = None,
    fatigue_index: Optional[float] = None,
    ambient_temp_c: Optional[float] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Simulates real-time IoT telemetry from DOTSOFT Smart City Gateway or Wearables:
    - sensor_type='crowd_density': Detects high congestion at a POI (e.g. Acropolis) and triggers dynamic replanning in the Feasibility Engine.
    - sensor_type='fatigue_alert': Detects physical exhaustion (>135 bpm) and injects a 30-min rest stop.
    - sensor_type='heatwave_alert': Detects extreme thermal stress and shifts outdoor activities indoors.
    """
    if sensor_type in ["crowd_density", "crowd_density_alert"]:
        payload = {
            "event_type": "crowd_density",
            "gateway": "DOTSOFT_SmartCity_MQTT_v2",
            "timestamp": "16:15:00",
            "telemetry": {
                "poi_id": poi_id or "acropolis_hill",
                "density_level": density_level or "high",
                "queue_wait_mins": wait_time_mins or 75,
                "congestion_index": 0.92,
            },
            "haptic_feedback": HapticPattern.LONG_WARNING.value,
        }
    elif sensor_type in ["fatigue_alert", "fatigue"]:
        payload = simulate_wearable_event(
            WearableEventType.FATIGUE_ALERT,
            heart_rate_bpm=heart_rate_bpm or 142,
            fatigue_index=fatigue_index or 0.88,
        )
    else:
        payload = simulate_wearable_event(
            WearableEventType.AMBIENT_HEAT_UV_ALERT,
            ambient_temp_c=ambient_temp_c or 40.2,
        )

    if agent is not None and hasattr(agent, "handle_iot_sensor_event"):
        return agent.handle_iot_sensor_event(payload)

    return payload

