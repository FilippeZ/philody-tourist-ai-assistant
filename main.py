"""
AI Tourist Assistant - Athens Prototype (Phases 1-3).
State-of-the-Art Dual Interface: Interactive Streamlit UI and Terminal CLI.
Features:
- Glassmorphic Premium Dark UI with Google Fonts
- Dual-Pane AI Companion & Dynamic Leaflet.js Route Map with Audio TTS Tour
- Real-time Visual Itinerary Timeline with Walking Nodes & Constraint Verification
- Smartwatch / Wearable IoT Edge Client Simulator (OLED display, ECG, Fatigue replanning)
- Deterministic Feasibility Studio & Haversine Distance Calculator
- Automated Evaluation Suite (18/18 Test Cases Benchmark Runner)
- Interactive 360° Video Showcase of the Parthenon Wearable Companion
"""

from __future__ import annotations
import argparse
import html
import json
import math
import os
import sys

# Ensure UTF-8 output handling on Windows consoles (prevents cp1253/charmap UnicodeEncodeError)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load local environment variables if .env exists
load_dotenv()

from orchestrator.agent import AthensTouristAgent, IntentType
from tools.weather import get_current_weather
from tools.wearable import (
    WearableEventType,
    HapticPattern,
    format_wearable_card,
    simulate_wearable_event,
)
from engine.feasibility import (
    FeasibilityEngine,
    haversine_distance,
    calculate_travel_time_mins,
    time_to_minutes,
    minutes_to_time,
)
from rag.retriever import AthensRAGRetriever


# -----------------------------------------------------------------------------
# Terminal CLI Interactive Mode
# -----------------------------------------------------------------------------

def run_cli():
    """Runs the terminal CLI interactive chat loop."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 65)
    print("  🏛️  AI TOURIST ASSISTANT - ATHENS GUIDE (CLI MODE)")
    print("=" * 65)
    print("Διαθέσιμες εντολές: γράψτε την ερώτησή σας ή 'exit' για έξοδο.\n")

    agent = AthensTouristAgent()

    while True:
        try:
            user_input = input("\n👤 Ταξιδιώτης > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", "έξοδος"]:
                print("\nΚαλό σας ταξίδι στην Αθήνα! 👋")
                break

            response = agent.chat(user_input)
            intent = response.get("intent", "general")
            print(f"\n[🔍 Intent Detected: {intent.upper()}]")

            print("\n🤖 Αθηνά AI:\n")
            print(response.get("reply", ""))

            raw_plan = response.get("raw_plan")
            if raw_plan:
                print("\n[📊 Feasibility Engine Verified JSON]:")
                print(f"  - Feasible: {raw_plan.get('feasible')}")
                print(f"  - Window: {raw_plan.get('start_time')} - {raw_plan.get('end_time')}")
                print(f"  - Walking Distance: {raw_plan.get('total_distance_km')} km")
                print(f"  - Schedule items: {len(raw_plan.get('schedule', []))}")

        except (KeyboardInterrupt, EOFError):
            print("\nΤερματισμός. Αντίο!")
            break


# -----------------------------------------------------------------------------
# Landing Server Helper
# -----------------------------------------------------------------------------

def run_landing_server(port: int = 8000):
    """Starts a local HTTP server serving the landing page and opens it in the browser."""
    import http.server
    import socketserver
    import webbrowser
    import threading

    os.chdir(Path(__file__).parent.resolve())

    Handler = http.server.SimpleHTTPRequestHandler
    Handler.extensions_map.update({
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript",
        ".css": "text/css",
    })

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/index.html"
            print("=" * 65)
            print("  🎬 AI TOURIST ASSISTANT - INTERACTIVE VIDEO LANDING PAGE")
            print("=" * 65)
            print(f"📡 Web server running at: {url}")
            print(f"📁 Serving root directory: {Path(__file__).parent.resolve()}")
            print("💡 Συνδυασμός 80 frames (000-079) σε HTML5 360° Video Canvas.")
            print("🔄 Πατήστε Ctrl+C για έξοδο από τον server.\n")

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            httpd.serve_forever()
    except OSError as e:
        print(f"⚠️ Port {port} is occupied or error starting server: {e}")
        print(f"💡 Δοκιμάστε άλλη θύρα: python main.py --landing --port 8080")


# -----------------------------------------------------------------------------
# Leaflet Map Component Generator
# -----------------------------------------------------------------------------

def generate_leaflet_map_html(
    attractions: List[Dict[str, Any]],
    active_itinerary: Optional[Dict[str, Any]] = None,
    user_coords: Optional[Dict[str, float]] = None,
) -> str:
    """Generates a responsive Leaflet dark-mode map with interactive pins and routes."""
    user_lat = user_coords.get("lat", 37.9715) if user_coords else 37.9715
    user_lon = user_coords.get("lon", 23.7257) if user_coords else 23.7257

    # Extract selected POI coordinates for polyline route
    route_points = []
    ordered_poi_ids = []
    if active_itinerary and active_itinerary.get("feasible"):
        selected_ids = active_itinerary.get("selected_poi_ids", [])
        attraction_map = {a["id"]: a for a in attractions}
        for pid in selected_ids:
            if pid in attraction_map:
                c = attraction_map[pid]["coordinates"]
                route_points.append([c["lat"], c["lon"]])
                ordered_poi_ids.append(pid)

    attractions_json = json.dumps(attractions, ensure_ascii=False)
    route_points_json = json.dumps(route_points)
    ordered_ids_json = json.dumps(ordered_poi_ids)

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #080d1a;
      font-family: 'Plus Jakarta Sans', sans-serif;
      color: #f8fafc;
    }}
    #map {{
      width: 100%;
      height: 480px;
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      border: 1px solid rgba(255,255,255,0.12);
    }}
    .leaflet-popup-content-wrapper {{
      background: rgba(15, 23, 42, 0.94);
      color: #f8fafc;
      border: 1px solid rgba(6, 182, 212, 0.35);
      border-radius: 12px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.6);
      backdrop-filter: blur(8px);
    }}
    .leaflet-popup-tip {{
      background: rgba(15, 23, 42, 0.94);
    }}
    .popup-title {{
      font-weight: 700;
      font-size: 1rem;
      color: #38bdf8;
      margin-bottom: 4px;
    }}
    .popup-meta {{
      font-size: 0.78rem;
      color: #94a3b8;
      margin-bottom: 8px;
    }}
    .popup-desc {{
      font-size: 0.8rem;
      line-height: 1.4;
      color: #cbd5e1;
      margin-bottom: 10px;
    }}
    .audio-btn {{
      background: linear-gradient(135deg, #06b6d4, #3b82f6);
      color: #fff;
      border: none;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .audio-btn:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(6, 182, 212, 0.4);
    }}
    .order-badge {{
      display: inline-block;
      background: #f59e0b;
      color: #000;
      font-weight: 800;
      border-radius: 50%;
      width: 22px;
      height: 22px;
      text-align: center;
      line-height: 22px;
      font-size: 12px;
      margin-right: 6px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const attractions = {attractions_json};
    const routePoints = {route_points_json};
    const orderedIds = {ordered_ids_json};

    // Dark-mode map centered on central Athens
    const map = L.map('map', {{
      center: [37.9735, 23.7290],
      zoom: 14,
      zoomControl: true
    }});

    // Modern CartoDB Voyager / Dark theme tiles
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB & OpenStreetMap',
      subdomains: 'abcd',
      maxZoom: 19
    }}).addTo(map);

    // Audio TTS Speech Synthesis helper
    window.playAudioGuide = function(name, text) {{
      if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(name + '. ' + text);
        utterance.lang = 'el-GR';
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
      }} else {{
        alert('Η λειτουργία Audio Tour δεν υποστηρίζεται από το πρόγραμμα περιήγησης.');
      }}
    }};

    // Custom Icon Generator
    function createCustomIcon(category, isRoute, orderNum) {{
      let bgColor = '#3b82f6';
      let symbol = '🏛️';

      if (category === 'museum') {{ bgColor = '#8b5cf6'; symbol = '🏛️'; }}
      else if (category === 'park') {{ bgColor = '#10b981'; symbol = '🌿'; }}
      else if (category === 'neighborhood') {{ bgColor = '#f59e0b'; symbol = '🏘️'; }}
      else if (category === 'monument' || category === 'archaeological_site') {{ bgColor = '#06b6d4'; symbol = '⚡'; }}

      if (isRoute) {{
        return L.divIcon({{
          className: 'custom-pin',
          html: `<div style="background: ${{bgColor}}; border: 2px solid #fff; color: #fff; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; box-shadow: 0 0 14px rgba(6,182,212,0.8);">${{orderNum}}</div>`,
          iconSize: [32, 32],
          iconAnchor: [16, 16]
        }});
      }}

      return L.divIcon({{
        className: 'custom-pin',
        html: `<div style="background: ${{bgColor}}; border: 2px solid rgba(255,255,255,0.7); color: #fff; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.4);">${{symbol}}</div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13]
      }});
    }}

    // Add Markers for all Attractions
    attractions.forEach(attr => {{
      const idxInRoute = orderedIds.indexOf(attr.id);
      const isRoute = idxInRoute !== -1;
      const orderNum = idxInRoute + 1;

      const marker = L.marker([attr.coordinates.lat, attr.coordinates.lon], {{
        icon: createCustomIcon(attr.category, isRoute, orderNum)
      }}).addTo(map);

      const safeName = attr.name.replace(/'/g, "\\\\'");
      const safeDesc = attr.description.replace(/'/g, "\\\\'").slice(0, 140) + '...';

      let orderTag = isRoute ? `<span class="order-badge">${{orderNum}}</span>` : '';
      let popupHtml = `
        <div class="popup-title">${{orderTag}}${{attr.name}}</div>
        <div class="popup-meta">🕒 ${{attr.opening_hours.open}} - ${{attr.opening_hours.close}} | ⏱️ ${{attr.avg_visit_duration_mins}} min | ${{attr.type === 'indoor' ? '🏛️ Στεγασμένο' : '☀️ Υπαίθριο'}}</div>
        <div class="popup-desc">${{attr.description}}</div>
        <button class="audio-btn" onclick="playAudioGuide('${{safeName}}', '${{safeDesc}}')">
          🔊 Ακρόαση Audio Guide
        </button>
      `;

      marker.bindPopup(popupHtml);
    }});

    // Draw Walking Route Polyline if Active Itinerary exists
    if (routePoints.length > 1) {{
      const polyline = L.polyline(routePoints, {{
        color: '#06b6d4',
        weight: 5,
        opacity: 0.85,
        dashArray: '8, 8',
        lineCap: 'round'
      }}).addTo(map);

      map.fitBounds(polyline.getBounds(), {{ padding: [40, 40] }});
    }}

    // User Location Pulsing Dot
    const userMarker = L.circleMarker([{user_lat}, {user_lon}], {{
      radius: 8,
      fillColor: '#38bdf8',
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.9
    }}).addTo(map);
    userMarker.bindPopup("<b>📍 Η Τρέχουσα Τοποθεσία σας</b><br>GPS Edge Telemetry Live");
  </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Main Streamlit Application
# -----------------------------------------------------------------------------

def run_streamlit():
    """Renders the comprehensive multi-view Streamlit Web Application."""
    import streamlit as st
    import streamlit.components.v1 as components

    # Streamlit Page Setup
    st.set_page_config(
        page_title="AI Tourist Assistant - Athens | Next-Gen AI Guide",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Ultra-Modern CSS Design System
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

        /* Theme Base */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #f1f5f9;
        }

        h1, h2, h3, h4, .brand-font {
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }

        .hero-title {
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }

        .hero-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
        }

        .system-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        /* Glassmorphic Cards */
        .glass-card {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(6, 182, 212, 0.4);
        }

        /* Timeline Step Cards */
        .timeline-card {
            border-left: 3px solid #06b6d4;
            padding: 12px 16px;
            margin-bottom: 12px;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 0 12px 12px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .timeline-time {
            font-weight: 700;
            color: #38bdf8;
            font-size: 0.9rem;
        }

        .transit-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(100, 116, 139, 0.2);
            color: #cbd5e1;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            margin: 6px 0 12px 12px;
        }

        /* Wearable OLED Screen Frame */
        .watch-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .watch-frame {
            width: 320px;
            height: 320px;
            background: #020617;
            border: 10px solid #1e293b;
            border-radius: 50%;
            box-shadow: 0 0 35px rgba(6, 182, 212, 0.35), inset 0 0 25px rgba(0,0,0,0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 24px;
            box-sizing: border-box;
            position: relative;
        }

        .watch-time {
            font-size: 2.2rem;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            color: #f8fafc;
        }

        .pulse-heart {
            color: #ef4444;
            animation: heartBeat 1.2s infinite ease-in-out;
            display: inline-block;
        }
        @keyframes heartBeat {
            0% { transform: scale(1); }
            14% { transform: scale(1.25); }
            28% { transform: scale(1); }
            42% { transform: scale(1.2); }
            70% { transform: scale(1); }
        }

        /* Quick chip buttons */
        .stButton button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize agent in session state
    if "agent" not in st.session_state:
        st.session_state.agent = AthensTouristAgent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "active_itinerary" not in st.session_state:
        st.session_state.active_itinerary = None

    agent: AthensTouristAgent = st.session_state.agent

    # Load Attractions Knowledge Base
    kb_path = Path("data/athens_attractions.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        attractions = json.load(f)

    # =========================================================================
    # SIDEBAR: Controls, Real-time Weather, Traveler Persona & Reset
    # =========================================================================
    with st.sidebar:
        st.title("🏛️ Αθηνά AI Controls")
        st.caption("Dual Engine: Probabilistic RAG + Deterministic Feasibility")

        st.markdown("---")
        st.subheader("🌤️ Live Weather & Fallback")
        weather_scenario = st.selectbox(
            "Σενάριο Καιρού:",
            options=["rain_at_17", "clear", "heatwave", "live_api"],
            format_func=lambda x: {
                "rain_at_17": "🌧️ Βροχή στις 17:00 (Rain Mock)",
                "clear": "☀️ Ηλιόλουστος & Αίθριος (Clear)",
                "heatwave": "🔥 Καύσωνας 38.5°C (Extreme Heat)",
                "live_api": "🌐 Live API (OpenWeatherMap)",
            }[x],
        )

        override_scenario = None if weather_scenario == "live_api" else weather_scenario
        current_w = get_current_weather(mock_scenario=override_scenario)

        st.metric(
            label=f"Καιρός: {current_w['city']}",
            value=f"{current_w['temperature_c']} °C",
            delta=current_w["condition"],
        )
        if current_w.get("is_indoor_recommended"):
            st.warning(f"⚠️ **Σύσταση Στεγασμένων Χώρων**: {current_w.get('condition')} - {current_w.get('description')}")
        elif current_w.get("rain_expected"):
            st.warning(f"⚠️ Αναμένεται βροχή στις {current_w.get('rain_time')}!")

        st.markdown("---")
        st.subheader("🎒 Προφίλ Ταξιδιώτη (User State)")
        col_kid1, col_kid2 = st.columns(2)
        with col_kid1:
            kids_mode = st.checkbox(
                "Με παιδιά",
                value=agent.user_state.traveling_with_kids,
            )
        with col_kid2:
            child_age = st.number_input(
                "Ηλικία παιδιού",
                min_value=1,
                max_value=17,
                value=agent.user_state.child_age or 10,
                disabled=not kids_mode,
            )

        agent.user_state.traveling_with_kids = kids_mode
        if kids_mode:
            agent.user_state.child_age = int(child_age)

        current_pace = getattr(agent.user_state, "preferred_pace", "moderate")
        if current_pace not in ["relaxed", "moderate", "active"]:
            current_pace = "moderate"

        pace = st.selectbox(
            "Ρυθμός περιήγησης",
            options=["relaxed", "moderate", "active"],
            index=["relaxed", "moderate", "active"].index(current_pace),
            format_func=lambda x: {
                "relaxed": "☕ Χαλαρός (Relaxed)",
                "moderate": "🚶 Κανονικός (Moderate)",
                "active": "🏃 Εντατικός (Active)",
            }[x],
        )
        agent.user_state.preferred_pace = pace

        st.markdown("---")
        st.subheader("🎬 Video Landing Page")
        st.caption("360° Animation Wearable Companion με τα 80 frames.")
        st.link_button(
            "🌐 Άνοιγμα Landing Page (Port 8000)",
            url="http://localhost:8000/index.html",
            use_container_width=True,
        )

        if st.button("🔄 Επαναφορά Συνομιλίας & State", use_container_width=True):
            st.session_state.agent = AthensTouristAgent()
            st.session_state.chat_history = []
            st.session_state.active_itinerary = None
            st.rerun()

    # =========================================================================
    # HERO BANNER
    # =========================================================================
    st.markdown(
        f"""
        <div class="hero-banner">
          <div>
            <div class="hero-title">🇬🇷 AI Tourist Assistant: Athens City Guide</div>
            <div class="hero-subtitle">
              Αυστηρός διαχωρισμός <b>Πιθανοτικού LLM</b> (Φυσική Γλώσσα & RAG) & <b>Προσδιοριστικής Feasibility Engine</b>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # MULTI-VIEW TAB NAVIGATION
    # =========================================================================
    tab_chat, tab_watch, tab_feasibility, tab_eval, tab_video = st.tabs([
        "🏛️ AI Guide & Dynamic Map",
        "⌚ Wearable IoT Companion Hub",
        "⚙️ Feasibility Studio & Constraints",
        "📊 Evaluation Benchmarks (18 TCs)",
        "🎬 360° Video Showcase",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: AI GUIDE & DYNAMIC MAP (DUAL PANE)
    # -------------------------------------------------------------------------
    with tab_chat:
        col_chat, col_map = st.columns([1.1, 0.9], gap="large")

        # LEFT COLUMN: Interactive Chat & Citations
        with col_chat:
            st.subheader("💬 Συνομιλία με τον Ψηφιακό Ξεναγό")

            # Quick Question Testing Chips
            st.markdown("**Γρήγορες ερωτήσεις δοκιμής:**")
            cq1, cq2 = st.columns(2)
            with cq1:
                if st.button("🗓️ Πρόγραμμα 4 ωρών με παιδί 10 ετών", use_container_width=True):
                    st.session_state.pending_prompt = (
                        "Έχω 4 ώρες το απόγευμα στην Αθήνα, είμαι με το παιδί μου 10 ετών και θέλω ένα χαλαρό πρόγραμμα."
                    )
                if st.button("♿ Πλάκα με Αμαξίδιο", use_container_width=True):
                    st.session_state.pending_prompt = (
                        "Θέλω ένα πρόγραμμα για την Πλάκα, αλλά είμαι με αναπηρικό αμαξίδιο."
                    )
            with cq2:
                if st.button("🌧️ Ανασχεδιασμός λόγω Βροχής", use_container_width=True):
                    st.session_state.pending_prompt = (
                        "Έχουμε κανονίσει να πάμε στον Λυκαβηττό στις 17:00, αλλά συννέφιασε απότομα και θα βρέξει."
                    )
                if st.button("🏛️ Ιστορία Ναού Ηφαίστου (RAG)", use_container_width=True):
                    st.session_state.pending_prompt = (
                        "Ποια είναι η ιστορία του Ναού του Ηφαίστου στην Αρχαία Αγορά και ποιες είναι οι ώρες λειτουργίας;"
                    )

            # Chat History Container
            chat_container = st.container(height=420)
            with chat_container:
                if not st.session_state.chat_history:
                    st.info(
                        "👋 Γεια σας! Είμαι η **Αθηνά**, ο AI Ταξιδιωτικός Βοηθός της Αθήνας. "
                        "Μπορώ να απαντήσω ερωτήσεις με αυστηρές πηγές (RAG), να ελέγξω τον καιρό ή να σχεδιάσω "
                        "ένα 100% εφικτό δρομολόγιο προσαρμοσμένο στον χρόνο και τις ανάγκες σας!"
                    )

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("citations"):
                            with st.expander("📚 Επαληθευμένες Πηγές RAG (Citations)"):
                                for c in msg["citations"]:
                                    st.markdown(f"- **{c}**")
                        if msg.get("intent"):
                            st.caption(f"🔍 Intent: `{msg['intent']}`")

            # Chat Input Box
            prompt = st.chat_input("Ρωτήστε για αξιοθέατα, ωράρια ή ζητήστε πρόγραμμα...")
            if hasattr(st.session_state, "pending_prompt") and st.session_state.pending_prompt:
                prompt = st.session_state.pending_prompt
                st.session_state.pending_prompt = None

            if prompt:
                # Append user prompt
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.spinner("Η Αθηνά AI σκέφτεται & επικυρώνει μέσω της Feasibility Engine..."):
                    if override_scenario:
                        weather_payload = get_current_weather(mock_scenario=override_scenario)
                        if agent.detect_intent(prompt) == "itinerary_request":
                            res = agent.handle_itinerary_request(prompt, weather_override=weather_payload)
                        else:
                            res = agent.chat(prompt)
                    else:
                        res = agent.chat(prompt)

                    reply = res.get("reply", "")
                    citations = res.get("citations")
                    raw_plan = res.get("raw_plan")
                    intent_detected = res.get("intent")

                    if raw_plan and raw_plan.get("feasible"):
                        st.session_state.active_itinerary = raw_plan

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": reply,
                        "citations": citations,
                        "raw_plan": raw_plan,
                        "intent": intent_detected,
                    })
                st.rerun()

        # RIGHT COLUMN: Dynamic Leaflet Map & Timeline
        with col_map:
            st.subheader("🗺️ Διαδραστικός Χάρτης & Δρομολόγιο")

            # Render Leaflet Map
            map_html = generate_leaflet_map_html(
                attractions=attractions,
                active_itinerary=st.session_state.active_itinerary,
            )
            components.html(map_html, height=490)

            # Step-by-Step Schedule Visualizer
            if st.session_state.active_itinerary and st.session_state.active_itinerary.get("feasible"):
                plan = st.session_state.active_itinerary
                st.markdown(f"#### ⏱️ Επικυρωμένο Πρόγραμμα ({plan.get('start_time')} - {plan.get('end_time')})")
                st.caption(f"🚶 Συνολική Απόσταση Βάδισης: **{plan.get('total_distance_km')} km**")

                for step in plan.get("schedule", []):
                    if step.get("type") == "transit":
                        st.markdown(
                            f"""
                            <div class="transit-badge">
                              🚶 Μετάβαση: <b>{step.get('action')}</b> ({step.get('duration_mins')} λεπτά, {step.get('distance_km')} km)
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        env_icon = "🏛️" if step.get("env_type") == "indoor" else "☀️"
                        st.markdown(
                            f"""
                            <div class="timeline-card">
                              <div>
                                <span class="timeline-time">{step.get('time_slot')}</span> &bull; <b>{step.get('poi_name')}</b>
                                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                                  {env_icon} {step.get('env_type').title()} &bull; Διάρκεια: {step.get('duration_mins')} λεπτά
                                </div>
                              </div>
                              <span style="color:#10b981; font-size:0.85rem; font-weight:600;">✓ Επαληθευμένο</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.info("💡 Ζητήστε ένα δρομολόγιο στη συνομιλία για να εμφανιστεί η διαδρομή στον χάρτη και το αναλυτικό χρονοδιάγραμμα!")

    # -------------------------------------------------------------------------
    # TAB 2: WEARABLE IoT COMPANION HUB
    # -------------------------------------------------------------------------
    with tab_watch:
        st.subheader("⌚ Wearable IoT Edge Client (Smartwatch / Smart Bracelet)")
        st.caption("Προσομοιωτής φορητής συσκευής με πραγματική τηλεμετρία αισθητήρων και adaptive dynamic replanning.")

        col_device, col_telemetry = st.columns([1, 1], gap="large")

        with col_device:
            st.markdown("#### 📱 Προβολή Οθόνης Smartwatch (Round OLED Display)")

            # Get current watch data
            bpm_val = st.session_state.get("watch_bpm", 76)
            temp_val = st.session_state.get("watch_temp", 26.5)
            uv_val = st.session_state.get("watch_uv", 4.8)
            nearest_poi = st.session_state.get("watch_poi", "Παρθενώνας")
            dist_poi = st.session_state.get("watch_dist", 45)

            st.markdown(
                f"""
                <div class="watch-container">
                  <div class="watch-frame">
                    <div style="font-size: 0.75rem; color: #38bdf8; letter-spacing: 0.1em; text-transform: uppercase;">Athens AI Edge</div>
                    <div class="watch-time">15:42</div>
                    <div style="display:flex; align-items:center; gap:8px; margin: 6px 0;">
                      <span class="pulse-heart">❤️</span>
                      <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{bpm_val} BPM</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;">
                      🌡️ {temp_val}°C &bull; UV {uv_val}
                    </div>
                    <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); border-radius: 10px; padding: 6px 12px; font-size: 0.75rem; color: #e2e8f0;">
                      📍 <b>{nearest_poi}</b><br>
                      Απόσταση: <b>{dist_poi}m</b> &bull; Haptic: Active
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_telemetry:
            st.markdown("#### 📡 Ένεση Τηλεμετρίας Αισθητήρων (IoT Telemetry Controls)")

            sim_bpm = st.slider("Καρδιακοί Παλμοί (BPM)", min_value=60, max_value=180, value=bpm_val)
            st.session_state["watch_bpm"] = sim_bpm

            sim_temp = st.slider("Θερμοκρασία Καρπού (°C)", min_value=15.0, max_value=45.0, value=temp_val, step=0.5)
            st.session_state["watch_temp"] = sim_temp

            sim_uv = st.slider("Δείκτης UV", min_value=1.0, max_value=12.0, value=uv_val, step=0.2)
            st.session_state["watch_uv"] = sim_uv

            st.markdown("---")
            st.markdown("##### ⚡ Δοκιμές Σεναρίων Wearable Replanning")

            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("🏃 Ένεση Κόπωσης (142 BPM)", use_container_width=True):
                    st.session_state["watch_bpm"] = 142
                    telemetry = simulate_wearable_event(
                        WearableEventType.FATIGUE_ALERT,
                        heart_rate_bpm=142,
                        fatigue_index=0.88,
                    )
                    st.warning(
                        "⚠️ **FATIGUE DETECTED**: Ο αισθητήρας κατέγραψε 142 BPM. "
                        "Η Feasibility Engine εισήγαγε αυτόματα στάση ανάπαυσης 30 λεπτών."
                    )
                    agent.user_state.preferred_pace = "relaxed"
                    st.success("✅ Το UserState ενημερώθηκε σε pace: relaxed!")
                    st.rerun()

            with col_btn2:
                if st.button("📍 Geofence Ναού Ηφαίστου", use_container_width=True):
                    st.session_state["watch_poi"] = "Ναός Ηφαίστου"
                    st.session_state["watch_dist"] = 45
                    st.info("📍 **GEOFENCE TRIGGER**: Βρίσκεστε 45m από τον Ναό Ηφαίστου! Εστάλη διπλός παλμός δόνησης (double_pulse).")
                    st.rerun()

            with col_btn3:
                if st.button("👥 Συνωστισμός Ακρόπολης", use_container_width=True):
                    crowd_res = agent.simulate_iot_sensor_event(
                        sensor_type="crowd_density",
                        poi_id="acropolis_hill",
                        density_level="high",
                        wait_time_mins=75,
                    )
                    st.warning(f"🚨 **DOTSOFT SMART CITY**: {crowd_res.get('wrist_notification')}")
                    st.success("✅ Η Feasibility Engine αναδρομολόγησε αυτόματα το πλάνο για αποφυγή ουρών!")
                    st.rerun()


            if st.session_state.active_itinerary:
                st.markdown("##### 🗂️ Bite-Sized Cards για Smartwatch")
                cards = []
                for idx, step in enumerate(st.session_state.active_itinerary.get("schedule", [])[:3], 1):
                    card = format_wearable_card(step, step_idx=idx)
                    cards.append(card.to_dict())
                st.json(cards)

    # -------------------------------------------------------------------------
    # TAB 3: FEASIBILITY STUDIO & CONSTRAINTS
    # -------------------------------------------------------------------------
    with tab_feasibility:
        st.subheader("⚙️ Deterministic Feasibility Studio")
        st.caption("Μαθηματικός επικυρωτής αποστάσεων (Haversine Formula) και χρονικών περιορισμών.")

        col_math1, col_math2 = st.columns(2, gap="large")

        with col_math1:
            st.markdown("#### 📐 Υπολογιστής Απόστασης Haversine")
            poi_names = [a["name"] for a in attractions]
            p1_name = st.selectbox("Σημείο Αφετηρίας:", poi_names, index=0)
            p2_name = st.selectbox("Σημείο Προορισμού:", poi_names, index=1)

            p1 = next(a for a in attractions if a["name"] == p1_name)
            p2 = next(a for a in attractions if a["name"] == p2_name)

            dist_km = haversine_distance(p1["coordinates"], p2["coordinates"])
            travel = calculate_travel_time_mins(p1["coordinates"], p2["coordinates"], mode="walking")

            st.metric(
                label=f"Απόσταση {p1['name'][:18]} &rarr; {p2['name'][:18]}",
                value=f"{dist_km:.3f} km ({int(dist_km * 1000)} m)",
                delta=f"⏱️ {travel['duration_mins']} λεπτά βάδισης",
            )
            st.latex(
                r"d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)"
            )

        with col_math2:
            st.markdown("#### ⏳ Έλεγχος Χρονικού Περιορισμού Ωραρίου")
            st.markdown(r"Απαίτηση: $t_{\text{arrival}} + t_{\text{visit}} \le t_{\text{closing}}$")

            test_poi = st.selectbox("Επιλέξτε POI για δοκιμή:", poi_names, index=2)
            sel_poi = next(a for a in attractions if a["name"] == test_poi)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                arr_time_str = st.time_input("Ώρα Άφιξης:", value=None)
                arr_time = "18:30" if arr_time_str is None else arr_time_str.strftime("%H:%M")
            with col_t2:
                visit_dur = st.number_input("Διάρκεια επίσκεψης (min):", min_value=15, max_value=240, value=sel_poi["avg_visit_duration_mins"])

            arr_mins = time_to_minutes(arr_time)
            close_mins = time_to_minutes(sel_poi["opening_hours"]["close"])
            open_mins = time_to_minutes(sel_poi["opening_hours"]["open"])
            dep_mins = arr_mins + int(visit_dur)

            is_open = (arr_mins >= open_mins) and (dep_mins <= close_mins)

            st.write(f"• **Ωράριο λειτουργίας:** {sel_poi['opening_hours']['open']} - {sel_poi['opening_hours']['close']}")
            st.write(f"• **Ώρα αναχώρησης:** {minutes_to_time(dep_mins)}")

            if is_open:
                st.success("🟢 **FEASIBLE**: Η επίσκεψη ολοκληρώνεται πριν το κλείσιμο του αξιοθέατου!")
            else:
                st.error("🔴 **VIOLATION**: Η επίσκεψη υπερβαίνει το ωράριο κλεισίματος!")

        st.markdown("---")
        st.markdown("#### 📋 Επικυρωμένο JSON Feasibility Engine (Τρέχον Πλάνο)")
        if st.session_state.active_itinerary:
            st.json(st.session_state.active_itinerary)
        else:
            st.info("Δεν υπάρχει ενεργό δρομολόγιο. Δημιουργήστε ένα από το Tab 1!")

    # -------------------------------------------------------------------------
    # TAB 4: EVALUATION BENCHMARKS (18 TCs)
    # -------------------------------------------------------------------------
    with tab_eval:
        st.subheader("📊 Automated Evaluation Suite (Phase 2 Benchmarks)")
        st.caption("Εκτέλεση και των 18 αντιπροσωπευτικών Test Cases με strict deterministic assertions.")

        if st.button("🚀 Εκτέλεση Όλων των 18 Test Cases", use_container_width=True, type="primary"):
            from evaluate_dataset import load_dataset, evaluate_test_case

            dataset = load_dataset()
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            results = []
            for i, tc in enumerate(dataset):
                status_text.text(f"Εκτέλεση TC {tc['id']}: {tc['category']}...")
                res = evaluate_test_case(tc)
                results.append(res)
                progress_bar.progress((i + 1) / len(dataset))
                time.sleep(0.02)

            progress_bar.progress(1.0)
            status_text.text("Ολοκληρώθηκε!")

            passed_count = sum(1 for r in results if r["passed"])
            total_count = len(results)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Συνολικά Test Cases", total_count)
            with m2:
                st.metric("Επιτυχημένα (Passed)", passed_count)
            with m3:
                st.metric("Ποσοστό Επιτυχίας (Pass Rate)", f"{(passed_count / total_count) * 100:.1f}%")

            st.markdown("---")
            st.markdown("##### Αναλυτικά Αποτελέσματα ανά Test Case:")
            for r in results:
                status_badge = "🟢 PASSED" if r["passed"] else "🔴 FAILED"
                with st.expander(f"{status_badge} | {r['id']} - {r['category']} ({r['user_input'][:50]}...)"):
                    st.write(f"**User Prompt:** {r['user_input']}")
                    st.write(f"**Intent:** `{r['intent']}`")
                    st.write(f"**Latency:** {r['latency_sec']}s")
                    st.markdown("**Citations / Sources:**")
                    st.write(r.get("citations", []))
                    st.markdown("**Αποτέλεσμα Ελέγχων (Assertions):**")
                    st.json(r["assertions"])

    # -------------------------------------------------------------------------
    # TAB 5: 360° VIDEO SHOWCASE
    # -------------------------------------------------------------------------
    with tab_video:
        st.subheader("🎬 360° Video Showcase (Parthenon Wearable Companion)")
        st.caption("Συνδυασμός 80 frames (`landing/Βίντεο_έτοιμο_για_προβολή_000.jpg` έως `079.jpg`).")

        video_frame_idx = st.slider("🎞️ Χειροκίνητο Scrubbing Καρέ (Frame 0 - 79):", min_value=0, max_value=79, value=0)

        # Path to current frame
        frame_name = f"landing/Βίντεο_έτοιμο_για_προβολή_{video_frame_idx:03d}.jpg"
        frame_path = Path(frame_name)

        if frame_path.exists():
            col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
            with col_v2:
                st.image(str(frame_path), caption=f"Καρέ {video_frame_idx:02d} / 79 &bull; Smartwatch Companion στο ηλιοβασίλεμα του Παρθενώνα", use_column_width=True)
        else:
            st.warning(f"Το αρχείο καρέ {frame_name} δεν βρέθηκε.")

        st.info("💡 Για πλήρη εμπειρία 24 FPS αυτόματης αναπαραγωγής, scroll-scrubbing και HUD overlays, ανοίξτε το Video Landing Page:")
        st.link_button(
            "🌐 Άνοιγμα Πλήρους Video Landing Page (HTML5 Canvas)",
            url="http://localhost:8000/index.html",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# CLI vs Streamlit Entrypoint
# -----------------------------------------------------------------------------

def main():
    """Main selector between CLI, Streamlit, and Landing Page Server."""
    parser = argparse.ArgumentParser(description="AI Tourist Assistant - Athens")
    parser.add_argument("--cli", action="store_true", help="Launch in interactive CLI terminal mode")
    parser.add_argument("--landing", action="store_true", help="Launch local HTTP server for interactive video landing page")
    parser.add_argument("--port", type=int, default=8000, help="Port for the landing page HTTP server (default: 8000)")
    args, unknown = parser.parse_known_args()

    # Detect if run via streamlit runner
    is_streamlit_running = False
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None or (hasattr(st, "runtime") and st.runtime.exists()):
            is_streamlit_running = True
    except Exception:
        pass

    if not is_streamlit_running:
        is_streamlit_running = "streamlit" in sys.modules and (
            any("streamlit" in str(arg).lower() for arg in sys.argv) or "STREAMLIT_SERVER_PORT" in os.environ
        )

    if args.landing:
        run_landing_server(port=args.port)
    elif args.cli:
        run_cli()
    elif is_streamlit_running:
        run_streamlit()
    else:
        print("[TIP] Μπορείτε να εκτελέσετε το Web UI με την εντολή:")
        print("   streamlit run main.py")
        print("[TIP] Για το Interactive Video Landing Page:")
        print("   python main.py --landing\n")
        run_cli()


if __name__ == "__main__":
    main()
