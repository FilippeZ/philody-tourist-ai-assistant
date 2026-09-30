"""
External tools package for AI Tourist Assistant.
Exports Weather, Wearables, Public Transit, and Live Ticketing APIs.
"""

from .weather import (
    BASE_URL,
    WEATHER_TOOL_SCHEMA,
    WeatherResponse,
    get_current_weather,
    get_live_weather,
)
from .wearable import (
    HapticPattern,
    WearableCard,
    WearableEventType,
    format_wearable_card,
    simulate_wearable_event,
)
from .transit import (
    TRANSIT_TOOL_SCHEMA,
    TransitLeg,
    TransitRouteRequest,
    TransitRouteResponse,
    get_station_schedule,
    get_transit_alerts,
    get_transit_route,
)
from .ticketing import (
    ATHENS_TICKET_CATALOG,
    TICKETING_TOOL_SCHEMA,
    AttractionTicketCatalog,
    AvailabilityResponse,
    BookingConfirmation,
    SlotAvailability,
    TicketBookingRequest,
    TicketPriceTier,
    check_ticket_availability,
    get_ticket_pricing,
    simulate_ticket_reservation,
)

from .watermarking import (
    SyntheticContentWatermarker,
    detect_text_watermark,
)
from .audio_tour import (
    SyntheticAudioTourGenerator,
    detect_audio_watermark,
)

__all__ = [
    # Weather
    "get_current_weather",
    "WeatherResponse",
    "WEATHER_TOOL_SCHEMA",
    "BASE_URL",
    # Wearable IoT
    "simulate_wearable_event",
    "format_wearable_card",
    "WearableEventType",
    "WearableCard",
    "HapticPattern",
    # Public Transit
    "get_transit_route",
    "get_station_schedule",
    "get_transit_alerts",
    "TRANSIT_TOOL_SCHEMA",
    "TransitRouteRequest",
    "TransitRouteResponse",
    "TransitLeg",
    # Ticketing & Availability
    "check_ticket_availability",
    "get_ticket_pricing",
    "simulate_ticket_reservation",
    "TICKETING_TOOL_SCHEMA",
    "ATHENS_TICKET_CATALOG",
    "AttractionTicketCatalog",
    "TicketBookingRequest",
    "BookingConfirmation",
    "AvailabilityResponse",
    "SlotAvailability",
    "TicketPriceTier",
    # EU AI Act Watermarking & Synthetic Media
    "SyntheticContentWatermarker",
    "detect_text_watermark",
    "SyntheticAudioTourGenerator",
    "detect_audio_watermark",
]

