"""
Live Tickets & Availability Mock API for Athens Attractions (tools/ticketing.py).

Simulates the official Hellenic Heritage e-ticketing platform (hhticket.gr)
and private Athens museum booking systems:
- Real pricing & fee tiers (General, Reduced EU/Youth/Seniors, Free entries)
- Live time-slot availability & capacity monitoring
- Combined Athens Archaeological Sites Ticket (30€ / 7 sites / 5 days)
- Live reservation simulation with booking codes (e.g. ATH-TKT-2026-XXXX) & QR tokens
- High demand & sell-out alerts (e.g. peak hours at the Acropolis)
"""

from __future__ import annotations
import datetime
import random
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas for Ticketing Tool
# ---------------------------------------------------------------------------
class TicketPriceTier(BaseModel):
    tier_name: str
    price_eur: float
    description: str


class AttractionTicketCatalog(BaseModel):
    poi_id: str
    poi_name: str
    official_vendor: str
    regular_price_eur: float
    reduced_price_eur: float
    is_free_entry: bool = False
    combined_ticket_eligible: bool = False
    price_tiers: List[TicketPriceTier]
    free_entry_conditions: str
    skip_the_line_available: bool = True


class SlotAvailability(BaseModel):
    time_slot: str  # e.g. "09:00-10:00"
    total_capacity: int
    remaining_slots: int
    status: str  # available, limited, sold_out
    fast_track: bool = True


class AvailabilityResponse(BaseModel):
    poi_id: str
    poi_name: str
    date: str
    slots: List[SlotAvailability]
    recommendation: str


class TicketBookingRequest(BaseModel):
    poi_id: str = Field(..., description="Target POI ID (e.g. acropolis_hill, acropolis_museum)")
    visit_date: str = Field(default="today", description="Date formatted as YYYY-MM-DD or 'today'")
    time_slot: str = Field(default="10:00-11:00", description="Time slot window HH:MM-HH:MM")
    num_tickets: int = Field(default=1, ge=1, le=20, description="Total number of tickets")
    ticket_tier: str = Field(default="adult", description="Tier: adult, student, child, senior, combined")
    visitor_name: str = Field(default="Guest Traveler", description="Lead visitor name")


class BookingConfirmation(BaseModel):
    booking_reference: str
    poi_id: str
    poi_name: str
    visit_date: str
    time_slot: str
    num_tickets: int
    ticket_tier: str
    total_amount_eur: float
    qr_token: str
    status: str = "CONFIRMED"
    entry_instructions: str


# ---------------------------------------------------------------------------
# Official Ticket Data for Athens Attractions
# ---------------------------------------------------------------------------
ATHENS_TICKET_CATALOG: Dict[str, AttractionTicketCatalog] = {
    "acropolis_hill": AttractionTicketCatalog(
        poi_id="acropolis_hill",
        poi_name="Ιερός Βράχος Ακρόπολης & Παρθενώνας",
        official_vendor="Hellenic Heritage e-Ticket (hhticket.gr)",
        regular_price_eur=20.0,
        reduced_price_eur=10.0,
        combined_ticket_eligible=True,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=20.0, description="Κανονικό εισιτήριο ενήλικα"),
            TicketPriceTier(tier_name="reduced", price_eur=10.0, description="Μειωμένο (Πολίτες Ε.Ε. 65+, φοιτητές εκτός Ε.Ε.)"),
            TicketPriceTier(tier_name="student", price_eur=0.0, description="Δωρεάν (Νέοι Ε.Ε. έως 25 ετών & παιδιά έως 5 ετών)"),
            TicketPriceTier(tier_name="combined", price_eur=30.0, description="Ενιαίο Εισιτήριο 7 Αρχαιολογικών Χώρων (ισχύς 5 ημέρες)"),
        ],
        free_entry_conditions="Δωρεάν για πολίτες Ε.Ε. έως 25 ετών με επίδειξη ταυτότητας/διαβατηρίου.",
    ),
    "acropolis_museum": AttractionTicketCatalog(
        poi_id="acropolis_museum",
        poi_name="Μουσείο Ακρόπολης",
        official_vendor="theacropolismuseum.gr",
        regular_price_eur=15.0,
        reduced_price_eur=10.0,
        combined_ticket_eligible=False,  # Not part of the 30€ archaeological ticket
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=15.0, description="Κανονικό γενικής εισόδου"),
            TicketPriceTier(tier_name="reduced", price_eur=10.0, description="Μειωμένο εισιτήριο (Φοιτητές, 65+)"),
            TicketPriceTier(tier_name="child", price_eur=0.0, description="Δωρεάν για παιδιά κάτω των 5 ετών & μαθητές Ε.Ε."),
        ],
        free_entry_conditions="Δωρεάν είσοδος την πρώτη Κυριακή κάθε μήνα (χειμερινή περίοδος) & 28η Οκτωβρίου.",
    ),
    "hephaestus_temple": AttractionTicketCatalog(
        poi_id="hephaestus_temple",
        poi_name="Ναός Ηφαίστου (Αρχαία Αγορά)",
        official_vendor="Hellenic Heritage e-Ticket (hhticket.gr)",
        regular_price_eur=10.0,
        reduced_price_eur=5.0,
        combined_ticket_eligible=True,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=10.0, description="Κανονικό Αρχαίας Αγοράς"),
            TicketPriceTier(tier_name="reduced", price_eur=5.0, description="Μειωμένο εισιτήριο"),
            TicketPriceTier(tier_name="combined", price_eur=30.0, description="Περιλαμβάνεται στο Ενιαίο Εισιτήριο (30 €)"),
        ],
        free_entry_conditions="Δωρεάν για νέους Ε.Ε. κάτω των 25 ετών.",
    ),
    "national_archaeological_museum": AttractionTicketCatalog(
        poi_id="national_archaeological_museum",
        poi_name="Εθνικό Αρχαιολογικό Μουσείο",
        official_vendor="namuseum.gr / hhticket.gr",
        regular_price_eur=12.0,
        reduced_price_eur=6.0,
        combined_ticket_eligible=False,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=12.0, description="Γενική είσοδος"),
            TicketPriceTier(tier_name="reduced", price_eur=6.0, description="Μειωμένο εισιτήριο"),
            TicketPriceTier(tier_name="student", price_eur=0.0, description="Δωρεάν για φοιτητές Ε.Ε."),
        ],
        free_entry_conditions="Δωρεάν κάθε πρώτη Κυριακή του μήνα (Νοέμβριο - Μάρτιο).",
    ),
    "cycladic_art_museum": AttractionTicketCatalog(
        poi_id="cycladic_art_museum",
        poi_name="Μουσείο Κυκλαδικής Τέχνης",
        official_vendor="cycladic.gr",
        regular_price_eur=12.0,
        reduced_price_eur=9.0,
        combined_ticket_eligible=False,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=12.0, description="Κανονικό εισιτήριο"),
            TicketPriceTier(tier_name="reduced", price_eur=9.0, description="Μειωμένο (65+, φοιτητές 19-26 ετών)"),
            TicketPriceTier(tier_name="child", price_eur=0.0, description="Δωρεάν για παιδιά κάτω των 18 ετών"),
        ],
        free_entry_conditions="Δωρεάν για παιδιά και εφήβους κάτω των 18 ετών και ΑμεΑ με συνοδό.",
    ),
    "benaki_museum_greek_culture": AttractionTicketCatalog(
        poi_id="benaki_museum_greek_culture",
        poi_name="Μουσείο Μπενάκη Ελληνικού Πολιτισμού",
        official_vendor="benaki.org",
        regular_price_eur=12.0,
        reduced_price_eur=9.0,
        combined_ticket_eligible=False,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=12.0, description="Κανονικό"),
            TicketPriceTier(tier_name="reduced", price_eur=9.0, description="Μειωμένο"),
            TicketPriceTier(tier_name="student", price_eur=0.0, description="Δωρεάν είσοδος κάθε Πέμπτη 18:00-24:00"),
        ],
        free_entry_conditions="Δωρεάν είσοδος στη μόνιμη έκθεση κάθε Πέμπτη (εκτός περιοδικών εκθέσεων).",
    ),
    "goulandris_modern_art": AttractionTicketCatalog(
        poi_id="goulandris_modern_art",
        poi_name="Ίδρυμα Βασίλη & Ελίζας Γουλανδρή",
        official_vendor="goulandris.gr",
        regular_price_eur=10.0,
        reduced_price_eur=7.0,
        combined_ticket_eligible=False,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=10.0, description="Γενική είσοδος"),
            TicketPriceTier(tier_name="reduced", price_eur=7.0, description="Μειωμένο (φοιτητές, 65+)"),
            TicketPriceTier(tier_name="child", price_eur=0.0, description="Δωρεάν για παιδιά κάτω των 12 ετών"),
        ],
        free_entry_conditions="Δωρεάν για παιδιά έως 12 ετών.",
    ),
    "panathenaic_stadium": AttractionTicketCatalog(
        poi_id="panathenaic_stadium",
        poi_name="Παναθηναϊκό Στάδιο (Καλλιμάρμαρο)",
        official_vendor="panathenaicstadium.gr",
        regular_price_eur=10.0,
        reduced_price_eur=5.0,
        combined_ticket_eligible=False,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=10.0, description="Εισιτήριο με ακουστική ξενάγηση"),
            TicketPriceTier(tier_name="reduced", price_eur=5.0, description="Μειωμένο (φοιτητές, άνω των 65 ετών)"),
            TicketPriceTier(tier_name="child", price_eur=0.0, description="Δωρεάν για παιδιά έως 6 ετών"),
        ],
        free_entry_conditions="Δωρεάν για παιδιά κάτω των 6 ετών.",
    ),
    "national_garden": AttractionTicketCatalog(
        poi_id="national_garden",
        poi_name="Εθνικός Κήπος",
        official_vendor="Δήμος Αθηναίων",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[TicketPriceTier(tier_name="free", price_eur=0.0, description="Ελεύθερη είσοδος για όλους")],
        free_entry_conditions="Δημόσιος κήπος με ελεύθερη είσοδο (από την ανατολή έως τη δύση του ηλίου).",
    ),
    "plaka_historic_walk": AttractionTicketCatalog(
        poi_id="plaka_historic_walk",
        poi_name="Ιστορική Βόλτα στην Πλάκα",
        official_vendor="Δημόσιος Χώρος",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[TicketPriceTier(tier_name="free", price_eur=0.0, description="Ελεύθερη πρόσβαση")],
        free_entry_conditions="Υπαίθριος δημόσιος περίπατος χωρίς εισιτήριο.",
    ),
    "anafiotika": AttractionTicketCatalog(
        poi_id="anafiotika",
        poi_name="Αναφιώτικα",
        official_vendor="Δημόσιος Χώρος",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[TicketPriceTier(tier_name="free", price_eur=0.0, description="Ελεύθερη πρόσβαση")],
        free_entry_conditions="Παραδοσιακή συνοικία χωρίς εισιτήριο.",
    ),
    "monastiraki_flea_market": AttractionTicketCatalog(
        poi_id="monastiraki_flea_market",
        poi_name="Μοναστηράκι & Υπαίθριο Παζάρι",
        official_vendor="Δημόσιος Χώρος",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[TicketPriceTier(tier_name="free", price_eur=0.0, description="Ελεύθερη πρόσβαση")],
        free_entry_conditions="Δημόσια πλατεία και αγορά χωρίς εισιτήριο.",
    ),
    "syntagma_changing_guards": AttractionTicketCatalog(
        poi_id="syntagma_changing_guards",
        poi_name="Πλατεία Συντάγματος & Αλλαγή Φρουράς",
        official_vendor="Δημόσιος Χώρος",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[TicketPriceTier(tier_name="free", price_eur=0.0, description="Ελεύθερη πρόσβαση")],
        free_entry_conditions="Ελεύθερη παρακολούθηση της Αλλαγής Φρουράς στο Μνημείο του Αγνώστου Στρατιώτη.",
    ),
    "lycabettus_hill": AttractionTicketCatalog(
        poi_id="lycabettus_hill",
        poi_name="Λόφος Λυκαβηττού",
        official_vendor="Δημόσιος Χώρος (Τελεφερίκ: lycabettuscablecar.gr)",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[
            TicketPriceTier(tier_name="walking", price_eur=0.0, description="Πεζοπορία στον λόφο: Δωρεάν"),
            TicketPriceTier(tier_name="cable_car_return", price_eur=10.0, description="Τελεφερίκ Λυκαβηττού (με επιστροφή)"),
        ],
        free_entry_conditions="Ελεύθερη είσοδος και θέα από την κορυφή του λόφου.",
    ),
    "hellenic_children_museum": AttractionTicketCatalog(
        poi_id="hellenic_children_museum",
        poi_name="Ελληνικό Παιδικό Μουσείο",
        official_vendor="hcm.gr",
        regular_price_eur=0.0,
        reduced_price_eur=0.0,
        is_free_entry=True,
        price_tiers=[TicketPriceTier(tier_name="free_donation", price_eur=0.0, description="Ελεύθερη είσοδος / Προαιρετική δωρεά")],
        free_entry_conditions="Ελεύθερη είσοδος (προτείνεται προκράτηση θέσης τα Σαββατοκύριακα).",
    ),
    "eugenides_planetarium": AttractionTicketCatalog(
        poi_id="eugenides_planetarium",
        poi_name="Ευγενίδειο Πλανητάριο",
        official_vendor="eef.edu.gr",
        regular_price_eur=8.5,
        reduced_price_eur=6.5,
        combined_ticket_eligible=False,
        price_tiers=[
            TicketPriceTier(tier_name="adult", price_eur=8.5, description="Κανονική προβολή"),
            TicketPriceTier(tier_name="reduced", price_eur=6.5, description="Μειωμένο (παιδιά, μαθητές, φοιτητές)"),
        ],
        free_entry_conditions="Μειωμένες τιμές για οικογενειακά πακέτα.",
    ),
}


def get_ticket_pricing(poi_id: str) -> Dict[str, Any]:
    """Returns official ticket pricing, categories, and vendors for a POI."""
    poi_key = poi_id.lower().strip()
    if poi_key not in ATHENS_TICKET_CATALOG:
        # fuzzy fallback
        for k in ATHENS_TICKET_CATALOG:
            if k in poi_key or poi_key in k:
                poi_key = k
                break

    catalog = ATHENS_TICKET_CATALOG.get(poi_key)
    if not catalog:
        return {
            "poi_id": poi_id,
            "found": False,
            "message": f"Δεν βρέθηκαν επίσημα στοιχεία εισιτηρίων για το αναγνωριστικό '{poi_id}'.",
        }

    return {
        "found": True,
        "poi_id": catalog.poi_id,
        "poi_name": catalog.poi_name,
        "vendor": catalog.official_vendor,
        "is_free": catalog.is_free_entry,
        "regular_price_eur": catalog.regular_price_eur,
        "reduced_price_eur": catalog.reduced_price_eur,
        "combined_ticket_eligible": catalog.combined_ticket_eligible,
        "price_tiers": [t.model_dump() for t in catalog.price_tiers],
        "free_entry_conditions": catalog.free_entry_conditions,
        "skip_the_line": catalog.skip_the_line_available,
    }


def check_ticket_availability(
    poi_id: str,
    visit_date: Optional[str] = None,
    time_slot: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Checks live time slot availability and capacity for a POI.
    Generates realistic availability metrics and warnings for high-demand slots.
    """
    date_str = visit_date or datetime.date.today().isoformat()
    pricing = get_ticket_pricing(poi_id)
    if not pricing.get("found"):
        return {"poi_id": poi_id, "available": False, "message": "Άγνωστο αξιοθέατο."}

    if pricing.get("is_free"):
        return {
            "poi_id": poi_id,
            "poi_name": pricing["poi_name"],
            "date": date_str,
            "is_free": True,
            "message": f"Η είσοδος στο {pricing['poi_name']} είναι ελεύθερη χωρίς απαίτηση κράτησης εισιτηρίου.",
            "slots": [],
        }

    # Standard hourly time slots
    slot_hours = [
        ("08:00-09:00", 120, 65, "available"),
        ("09:00-10:00", 250, 15, "limited"),
        ("10:00-11:00", 300, 0, "sold_out"),
        ("11:00-12:00", 300, 12, "limited"),
        ("12:00-13:00", 200, 85, "available"),
        ("13:00-14:00", 180, 110, "available"),
        ("14:00-15:00", 180, 140, "available"),
        ("15:00-16:00", 200, 95, "available"),
        ("16:00-17:00", 250, 40, "available"),
        ("17:00-18:00", 280, 8, "limited"),
        ("18:00-19:00", 200, 45, "available"),
        ("19:00-20:00", 150, 70, "available"),
    ]

    slots: List[SlotAvailability] = []
    for slot_name, cap, rem, st in slot_hours:
        # If user asked for specific slot, keep or filter
        slots.append(SlotAvailability(
            time_slot=slot_name,
            total_capacity=cap,
            remaining_slots=rem,
            status=st,
            fast_track=True,
        ))

    recommendation = (
        f"💡 **Σύσταση Κράτησης για {pricing['poi_name']} ({date_str}):**\n"
        f"Οι πρωινές ώρες αιχμής (09:00-11:00) έχουν εξαντληθεί ή έχουν περιορισμένα εισιτήρια. "
        f"Συστήνουμε μεσημεριανή (13:00-15:00) ή απογευματινή επίσκεψη (18:00-19:30) για άνετη ξενάγηση."
    )

    resp = AvailabilityResponse(
        poi_id=pricing["poi_id"],
        poi_name=pricing["poi_name"],
        date=date_str,
        slots=slots,
        recommendation=recommendation,
    )
    return resp.model_dump()


def simulate_ticket_reservation(
    poi_id: str,
    visit_date: str = "today",
    time_slot: str = "10:00-11:00",
    num_tickets: int = 1,
    ticket_tier: str = "adult",
    visitor_name: str = "Guest Traveler",
) -> Dict[str, Any]:
    """
    Simulates instantaneous booking and issuance of verified e-tickets with QR codes.
    """
    date_str = visit_date if visit_date != "today" else datetime.date.today().isoformat()
    pricing = get_ticket_pricing(poi_id)
    if not pricing.get("found"):
        return {"status": "FAILED", "error": f"Invalid POI '{poi_id}'"}

    # Pricing calculation
    if pricing.get("is_free"):
        unit_price = 0.0
    elif ticket_tier in ["reduced", "senior"]:
        unit_price = pricing["reduced_price_eur"]
    elif ticket_tier in ["student", "child"]:
        unit_price = 0.0 if "acropolis" in poi_id else pricing["reduced_price_eur"]
    elif ticket_tier == "combined":
        unit_price = 30.0
    else:
        unit_price = pricing["regular_price_eur"]

    total_amount = round(unit_price * num_tickets, 2)
    booking_ref = f"ATH-{datetime.date.today().year}-TKT-{random.randint(10000, 99999)}"
    qr_token = f"QR-AUTH-ATHENS-{pricing['poi_id'][:4].upper()}-{random.randint(100000, 999999)}"

    instructions = (
        f"🎟️ **Επιβεβαίωση Κράτησης Εισιτηρίου ({booking_ref})**\n"
        f"• **Αξιοθέατο:** {pricing['poi_name']}\n"
        f"• **Ημερομηνία & Ώρα:** {date_str} | Ζώνη Εισόδου: {time_slot}\n"
        f"• **Εισιτήρια:** {num_tickets}x ({ticket_tier.upper()}) | Συνολικό Κόστος: {total_amount:.2f} €\n"
        f"• **Ηλεκτρονικό Barcode / QR:** `{qr_token}`\n"
        f"• **Οδηγία Εισόδου:** Παρακαλείστε να βρίσκεστε στην είσοδο 15 λεπτά πριν τη ζώνη σας. "
        f"Επιδείξτε το ψηφιακό QR στο μηχάνημα επικύρωσης."
    )

    confirmation = BookingConfirmation(
        booking_reference=booking_ref,
        poi_id=pricing["poi_id"],
        poi_name=pricing["poi_name"],
        visit_date=date_str,
        time_slot=time_slot,
        num_tickets=num_tickets,
        ticket_tier=ticket_tier,
        total_amount_eur=total_amount,
        qr_token=qr_token,
        status="CONFIRMED",
        entry_instructions=instructions,
    )
    return confirmation.model_dump()


# Tool schema for OpenAI / LLM function calling
TICKETING_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_ticket_availability",
        "description": (
            "Ελέγχει τιμές, διαθεσιμότητα χρονικών ζωνών (time-slots) και προσομοιώνει κρατήσεις "
            "εισιτηρίων για τα αξιοθέατα και τα μουσεία της Αθήνας. Χρησιμοποιείται όταν ο χρήστης "
            "ρωτά για κόστος εισιτηρίου, διαθεσιμότητα, συνδυαστικά εισιτήρια ή θέλει να κλείσει θέση."
        ),
        "parameters": TicketBookingRequest.model_json_schema(),
    },
}
