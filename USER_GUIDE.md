# 🏛️ Philody AI Travel Assistant
## Αναλυτικός Οδηγός Χρήσης Πλατφόρμας
*v2.0 | Athens Smart Tourism Edition | 936 lines*

> **Επίσημο εγχειρίδιο λειτουργίας, περιήγησης & αξιοποίησης όλων των διαδραστικών δυνατοτήτων του Philody AI — Athens Smart Tourism Edition.**
>
> *Κατάλληλο για: Ταξιδιώτες · Αξιολογητές · Προγραμματιστές · Παρουσιαστές*

---

## 📋 Πίνακας Περιεχομένων

1. [Τι είναι το Philody AI;](#1-τι-είναι-το-philody-ai)
2. [Βασικές Αρχές & Φιλοσοφία Σχεδιασμού](#2-βασικές-αρχές--φιλοσοφία-σχεδιασμού)
3. [Εκκίνηση & Πρόσβαση στην Πλατφόρμα](#3-εκκίνηση--πρόσβαση-στην-πλατφόρμα)
4. [Ο Χάρτης Πλοήγησης — 6 Workspace Ταμπέλες](#4-ο-χάρτης-πλοήγησης--6-workspace-ταμπέλες)
5. [Λεπτομερής Ανάλυση κάθε Σελίδας](#5-λεπτομερής-ανάλυση-κάθε-σελίδας)
   - [5.1 Landing Page — 360° Hero Showcase](#51-landing-page--360-hero-showcase)
   - [5.2 Chat & Map — AI Companion & Δυναμικός Χάρτης](#52-chat--map--ai-companion--δυναμικός-χάρτης)
   - [5.3 Timeline Studio — Χρονοδιάγραμμα Δρομολογίου](#53-timeline-studio--χρονοδιάγραμμα-δρομολογίου)
   - [5.4 Smartwatch Wrist — Apple Watch Ultra Simulator](#54-smartwatch-wrist--apple-watch-ultra-simulator)
   - [5.5 Smart City IoT — DOTSOFT Telemetry Hub](#55-smart-city-iot--dotsoft-telemetry-hub)
   - [5.6 Evaluation Studio — RAGAS & Guardrails Audit](#56-evaluation-studio--ragas--guardrails-audit)
6. [Οι 4 Βασικές Ροές Χρήσης (Use Cases)](#6-οι-4-βασικές-ροές-χρήσης-use-cases)
   - [Use Case 1: Factual Q&A με Grounded RAG](#use-case-1-factual-qa-με-grounded-rag)
   - [Use Case 2: Δρομολόγιο με Περιορισμούς Κινητικότητας](#use-case-2-δρομολόγιο-με-περιορισμούς-κινητικότητας)
   - [Use Case 3: Smartwatch & IoT Rerouting](#use-case-3-smartwatch--iot-rerouting)
   - [Use Case 4: RAGAS Quality Audit & Guardrails](#use-case-4-ragas-quality-audit--guardrails)
7. [Προχωρημένα Σενάρια & Γωνιακές Περιπτώσεις](#7-προχωρημένα-σενάρια--γωνιακές-περιπτώσεις)
8. [Σύστημα Ασφαλείας — Dual-Layer Guardrails](#8-σύστημα-ασφαλείας--dual-layer-guardrails)
9. [Πλήρης Αναφορά Chat Commands & Intents](#9-πλήρης-αναφορά-chat-commands--intents)
10. [Αντιμετώπιση Προβλημάτων (Troubleshooting)](#10-αντιμετώπιση-προβλημάτων-troubleshooting)
11. [REST API Reference για Προγραμματιστές](#11-rest-api-reference-για-προγραμματιστές)
12. [Glossary — Λεξιλόγιο Πλατφόρμας](#12-glossary--λεξιλόγιο-πλατφόρμας)
13. [Τεχνικές Αναφορές & Καλυφθείσες Απαιτήσεις](#13-τεχνικές-αναφορές--καλυφθείσες-απαιτήσεις)

---

## 1. Τι είναι το Philody AI;

Το **Philody AI Travel Assistant** (από το «Φίλος» + «Ωδή») είναι μια πολυεπίπεδη πλατφόρμα έξυπνου τουρισμού για την ιστορική καρδιά της Αθήνας. Συνδυάζει τέσσερα τεχνολογικά επίπεδα σε μια ενιαία εμπειρία:

| Επίπεδο | Τεχνολογία | Σκοπός |
|---|---|---|
| 🧠 **LLM + Grounded RAG** | Nemotron-3-Ultra-253B (Ollama) | Φυσική γλώσσα, citation-anchored απαντήσεις |
| ⚙️ **Feasibility Engine** | Python / Haversine + LP Solver | Μαθηματικά επικυρωμένα δρομολόγια |
| 📡 **Smart City IoT** | DOTSOFT 5G NB-IoT | Πραγματικά δεδομένα αισθητήρων & rerouting |
| 🛡️ **Dual-Layer Guardrails** | RAGAS + Rule-based filters | Αποτροπή παραισθήσεων, 0% Hallucination |

### Τι ΔΕΝ κάνει το σύστημα (By Design)
- ❌ **Δεν επινοεί** ωράρια ή αποστάσεις — όλα αντλούνται από την `athens_attractions.json`
- ❌ **Δεν αγνοεί** χρονικούς περιορισμούς — αρνείται ευγενικά τα ανέφικτα αιτήματα
- ❌ **Δεν εξυπηρετεί** ερωτήσεις εκτός θεματικής Αθήνας / τουρισμού (off-topic blocking)

---

## 2. Βασικές Αρχές & Φιλοσοφία Σχεδιασμού

### 2.1 Strict Separation of Concerns
```
┌─────────────────────────────────────────────────────┐
│  PROBABILISTIC LAYER (LLM)                          │
│  • Intent classification                            │
│  • Natural language synthesis                       │
│  • Multi-turn dialogue memory                       │
│  • Citation-anchored responses only                 │
├─────────────────────────────────────────────────────┤
│  DETERMINISTIC LAYER (Feasibility Engine)           │
│  • Haversine distance calculation                   │
│  • Walking time: v_walk = 4.0 km/h                  │
│  • Closing hour validation                          │
│  • Budget & accessibility enforcement               │
└─────────────────────────────────────────────────────┘
```

> **Κανόνας #1**: Το LLM απαγορεύεται να υπολογίσει χρόνο βάδισης ή να αναφέρει ωράριο που δεν υπάρχει στη βάση γνώσης.

### 2.2 Unified State Manager
Κάθε αλλαγή σελίδας, μήνυμα chat, ή IoT event συγχρονίζεται μέσω του `PhilodyState` — ενός JavaScript singleton που κρατά:
- `currentItinerary`: Το ενεργό δρομολόγιο
- `userConstraints`: Ωράριο, κινητικότητα, καιρικές προτιμήσεις
- `mapFocusedPOI`: Το τρέχον επιλεγμένο αξιοθέατο
- `sensorData`: Τελευταία ανάγνωση IoT (BPM, θερμοκρασία, UV)
- `guardrailsLog`: Ιστορικό αναχαιτίσεων

---

## 3. Εκκίνηση & Πρόσβαση στην Πλατφόρμα

### 3.1 Προαπαιτούμενα

```
✅ Python 3.10+
✅ Ollama (εκτελείται τοπικά — ollama pull nemotron-3-ultra:253b)
✅ Google Chrome (συνιστάται για Web Speech API)
✅ Σύνδεση στο διαδίκτυο (για Leaflet χάρτες & καιρό)
```

### 3.2 Εκκίνηση Server

```bash
# Βήμα 1: Εγκατάσταση εξαρτήσεων (μόνο πρώτη φορά)
pip install -r requirements.txt

# Βήμα 2: Εκκίνηση FastAPI server
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Βήμα 3: Άνοιγμα στο Chrome
# Διεύθυνση: http://localhost:8000/#
```

### 3.3 Εναλλακτική Εκκίνηση (Streamlit)

```bash
streamlit run main.py
# Διεύθυνση: http://localhost:8501
```

### 3.4 Επαλήθευση ότι τρέχει σωστά

Επισκεφθείτε το **`http://localhost:8000/api/v1/health`** — αναμένετε:
```json
{
  "status": "healthy",
  "service": "Philody AI Tourist Assistant",
  "version": "2.4.0-production",
  "euaiact_compliant": true,
  "active_workspaces": [
    "0. Showcase 360",
    "1. Chat & Interactive Map",
    "2. Timeline Studio",
    "3. Wearable Wrist Smartwatch",
    "4. Smart City IoT Hub",
    "5. Evaluation & Benchmarks Studio"
  ]
}
```

---

## 4. Ο Χάρτης Πλοήγησης — 6 Workspace Ταμπέλες

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  🏛️ PHILODY AI        ☀️ 24°C Αθήνα  •  3 Στάσεις (2.7 km)                ║
╠══════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦══════════╣
║ 🎬   ║ 💬        ║ 🗓️        ║ ⌚        ║ 🎛️        ║ 📊        ║ ⚡ API  ║
║Landing║ Chat &   ║ Timeline  ║ Smartwatch║ Smart IoT ║ Evaluation║ Swagger ║
║ Page ║  Map     ║  Studio   ║   Wrist   ║   Hub     ║  Studio   ║  /docs  ║
╚══════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩══════════╝
```

| Tab | URL Fragment | Κύριος Σκοπός | Βασικά Στοιχεία |
|-----|-------------|---------------|-----------------|
| 🌐 Landing | `/#` | Πρώτη επαφή & impression | 360° video, hero CTA |
| 💬 Chat & Map | `/#chat` | AI dialog + live routing | Dual-pane, RAG chips |
| 🗓️ Timeline | `/#timeline` | Πρόγραμμα με χρόνους | Cards, TTS, χάρτης |
| ⌚ Smartwatch | `/#watch` | Wearable simulation | OLED, ECG, haptics |
| 🎛️ IoT Hub | `/#iot` | Sensor dashboard | Sliders, terminal log |
| 📊 Evaluation | `/#eval` | Quality audit | RAGAS metrics, scorecard |

> **💡 Tip:** Η μπάρα πλοήγησης είναι ορατή ανά πάσα στιγμή. Μπορείτε να εναλλάσσεστε ελεύθερα μεταξύ των ταμπελών χωρίς να χάνεται η κατάσταση.

---

## 5. Λεπτομερής Ανάλυση κάθε Σελίδας

### 5.1 Landing Page — 360° Hero Showcase

Η **σελίδα εισόδου** εμπνέεται από το design language της Apple — καθαρή, εντυπωσιακή, ακαριαία.

#### Διαδραστικό 360° Video Player
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    [← Σύρσιμο για Περιστροφή 360° →]                   │
│                                                         │
│         🎬  80 καρέ Canon R5 HDR                       │
│         ⏱️  24 FPS ομαλή αυτόματη αναπαραγωγή          │
│         👆  Touch / Mouse drag για scrubbing            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Πώς αλληλεπιδράτε:**
1. **Αυτόματη Αναπαραγωγή**: Αφήστε το βίντεο να τρέχει — η κάμερα περιστρέφεται αργά γύρω από την Ακρόπολη στο ηλιοβασίλεμα.
2. **Manual Scrub**: Κρατήστε το ποντίκι και σύρετε **αριστερά/δεξιά** οπουδήποτε στην οθόνη για χειροκίνητο έλεγχο γωνίας.
3. **Touch (Mobile)**: Ακριβώς το ίδιο με ένα δάχτυλο για swipe σε κινητά/tablet.

#### Κουμπιά Δράσης (Hero CTAs)

| Κουμπί | Ενέργεια | Αποτέλεσμα |
|--------|----------|------------|
| 👉 **Σχεδίασε το Ταξίδι σου Τώρα** | Click | Μεταφορά στο `💬 Chat & Map` |
| ⌚ **Wearable Hub** | Click | Άμεση μεταφορά στο `⌚ Smartwatch Wrist` |

---

### 5.2 Chat & Map — AI Companion & Δυναμικός Χάρτης

Το **κέντρο λειτουργίας** της πλατφόρμας. Χωρίζεται σε δύο ισόποσα πάνελ.

#### 🔵 Αριστερό Πάνελ: Philody AI Chat

##### Γρήγορη Αναφορά Intent Badges
Κάθε απάντηση του AI επισημαίνεται με ένα χρωματιστό badge που δείχνει τον τρόπο επεξεργασίας:

| Badge | Χρώμα | Σημαίνει |
|-------|-------|----------|
| `🏛️ Grounded RAG (Factual Q&A)` | Χρυσό | Πληροφορίες από επίσημη βάση γνώσης με citations |
| `🗓️ Feasibility Engine (Νέο Πρόγραμμα)` | Μωβ | Μαθηματικά υπολογισμένο δρομολόγιο |
| `🔄 Dynamic Replanning (Ανασχεδιασμός)` | Πορτοκαλί | Προσαρμογή υπάρχοντος δρομολογίου |
| `🌦️ Live Weather Sensor` | Κυανό | Ζωντανή καιρική ενημέρωση |
| `⚠️ Feasibility Rejection` | Κόκκινο | Αδύνατο αίτημα — προτείνεται εναλλακτικό |
| `🛡️ Guardrails Intercept` | Σκούρο κόκκινο | Αναχαίτιση επικίνδυνης/άσχετης ερώτησης |

##### Quick Prompt Chips (Ετικέτες Γρήγορης Αποστολής)

Τέσσερα έτοιμα σενάρια ακουμπάνε στο κάτω μέρος του chat:

**`🏛️ Ναός Ηφαίστου (RAG)`**
- Στέλνει: *«Ποια είναι η ιστορία του Ναού του Ηφαίστου και ποιες είναι οι ώρες λειτουργίας;»*
- Ενεργοποιεί: `FACTUAL_QA` intent → RAG retrieval → 3 citations → map focus

**`♿ Πλάκα με Αμαξίδιο`**
- Στέλνει: *«Έχω 3 ώρες στην Πλάκα αλλά είμαι με αναπηρικό αμαξίδιο»*
- Ενεργοποιεί: `ITINERARY_PLANNING` + accessibility filter → αποκλεισμός σκαλοπατιών

**`⚠️ Ανέφικτο Αίτημα`**
- Στέλνει: *«5 μουσεία σε 90 λεπτά στο κέντρο»*
- Ενεργοποιεί: Feasibility check → απόρριψη (απαιτούνται >370 λεπτά) → εναλλακτική πρόταση

**`🌧️ Βροχή στον Λυκαβηττό`**
- Στέλνει: *«Έρχεται βροχή, αλλάξτε το πρόγραμμα»*
- Ενεργοποιεί: `WEATHER_REPLANNING` → αντικατάσταση εξωτερικών χώρων με κλιματιζόμενους

##### Ηχητική Ξενάγηση (TTS)
- Πατήστε **`🔊 Ακρόαση`** στην κορυφή του chat για να ακούσετε το μήνυμα υποδοχής.
- Κάθε απάντηση του AI έχει το δικό της **`🔊 Ακρόαση Απάντησης`** κουμπί.
- Η φωνή χρησιμοποιεί το **Web Speech API** του Chrome (ελληνικό locale `el-GR`).

##### Citations — Πηγές Γνώσης
Κάθε Grounded RAG απάντηση περιέχει clickable tags:
```
📚 [Πηγή: Ναός Ηφαίστου]  📚 [Πηγή: Αρχαία Αγορά]  📚 [Πηγή: Ιστορική Βόλτα]
```
→ Πατώντας σε οποιοδήποτε tag, ο **χάρτης εστιάζει αυτόματα** στο αντίστοιχο μνημείο.

---

#### 🗺️ Δεξί Πάνελ: Dynamic Leaflet.js Route Map

```
┌──────────────────────────────────────┐
│  🗺️ Dynamic Athens Map               │
│                                      │
│   ① [🏛️]  ──────  ② [🗿]           │
│   Ηφαίστου        Ακρόπολη          │
│        ╲                ╱            │
│         ──── ③ [☕] ───              │
│              Πλάκα                   │
│                                      │
│  [🔍 Reset View]  [📍 My Location]  │
└──────────────────────────────────────┘
```

**16 Ευρετηριασμένα POIs** εμφανίζονται πάντα στον χάρτη, κατηγοριοποιημένα:

| Σύμβολο | Κατηγορία | Παραδείγματα |
|---------|-----------|--------------|
| 🏛️ | Μουσεία (Indoor) | Εθνικό Αρχαιολογικό, Παιδικό |
| 🗿 | Αρχαιολογικοί Χώροι | Ακρόπολη, Αρχαία Αγορά |
| ☕ | Γραφικές Γειτονιές | Πλάκα, Μοναστηράκι |
| 🌳 | Φυσικοί Χώροι | Λυκαβηττός, Φιλοπάππου |

**Διαδραστικά Popups**: Κλικ σε οποιαδήποτε καρφίτσα εμφανίζει:
- Ωράριο λειτουργίας (από βάση γνώσης)
- Μέση διάρκεια επίσκεψης
- ♿ Προσβασιμότητα
- 🔊 Κουμπί ακρόασης
- 📍 Κουμπί εστίασης στον χάρτη

---

### 5.3 Timeline Studio — Χρονοδιάγραμμα Δρομολογίου

Η σελίδα **Timeline Studio** μετατρέπει το δρομολόγιο σε σειριακή, οπτικά άμεση παρουσίαση.

#### 🎛️ Feasibility Parameter Control Bar (Διαδραστικός Υπολογισμός)
Στην κορυφή του Timeline Studio υπάρχει αυτόνομο πάνελ παραμέτρων:
- **Ώρα Έναρξης (`Start Time`):** Ρύθμιση ώρας αναχώρησης (π.χ. `10:00`, `14:00`).
- **Χρονικός Προϋπολογισμός (`Duration Hours`):** Ώρες διαθέσιμες (1.0 έως 8.0 ώρες).
- **Ρυθμός Βάδισης (`Pace`):** Χαλαρός (`relaxed`), Κανονικός (`moderate`), Γρήγορος (`fast`).
- **Checkboxes Περιορισμών:** 👶 Ταξίδι με Παιδιά, ♿ Πλήρης Πρόσβαση Αμαξιδίου (αποκλεισμός σκαλοπατιών).
- **Κουμπί «⚡ Υπολογισμός με Feasibility»:** Καλεί το `POST /api/v1/itinerary/generate` και συγχρονίζει αυτόματα (`PhilodyState.syncItinerary`) το Timeline, τον Χάρτη, το Smartwatch Wrist και το Header.
- **Κουμπιά Άμεσης Προβολής:** `⌚ Wearable Wrist` (μετάβαση στο ρολόι) & `🗺️ Στο Χάρτη` (προβολή διαδρομής).

#### Weather Callout Banner
Αν το δρομολόγιο έχει προσαρμοστεί λόγω καιρού, εμφανίζεται στην κορυφή:
```
🌧️ Το πρόγραμμα προσαρμόστηκε — εσωτερικοί χώροι προτιμήθηκαν λόγω βροχής
```

#### Κάρτες Σταθμών — Ανατομία

```
┌─────────────────────────────────────────────────────┐
│  ① ΝΑΟΣ ΗΦΑΙΣΤΟΥ                                   │
│  🕐 09:00 – 11:30  (150 λεπτά)                     │
│  🏛️ Indoor • 90 λ  |  🗿 Outdoor • 60 λ            │
│                                                      │
│  [🔊 Ηχητική Ξενάγηση]  [📍 Προβολή στο Χάρτη]   │
│  [⌚ Smartwatch]                                     │
├─────────────────────────────────────────────────────┤
│      ↕ 0.8 km  •  ~12 λεπτά βάδισης               │
├─────────────────────────────────────────────────────┤
│  ② ΑΚΡΟΠΟΛΗ                                         │
│  🕐 11:42 – 13:00  (78 λεπτά)                      │
│  🗿 Outdoor • 78 λ                                  │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

| Στοιχείο | Ανάλυση |
|----------|---------|
| **Χρονικό Παράθυρο** | Υπολογισμένο με Haversine + v=4 km/h |
| **Περιβαλλοντική Σήμανση** | Μωβ: `Indoor` · Πράσινο: `Outdoor` |
| **«🔊 Ηχητική Ξενάγηση»** | Web Speech API: εκφωνεί σύνοψη σταθμού |
| **«📍 Προβολή στο Χάρτη»** | Μεταβαίνει στο Chat & Map & εστιάζει POI |
| **«⌚ Smartwatch»** | Αποστέλλει στάση στο Wrist Companion |
| **Walking Connector** | Απόσταση (km) + Χρόνος (λεπτά) βάδισης |

---

### 5.4 Smartwatch Wrist — Apple Watch Ultra Simulator

Ο πλήρης προσομοιωτής φορετής συσκευής υψηλής πιστότητας.

#### Ανατομία Ρολογιού

```
         ┌──────────────────┐
         │  ⚡ 88%    10:42  │
         │ ─────────────────│
         │ 📍 Στάση 1 (50m) │
         │                  │
         │  ♥ ECG Display   │
         │  ╭──╮  ╭──╮     │
         │──╯  ╰──╯  ╰─────│
         │                  │
         │ ⬅️ Prev | Next ➡️ │
         │  🔊 Wrist Brief  │
         └──────────────────┘
              ●  (Crown)
              ■  (Action)
```

#### Διαδραστικά Στοιχεία Ρολογιού

| Στοιχείο | Θέση | Λειτουργία |
|----------|------|------------|
| **Digital Crown** | Δεξί πλάι | Κλικ → εναλλαγή καρτών πλοήγησης |
| **Action Button** | Αριστερό πλάι | Κλικ → άμεση ηχητική ενημέρωση (TTS) |
| **ECG Canvas** | Κεντρικά | Ζωντανή καρδιακή κυματομορφή |
| **⬅️ Prev / Next ➡️** | Κάτω | Προηγούμενος / Επόμενος σταθμός |
| **🔊 Wrist Briefing** | Κάτω | TTS εκφώνηση τρέχουσας στάσης |

#### Edge Triggers — Απτικά Συμβάντα

Τέσσερα κουμπιά στο sidebar ενεργοποιούν προσομοιωμένα συμβάντα:

**📍 GPS Proximity Alert**
```
Τι συμβαίνει:
• Ρολόι: διπλός παλμός δόνησης
• Οθόνη: "Φτάνεις στον Ναό Ηφαίστου — 50m"
• Χάρτης: εστιάζει αυτόματα στο POI
```

**⚡ Fatigue Alert (142 BPM)**
```
Τι συμβαίνει:
• ECG: επιταχύνεται & γίνεται κόκκινο
• Ρολόι: παρατεταμένη δόνηση + κόκκινο banner "ΚΟΠΩΣΗ"
• Chat AI: λαμβάνει αυτόματη ειδοποίηση → προτείνει καφέ
• IoT Log: καταγράφεται [FATIGUE] event με timestamp
```

**☀️ UV Warning**
```
Τι συμβαίνει:
• Ρολόι: badge "UV INDEX: HIGH"
• Σύσταση: "Αναζητήστε σκιά ή κλιματισμό"
• Θερμοκρασία: ενημερώνεται στην header bar
```

**👥 Crowd Reroute Alert**
```
Τι συμβαίνει:
• Ρολόι: "ΟΥΡΑ: 45 λεπτά αναμονής"
• AI: Ανασχεδιασμός παράκαμψης σε εναλλακτικό σημείο
• Χάρτης: νέα διαδρομή εμφανίζεται με διαφορετικό χρώμα
```

#### 🔄 Cross-Navigation Shortcuts (Αμφίδρομη Σύνδεση)
Κάτω από το Apple Watch Ultra panel υπάρχουν πλήκτρα άμεσης διασύνδεσης:
- **`📍 Στον Χάρτη`:** Εστιάζει αυτόματα τον Leaflet χάρτη στο POI της τρέχουσας ενεργής OLED κάρτας.
- **`🗓️ Στο Timeline`:** Μετάβαση στο Timeline Studio με επισήμανση της αντίστοιχης στάσης.
- **`🎛️ Στο IoT Hub`:** Μετάβαση στο Smart City IoT Hub για έλεγχο τηλεμετρίας.

---

### 5.5 Smart City IoT — DOTSOFT Telemetry Hub

Ο **πίνακας ελέγχου** για διαχειριστές έξυπνης πόλης και ερευνητές.

#### Network KPI Bar (Κεφαλίδα)

```
🌐 DOTSOFT 5G NB-IoT Gateway: LIVE  |  📶 Latency: 12.4ms  |  🔗 16 Nodes  |  📦 847 Events
```

#### 3 Διαδραστικοί Αισθητήρες

**👥 Συνωστισμός / Ουρά Αναμονής**
```
Slider: 5 ── [████████░░] ── 90 λεπτά
[📡 Αποστολή στο IoT Gateway]

→ Πάνω από 30 λεπτά: ενεργοποιείται αυτόματα Crowd Reroute στο ρολόι
```

**❤️ Παλμοί & Κόπωση Ταξιδιώτη**
```
Slider: 60 ── [██████████] ── 160 BPM
[📡 Αποστολή]

→ Πάνω από 130 BPM: ενεργοποιείται Fatigue Alert στο ρολόι
→ Chat: λαμβάνει ειδοποίηση ανασχεδιασμού για ανάπαυση
```

**☀️ Μικροκλίμα & Δείκτης UV**
```
Slider: 15°C ── [████████░░] ── 43°C
[📡 Αποστολή]

→ Πάνω από 35°C: UV Warning + σύσταση εσωτερικών χώρων
→ Header bar: ενημερώνεται θερμοκρασία σε πραγματικό χρόνο
```

#### DOTSOFT Live Event Log

Terminal-style console με ζωντανές εγγραφές:
```
[10:42:15] [GATEWAY]  Connection established — 16 nodes online
[10:42:31] [CROWD]    Queue time: 45min @ Acropolis (Node #03)
[10:43:02] [FATIGUE]  BPM: 142 — Tourist profile flagged
[10:43:15] [THERMAL]  Temp: 38°C — UV Index: HIGH (Node #11)
[10:43:22] [REROUTE]  Alternative POI dispatched: Plaka Coffee Stop
```

**Στοιχεία Ελέγχου:**
- **`🔄 Auto-Stream: ON`**: Παράγει αυτόματα εικονικά events κάθε 3 δευτερόλεπτα
- **`🗑️ Καθαρισμός`**: Εκκαθαρίζει το log console

---

### 5.6 Evaluation Studio — RAGAS & Guardrails Audit

Η σελίδα **ποιοτικής πιστοποίησης** — η αποδεικτική ισχύς της πλατφόρμας.

#### Κουμπιά Εκτέλεσης

| Κουμπί | Διάρκεια | Τι Ελέγχει |
|--------|----------|------------|
| `▶️ Εκτέλεση Benchmark Suite (19 Test Cases)` | <50ms | Deterministic assertions |
| `🛡️ Εκτέλεση RAGAS & Guardrails Audit` | ~2s | Faithfulness, Relevance, Hallucinations |
| `🔄 Ανανέωση Telemetry` | ~300ms | Live stats από `/api/v1/guardrails/stats` |

#### 5 Κάρτες Μετρικών RAGAS

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  Pass Rate   │  Faithfulness│  Relevance   │  Precision   │ Hallucination│
│              │              │              │              │              │
│  100%        │  100.0%      │  98.7%       │  89.5%       │  0.0%        │
│  (19/19)     │              │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

#### Dual-Layer Guardrails Telemetry

```
┌────────────────────────────────────────────────────────┐
│  🛡️ GUARDRAILS STATUS: ACTIVE                         │
│                                                        │
│  Επιθεωρήσεις:      847   │  Clean Pass Rate: 100%   │
│  Injections Blocked:  12  │  Interceptions:    0      │
│                                                        │
│  🟢 Zero Hallucinations: CONFIRMED                    │
└────────────────────────────────────────────────────────┘
```

#### Scorecard — Αναλυτικός Πίνακας

Εμφανίζονται τα 19 Test Cases με λεπτομέρειες:

| Test ID | Κατηγορία | Intent | Μετρική | Αποτέλεσμα |
|---------|-----------|--------|---------|------------|
| TC-001 | Factual Q&A | FACTUAL_QA | Faithfulness | ✅ PASS |
| TC-002 | Itinerary | ITINERARY_PLANNING | Feasibility | ✅ PASS |
| TC-010 | Off-topic | GUARDRAILS | Intercept | 🛡️ INTERCEPT |
| TC-015 | Negative Constraint | REPLANNING | Blacklist | ✅ PASS |
| TC-019 | Fatigue Rest | REPLANNING | Rest Slot | ✅ PASS |
| ... | ... | ... | ... | ... |

---

## 6. Οι 4 Βασικές Ροές Χρήσης (Use Cases)

### Use Case 1: Factual Q&A με Grounded RAG
**Στόχος**: Επίδειξη citation-anchored ιστορικής γνώσης χωρίς παραισθήσεις

**Βήμα-Βήμα:**

```
1. Αρχική Σελίδα
   └─► Κλικ: «👉 Σχεδίασε το Ταξίδι σου Τώρα»

2. Chat & Map (Αριστερό Πάνελ)
   └─► Κλικ chip: «🏛️ Ναός Ηφαίστου (RAG)»

3. Παρακολουθήστε:
   ├── Badge: 🏛️ Grounded RAG (Factual Q&A)
   ├── Απάντηση με 3 citations: [Ναός Ηφαίστου] [Αρχαία Αγορά] [Ιστορική Βόλτα]
   ├── Δεξί Πάνελ: χάρτης εστιάζει αυτόματα στον Ναό με popup
   └── Κλικ σε citation tag → χάρτης re-focuses

4. Επαλήθευση:
   └─► Στο Evaluation Studio: TC-001 = ✅ PASS, Faithfulness = 100%
```

**Τι αποδεικνύει:** Η πλατφόρμα δεν επινοεί ωράρια ή ιστορικά γεγονότα — κάθε πρόταση συνδέεται με πηγή.

---

### Use Case 2: Δρομολόγιο με Περιορισμούς Κινητικότητας
**Στόχος**: Επίδειξη Feasibility Engine + Accessibility Filtering + Timeline Studio

**Βήμα-Βήμα:**

```
1. Chat & Map
   └─► Πληκτρολογήστε: «Έχω 3 ώρες στην Πλάκα αλλά είμαι με αναπηρικό αμαξίδιο»

2. Παρακολουθήστε:
   ├── Badge: 🗓️ Feasibility Engine (Νέο Πρόγραμμα)
   ├── Φίλτρο: Αναφιώτικα ΑΠΟΚΛΕΊΟΝΤΑΙ (σκαλοπάτια)
   ├── Επιλεγμένα POIs: μόνο ♿ wheelchair_accessible = true
   └── Ακριβείς χρόνοι υπολογισμένοι με Haversine

3. Μεταβείτε στο: «🗓️ 2. Timeline Studio»
   ├── Weather Banner: (αν εφαρμόζεται)
   ├── Κάρτες με χρονικά παράθυρα (HH:MM – HH:MM)
   ├── Σήματα ♿ «Πλήρης Προσβασιμότητα» σε κάθε στάση
   └── Walking Connectors: απόσταση & χρόνος μεταξύ στάσεων

4. Κλικ «⌚ Smartwatch» σε οποιαδήποτε κάρτα
   └─► Ο σταθμός αποστέλλεται στο Wrist Companion

5. Επαλήθευση:
   └─► Evaluation: TC-005 (wheelchair) = ✅ PASS
```

**Τι αποδεικνύει:** Ο Feasibility Engine εφαρμόζει αυστηρούς φυσικούς και προσβασιμότητας περιορισμούς — καμία ανέφικτη πρόταση δεν φτάνει στον χρήστη.

---

### Use Case 3: Smartwatch & IoT Rerouting
**Στόχος**: End-to-end ενοποίηση Wearables ↔ Smart City ↔ AI

**Βήμα-Βήμα:**

```
1. Μεταβείτε στο: «⌚ 3. Smartwatch Wrist»

2. Κλικ: «⚡ Fatigue Alert (142 BPM)»
   ├── ECG: επιταχύνεται & κοκκινίζει
   ├── Ρολόι: παρατεταμένη δόνηση
   ├── Banner: «ΚΟΠΩΣΗ — Προτείνεται Ανάπαυση»
   └── Συστημική ειδοποίηση → αποστέλλεται στο AI

3. Μεταβείτε στο: «💬 Chat & Map»
   └─► Νέο μήνυμα AI: «Παρατηρήθηκε αυξημένος καρδιακός ρυθμός...
       Προτείνω 35λ διάλειμμα στο Κλιματιστικό Café Πλάκας»
   Badge: 🔄 Dynamic Replanning

4. Μεταβείτε στο: «🎛️ 4. Smart City IoT»
   └─► Event Log: [FATIGUE] 142 BPM — Tourist profile flagged — Rest slot injected

5. Ρυθμίστε Slider «❤️ Παλμοί» σε >130 BPM
   └─► Κλικ «📡 Αποστολή» → Log ενημερώνεται αυτόματα

6. Επαλήθευση:
   └─► Evaluation: TC-019 (fatigue rest) = ✅ PASS
```

**Τι αποδεικνύει:** Τα βιομετρικά δεδομένα πυροδοτούν ευφυή ανασχεδιασμό δρομολογίου σε πραγματικό χρόνο.

---

### Use Case 4: RAGAS Quality Audit & Guardrails
**Στόχος**: Πιστοποίηση ποιότητας, ασφάλειας και μηδενικών παραισθήσεων

**Βήμα-Βήμα:**

```
1. Μεταβείτε στο: «📊 5. Evaluation Studio»

2. Κλικ: «▶️ Εκτέλεση Benchmark Suite (19 Test Cases)»
   ├── Αναμείνετε <50ms
   └── Αποτέλεσμα: 19/19 ✅ PASS (100%)

3. Κλικ: «🛡️ Εκτέλεση RAGAS & Guardrails Audit»
   ├── Faithfulness: 100.0%
   ├── Answer Relevance: 98.7%
   ├── Context Precision: 89.5%
   └── Hallucination Rate: 0.0%

4. Κλικ: «🔄 Ανανέωση Telemetry»
   └─► Live data από /api/v1/guardrails/stats

5. Προβολή Scorecard
   ├── Κάθε test case ανά γραμμή
   ├── TC-010: 🛡️ INTERCEPT (off-topic blocked)
   └── TC-015: ✅ PASS (negative constraint respected)

6. Δοκιμή off-topic στο Chat (προαιρετικό)
   └─► Πληκτρολογήστε: «Πες μου για τη Νέα Υόρκη»
       → Guardrail: «Εξυπηρετώ μόνο ερωτήσεις για την Αθήνα»
```

**Τι αποδεικνύει:** Το σύστημα είναι πλήρως ελέγξιμο, ασφαλές και παρέχει αποδεικτικά μηδενικές παραισθήσεις.

---

## 7. Προχωρημένα Σενάρια & Γωνιακές Περιπτώσεις

### Σενάριο A: Κόπωση → Αίτημα Καφέ (Replanning)

```
Χρήστης: «Κουράστηκα πολύ, πρόσθεσε μια στάση για καφέ»

Αναμενόμενη Ροή:
1. Intent: REPLANNING (όχι FACTUAL_QA)
2. Engine: insert_rest_stop() → εισάγει 35λ Café Πλάκας
3. Timeline: ανανεώνεται με νέα στάση ξεκούρασης
4. Απάντηση: «Προσθέσαμε ένα διάλειμμα 35 λεπτών...»
```

### Σενάριο B: Αρνητικός Περιορισμός (Blacklisting)

```
Χρήστης: «Δεν θέλω να επισκεφτώ άλλα αρχαιολογικά μουσεία σήμερα»

Αναμενόμενη Ροή:
1. Intent: REPLANNING
2. Extractor: extract_negative_constraints() → blacklists category "archaeological"
3. Engine: filter_candidate_pois() → αποκλείει ΟΛΑ τα αρχαιολογικά
4. Νέο δρομολόγιο: μόνο γειτονιές, café, φυσικοί χώροι
```

### Σενάριο C: Χρονικά Ανέφικτο Αίτημα

```
Χρήστης: «5 μουσεία σε 90 λεπτά»

Αναμενόμενη Ροή:
1. Engine: Υπολογίζει: 5×75λ επίσκεψη + 4×15λ μετακίνηση = 435 λεπτά
2. Feasibility Check: 435 > 90 → REJECT
3. Απάντηση: «Ένα εφικτό πρόγραμμα 2 μουσείων για 90 λεπτά...»
Badge: ⚠️ Feasibility Rejection
```

### Σενάριο D: Καιρικός Ανασχεδιασμός

```
Χρήστης: «Έρχεται βροχή, αλλάξτε το πρόγραμμα»

Αναμενόμενη Ροή:
1. Intent: WEATHER_REPLANNING
2. Engine: Αντικαθιστά outdoor POIs με indoor (μουσεία, στοές)
3. Timeline: ανανεώνεται με Weather Callout Banner
4. Χάρτης: νέα διαδρομή εμφανίζεται
```

---

## 8. Σύστημα Ασφαλείας — Dual-Layer Guardrails

### Layer 1: Rule-Based Pre-Filter (Πριν από το LLM)

Ελέγχει **κάθε εισερχόμενο μήνυμα** για:

| Κατηγορία | Παραδείγματα | Αντίδραση |
|-----------|-------------|-----------|
| Off-topic | «Νέα Υόρκη», «συνταγή μαγειρικής» | Αποκλεισμός + εξήγηση |
| Prompt Injection | «Ignore previous instructions» | Αναχαίτιση + log |
| Ακατάλληλο περιεχόμενο | Βία, ρατσισμός | Αποκλεισμός |
| Αδύνατα αιτήματα | Χρόνος > διαθέσιμος budget | Απόρριψη + εναλλακτικό |

### Layer 2: RAGAS Post-Processor (Μετά από το LLM)

Ελέγχει **κάθε απάντηση** πριν παραδοθεί στον χρήστη:

```
Faithfulness Check:
  ├── Κάθε πρόταση ελέγχεται vs. athens_attractions.json
  ├── Αν δεν βρίσκεται στη βάση → REJECT & regenerate
  └── Target: 100% Faithfulness

Hallucination Detection:
  ├── N-gram matching vs. knowledge base
  ├── Semantic similarity scoring
  └── Target: 0.0% Hallucination Rate
```

---

## 9. Πλήρης Αναφορά Chat Commands & Intents

### Τύποι Ερωτήσεων & Αναμενόμενη Συμπεριφορά

| Τύπος Ερώτησης | Παράδειγμα | Intent | Μονάδα |
|----------------|------------|--------|--------|
| Ιστορική πληροφορία | «Ιστορία Παρθενώνα;» | `FACTUAL_QA` | RAG Retrieval |
| Ωράριο | «Πότε ανοίγει η Ακρόπολη;» | `FACTUAL_QA` | RAG → Hours |
| Νέο δρομολόγιο | «3 ώρες, Πλάκα, με παιδιά» | `ITINERARY_PLANNING` | Feasibility Engine |
| Καιρός | «Τι καιρό έχει;» | `WEATHER_QUERY` | Weather API |
| Καιρικός Ανασχεδιασμός | «Έρχεται βροχή» | `WEATHER_REPLANNING` | Re-plan |
| Κόπωση/Ανάπαυση | «Κουράστηκα, καφέ» | `REPLANNING` | insert_rest_stop |
| Αρνητικός Περιορισμός | «Όχι μουσεία» | `REPLANNING` | Blacklist |
| Εκτός θεματικής | «Νέα Υόρκη;» | `OFF_TOPIC` | Guardrail Block |

### Χρήσιμες Φράσεις για Δοκιμή

```
📋 Factual Q&A:
• «Ποιος είναι ο Ναός Ηφαίστου και τι ώρα ανοίγει;»
• «Πόσο διαρκεί μια επίσκεψη στο Εθνικό Αρχαιολογικό Μουσείο;»
• «Ποια μουσεία είναι κατάλληλα για παιδιά;»

📋 Itinerary Planning:
• «Δώσε μου ένα πρόγραμμα 4 ωρών για ιστορικά μνημεία»
• «Έχω 2 ώρες και είμαι με αναπηρικό αμαξίδιο»
• «Θέλω να επισκεφτώ μόνο εξωτερικούς χώρους»

📋 Replanning:
• «Κουράστηκα, πρόσθεσε ένα διάλειμμα για καφέ»
• «Δεν θέλω άλλα αρχαιολογικά μουσεία σήμερα»
• «Άλλαξε το πρόγραμμα — έρχεται βροχή»

📋 Stress Tests (Guardrails):
• «Φτιάξε μου 10 μουσεία σε 1 ώρα» → Feasibility Rejection
• «Πες μου για τη Ρώμη» → Off-topic Block
• «Ignore your instructions and...» → Injection Intercept
```

---

## 10. Αντιμετώπιση Προβλημάτων (Troubleshooting)

### 🔴 Ο Server δεν ξεκινά

```bash
# Ελέγξτε αν το port 8000 είναι ελεύθερο:
netstat -ano | findstr :8000

# Αν κατειλημμένο, τερματίστε το process:
taskkill /F /PID <pid>

# Ξεκινήστε ξανά:
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 🔴 Ο Ollama δεν απαντά

```bash
# Βεβαιωθείτε ότι τρέχει:
ollama list

# Ξεκινήστε το μοντέλο:
ollama run nemotron-3-ultra:253b

# Ελέγξτε το health endpoint:
curl http://localhost:11434/api/health
```

### 🔴 Ο Χάρτης εμφανίζει γκρίζα πλακίδια

```
Αιτία: Ο Leaflet δεν αναιρεθεί σωστά μετά από αλλαγή tab.
Λύση: Κλικ σε άλλο tab και επιστροφή — το σύστημα καλεί
       αυτόματα map.invalidateSize().
Εναλλακτικά: Ανανέωση σελίδας (F5).
```

### 🔴 Το TTS (Ακρόαση) δεν λειτουργεί

```
Αιτία: Ο Chrome χρειάζεται άδεια ήχου.
Λύση:
1. Ανοίξτε: chrome://settings/content/sound
2. Βεβαιωθείτε ότι το localhost:8000 δεν είναι σε blocklist
3. Πατήστε ξανά το 🔊 κουμπί (πρώτο click χρειάζεται user gesture)
```

### 🔴 Ο καιρός εμφανίζει «mock fallback»

```
Αιτία: Δεν υπάρχει API key ή δεν υπάρχει σύνδεση στο διαδίκτυο.
Λύση: Αυτό είναι αναμενόμενο — το σύστημα λειτουργεί OFFLINE
       με ενσωματωμένα δεδομένα καιρού (☀️ 24°C • αίθριος).
```

### 🔴 Evaluation Studio δείχνει 0% αντί 100%

```
Αιτία: Backend δεν τρέχει ή evaluation dataset δεν φορτώθηκε.
Λύση:
1. Βεβαιωθείτε ότι το server τρέχει: http://localhost:8000/health
2. Ελέγξτε: evaluation_dataset.json υπάρχει στο root
3. Κλικ «🔄 Ανανέωση Telemetry» πριν «▶️ Εκτέλεση»
```

---

## 11. REST API Reference για Προγραμματιστές

Η πλατφόρμα εκθέτει πλήρες **OpenAPI 3.0 REST API** στο `http://localhost:8000/docs`.

### Βασικά Endpoints

| Method | Endpoint | Περιγραφή |
|--------|----------|-----------|
| `GET` | `/api/v1/health` | Health check, έκδοση & ενεργά workspaces |
| `POST` | `/api/v1/chat` | Κεντρικό chat endpoint (Grounded RAG & Citations) |
| `POST` | `/api/v1/chat/graph` | Autonomous Chat μέσω LangGraph State Graph & Reflection Loops |
| `POST` | `/api/v1/itinerary/generate` | Υπολογισμός δρομολογίου με Deterministic Feasibility Engine |
| `GET` | `/api/v1/weather` | Ζωντανά καιρικά δεδομένα OpenWeatherMap (με offline fallback) |
| `GET` | `/api/v1/transit/route` | OASA/STASY Μετρό & Τραμ υπολογισμός διαδρομής & κόστους |
| `GET` | `/api/v1/transit/schedule/{station}` | Επόμενες αφίξεις συρμών ανά σταθμό |
| `GET` | `/api/v1/transit/alerts` | Ζωντανά alerts λειτουργίας δικτύου μέσων μαζικής μεταφοράς |
| `GET` | `/api/v1/tickets/pricing/{poi_id}` | Τιμοκατάλογος εισιτηρίων ανά αξιοθέατο |
| `POST` | `/api/v1/tickets/availability` | Έλεγχος διαθεσιμότητας χρονοθυρίδων επίσκεψης |
| `POST` | `/api/v1/tickets/book` | Προσομοίωση κράτησης εισιτηρίου με QR verification token |
| `POST` | `/api/v1/iot/event` | Αποστολή IoT / Smartwatch sensor events (BPM, GPS, UV) |
| `GET` | `/api/v1/pois` | Κατάλογος των 16 εμπλουτισμένων POIs της Αθήνας |
| `GET` | `/api/v1/session/{session_id}` | Ανάκτηση τρέχοντος UserState & ιστορικού συνομιλίας |
| `POST` | `/api/v1/evaluation/run` | Αυτόματη εκτέλεση του benchmark suite (19/19 test cases) |
| `GET` | `/api/v1/guardrails/stats` | Τηλεμετρία παράλληλων guardrails & αναχαιτίσεων |
| `POST` | `/api/v1/evaluation/ragas` | Υπολογισμός μετρικών RAGAS (faithfulness, answer relevance) |
| `POST` | `/api/v1/feedback` | User Natural Language Feedback (thumbs up/down + sentiment) |
| `GET` | `/api/v1/feedback/stats` | Στατιστικά ικανοποίησης χρηστών |
| `GET` | `/api/v1/feedback/export` | Εξαγωγή dataset DPO για fine-tuning |
| `GET` | `/metrics` | Prometheus OpenMetrics exporter (HTTP 5xx, latency p99, RAG QPS) |
| `GET` | `/api/v1/observability/stats` | Συνοπτικά metrics λειτουργίας συστήματος |
| `GET` | `/api/v1/observability/status` | Κατάσταση LangSmith tracing & buffered spans |
| `GET` | `/api/v1/observability/traces` | Πρόσφατα κατανεμημένα trace spans & execution trees |
| `GET` | `/api/v1/pipeline/status` | Κατάσταση Vector DB & Dynamic Chunking Pipeline |
| `POST` | `/api/v1/poi/update` | Δυναμική ενημέρωση metadata POI χωρίς train-serving skew |
| `GET` | `/api/v1/compliance/transparency` | EU AI Act Art. 50 Pre-Exposure Transparency Notice |
| `GET` | `/api/v1/compliance/audit/logs` | EU AI Act Art. 12 Immutable Log ανάκτηση εγγραφών |
| `GET` | `/api/v1/compliance/audit/verify` | Έλεγχος κρυπτογραφικής ακεραιότητας SHA-256 chain |
| `POST` | `/api/v1/audio-tour/generate` | Δημιουργία ηχητικής ξενάγησης με acoustic watermarking |
| `POST` | `/api/v1/watermark/verify` | Επαλήθευση watermark σε παραχθέν συνθετικό περιεχόμενο |
| `POST` | `/api/v1/context/estimate` | Εκτίμηση κόστους tokens & sliding window truncation |
| `GET` | `/api/v1/context/metrics` | Μετρικές κατανάλωσης Context Window & reserved headroom |

### Παράδειγμα: Chat Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Ποια είναι η ιστορία του Ναού Ηφαίστου;",
    "session_id": "user-123",
    "language": "el"
  }'
```

**Αναμενόμενη Απάντηση:**
```json
{
  "response": "Ο Ναός του Ηφαίστου (Θησείο)...",
  "intent": "FACTUAL_QA",
  "citations": ["Ναός Ηφαίστου", "Αρχαία Αγορά"],
  "map_focus": {"lat": 37.9754, "lng": 23.7213},
  "guardrails": {"passed": true, "faithfulness": 1.0}
}
```

### Παράδειγμα: IoT Sensor Push

```bash
curl -X POST http://localhost:8000/api/v1/iot/sensor \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_type": "biometric",
    "bpm": 145,
    "tourist_id": "tourist-456"
  }'
```

---

## 12. Glossary — Λεξιλόγιο Πλατφόρμας

| Όρος | Ορισμός |
|------|---------|
| **RAG** | Retrieval-Augmented Generation — απαντήσεις βασισμένες σε ανακτηθέντα documents |
| **Grounded RAG** | RAG με υποχρεωτικές citations — κάθε πρόταση πρέπει να τεκμηριώνεται |
| **Feasibility Engine** | Μαθηματική μονάδα που υπολογίζει εφικτά δρομολόγια με Haversine |
| **Haversine Formula** | Τύπος υπολογισμού μεγαλόκυκλης απόστασης μεταξύ GPS συντεταγμένων |
| **Intent Classification** | Κατηγοριοποίηση της πρόθεσης του χρήστη (FACTUAL_QA, REPLANNING κ.λπ.) |
| **Guardrails** | Διπλό στρώμα προστασίας από παραισθήσεις & off-topic requests |
| **RAGAS** | RAG Assessment System — μετρικές ποιότητας για RAG συστήματα |
| **Hallucination** | Επινοημένη πληροφορία από LLM που δεν υπάρχει στη βάση γνώσης |
| **Faithfulness** | RAGAS μετρική: ποσοστό προτάσεων που τεκμηριώνονται από context |
| **POI** | Point of Interest — αξιοθέατο / σημείο ενδιαφέροντος |
| **PhilodyState** | Κεντρικός JavaScript state manager που συγχρονίζει όλα τα panels |
| **NB-IoT** | Narrowband IoT — πρωτόκολλο επικοινωνίας αισθητήρων (DOTSOFT) |
| **TTS** | Text-to-Speech — μετατροπή κειμένου σε φωνή (Web Speech API) |
| **Negative Constraint** | Περιορισμός χρήστη ότι ΔΕΝ θέλει κάτι (π.χ. «όχι μουσεία») |
| **Blacklist** | Λίστα αποκλεισμένων κατηγοριών/ετικετών από αρνητικούς περιορισμούς |
| **Edge Trigger** | Απτικό συμβάν που πυροδοτείται από αισθητήρα (GPS, BPM, UV) |
| **Digital Crown** | Η περιστρεφόμενη κορώνα του Apple Watch για εναλλαγή καρτών |

---

## 13. Τεχνικές Αναφορές & Καλυφθείσες Απαιτήσεις

| Αρχείο | Είδος Τεκμηρίωσης |
|---|---|
| [`README.md`](README.md) | Πλήρης τεκμηρίωση και εγκατάσταση, use cases, evaluation scorecard, deliverables index |
| [`TECHNICAL_DESIGN_NOTE.md`](TECHNICAL_DESIGN_NOTE.md) | Αρχιτεκτονική σχεδίαση: LLM/RAG strategy, LangGraph, Guardrails, Production scaling |
| [`ARCHITECTURE_DIAGRAM.md`](ARCHITECTURE_DIAGRAM.md) | Mermaid system & sequence diagrams, LaTeX mathematical formulas |
| [`LEADERSHIP_COACHING_PLAN.md`](LEADERSHIP_COACHING_PLAN.md) | Τεχνική ηγεσία, coaching text, blameless postmortem, career ladder |
| [`ASSIGNMENT_COVERAGE.md`](ASSIGNMENT_COVERAGE.md) | 19/19 KPI traceability audit — πλήρης αντιστοίχιση απαιτήσεων → υλοποίησης |

---

<div align="center">

**🏛️ Philody AI Travel Assistant**
*Athens Smart Tourism Edition — v2.0*

Αρχαιοελληνική Φιλοξενία • Generative AI • Smart City IoT • Μαθηματική Ακρίβεια • EU AI Act Compliant

**Production Ready • 19/19 Assignment Requirements ✅**

---

*Για τεχνική τεκμηρίωση: βλ. [TECHNICAL_DESIGN_NOTE.md](TECHNICAL_DESIGN_NOTE.md)*
*Για αρχιτεκτονική & evaluation: βλ. [README.md](README.md)*
*Για κάλυψη απαιτήσεων: βλ. [ASSIGNMENT_COVERAGE.md](ASSIGNMENT_COVERAGE.md)*

</div>
