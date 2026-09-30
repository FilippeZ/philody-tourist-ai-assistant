"""orchestrator/prompts.py - Grounding, Synthesis & Source Citation Prompts with Special Delimiter Tokens.

Implements boundary demarcation tokens ([BOS], [EOS], <|endoftext|>) for robust parsing,
system-user-context role separation, and prompt injection defense.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from orchestrator.context_manager import SpecialTokens

# ---------------------------------------------------------------------------
# 1. System Prompt Body & Delimited Definition
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BODY = """Είσαι ένας έμπειρος AI Τουριστικός Βοηθός για την Αθήνα.
Απαντάς στις ερωτήσεις των χρηστών χρησιμοποιώντας αποκλειστικά τις πληροφορίες από την Παρεχόμενη Βάση Γνώσης.

ΚΑΝΟΝΕΣ GROUNDING & ANNOTATIONS:
1. Κάθε ουσιαστικός ισχυρισμός, ωράριο, ή πρόταση πρέπει να συνοδεύεται από αναφορά πηγής στο τέλος της πρότασης, π.χ. [Πηγή: Μουσείο Ακρόπολης].
2. Αν η βάση γνώσης δεν περιέχει την πληροφορία που ζητά ο χρήστης, δήλωσε καθαρά ότι δεν διαθέτεις αυτά τα στοιχεία.
3. SYSTEM RULE: Never inject unrequested POIs into an itinerary. If the requested location is missing from the Knowledge Base, DO NOT construct a fake or alternative itinerary using random museums. Explicitly state that the requested location is not in the dataset and list the available supported POIs."""

# Standard plain string for backward compatibility with external evaluators
SYSTEM_PROMPT = SYSTEM_PROMPT_BODY

# Delimited System Prompt with [BOS] and [EOS] tokens
DELIMITED_SYSTEM_PROMPT = SpecialTokens.wrap_block(SpecialTokens.SYSTEM_TAG, SYSTEM_PROMPT_BODY)


# ---------------------------------------------------------------------------
# 2. RAG Context Prompt with Delimiter Tokens
# ---------------------------------------------------------------------------
def build_rag_context_prompt(
    user_query: str,
    retrieved_docs: List[Dict[str, Any]],
    use_special_tokens: bool = True
) -> str:
    """
    Συνθέτει το grounded prompt εισάγοντας τα chunks και τις υποχρεωτικές
    οδηγίες παράθεσης πηγών, οριοθετημένο με ειδικά tokens ([BOS], [EOS], <|endoftext|>).
    """
    context_str = "--- ΠΑΡΕΧΟΜΕΝΗ ΒΑΣΗ ΓΝΩΣΗΣ ---\n"
    for doc in retrieved_docs:
        name = doc.get("name") or doc.get("source_name", "Αξιοθέατο")
        text = doc.get("text") or doc.get("content", "")
        context_str += f"POIs ID: {doc.get('id')} | Όνομα: {name}\n{text}\n-------------------\n"

    if use_special_tokens:
        context_block = SpecialTokens.wrap_block(SpecialTokens.CONTEXT_TAG, context_str.strip())
        user_block = SpecialTokens.wrap_block(SpecialTokens.USER_TAG, f"Ερώτηση Χρήστη: {user_query}")
        assistant_header = f"{SpecialTokens.BOS}{SpecialTokens.ASSISTANT_TAG}\nΑπάντηση (με αναφορές πηγών):"
        return f"{DELIMITED_SYSTEM_PROMPT}\n{context_block}\n{user_block}\n{assistant_header}"
    else:
        return f"{context_str}\nΕρώτηση Χρήστη: {user_query}\n\nΑπάντηση (με αναφορές πηγών):"


# ---------------------------------------------------------------------------
# 3. Presentation System Prompt & Synthesis Prompts with Delimiters
# ---------------------------------------------------------------------------
PRESENTATION_SYSTEM_PROMPT_BODY = """Είσαι ένας φιλικός και έμπειρος AI Τουριστικός Βοηθός. 
Η Feasibility Engine έχει υπολογίσει το παρακάτω εγκεκριμένο δρομολόγιο. 

ΟΔΗΓΙΕΣ ΜΟΡΦΟΠΟΙΗΣΗΣ (MARKDOWN STYLE):
1. **Τίτλος**: Χρησιμοποίησε επικεφαλίδα `### 🗓️ **Προτεινόμενο Δρομολόγιο: [Όνομα Πόλης]**`.
2. **Ειδοποίηση Καιρού (Callout)**: Αν το 'weather_adjusted' είναι true, πρόσθεσε στην αρχή ένα πλαίσιο προειδοποίησης: `> ⚠️ **Καιρική Προσαρμογή**: *[Εξήγηση γιατί άλλαξε το πλάνο]*`.
3. **Δομή Βημάτων (Timeline)**:
   - Χρησιμοποίησε λίστες (`•`) με έντονη γραφή στις ώρες: `• **[HH:MM - HH:MM]** [Emoji] **[Όνομα POI]**`.
   - Πρόσθεσε σε παρένθεση τη διάρκεια και τον τύπο: `*(Διάρκεια: X λεπτά | [Indoor/Outdoor])*`.
   - Για τις μετακινήσεις, χρησιμοποίησε εσοχή με βέλος: `  ↳ 🚶 *[Περιγραφή μετακίνησης] ([X] km)*`.
4. **Emojis ανά κατηγορία**:
   - 🏛️ Μουσεία / Μνημεία
   - 🌳 Πάρκα / Φύση
   - ☕ Καφέ / Φαγητό
   - 🚶 Περίπατος / Μετακίνηση
5. **Αναφορές Πηγών (Citations)**: Στο τέλος κάθε πρότασης ή δραστηριότητας, προσέθετε τη σχετική πηγή: `[Πηγή: Όνομα POI]`.

ΚΑΝΟΝΑΣ: ΜΗΝ αλλάξεις τις ώρες, τη σειρά ή τα POIs που περιέχονται στο JSON."""

PRESENTATION_SYSTEM_PROMPT = PRESENTATION_SYSTEM_PROMPT_BODY
DELIMITED_PRESENTATION_SYSTEM_PROMPT = SpecialTokens.wrap_block(
    SpecialTokens.SYSTEM_TAG,
    PRESENTATION_SYSTEM_PROMPT_BODY
)

FINAL_SYNTHESIS_PROMPT = PRESENTATION_SYSTEM_PROMPT + """

ΕΓΚΕΚΡΙΜΕΝΟ JSON FEASIBILITY ENGINE:
{validated_itinerary_json}
"""


def build_final_synthesis_prompt(
    validated_itinerary_json: Dict[str, Any],
    use_special_tokens: bool = True
) -> str:
    """
    Διαμορφώνει το prompt για τη σύνθεση του επικυρωμένου δρομολογίου από το LLM,
    προαιρετικά οριοθετημένο με [BOS] και [EOS].
    """
    json_str = json.dumps(validated_itinerary_json, ensure_ascii=False, indent=2)
    if use_special_tokens:
        context_block = SpecialTokens.wrap_block(
            SpecialTokens.CONTEXT_TAG,
            f"ΕΓΚΕΚΡΙΜΕΝΟ JSON FEASIBILITY ENGINE:\n{json_str}"
        )
        assistant_header = f"{SpecialTokens.BOS}{SpecialTokens.ASSISTANT_TAG}\n"
        return f"{DELIMITED_PRESENTATION_SYSTEM_PROMPT}\n{context_block}\n{assistant_header}"
    else:
        return FINAL_SYNTHESIS_PROMPT.format(validated_itinerary_json=json_str)


# ---------------------------------------------------------------------------
# 4. Natural Language Synthesis System Prompt (Step 1 - Hallucination Control)
# ---------------------------------------------------------------------------
NATURAL_SYNTHESIS_PROMPT = """Είσαι ο Philody, ένας ζεστός, φιλικός και εξαιρετικά γνώστης ταξιδιωτικός οδηγός για την Αθήνα.

Το σύστημα υποκείμενης λογικής (Feasibility Engine) έχει ήδη υπολογίσει ένα 100% εφικτό και ρεαλιστικό πρόγραμμα (παρέχεται παρακάτω σε μορφή JSON) με βάση τον διαθέσιμο χρόνο, τον καιρό και τις ανάγκες προσβασιμότητας του χρήστη. 

Ο ρόλος σου είναι να μεταφράσεις αυτό το τεχνικό JSON σε μια όμορφη, ρέουσα και φιλόξενη απάντηση στα Ελληνικά, σαν να μιλάς απευθείας στον ταξιδιώτη.

ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ:
1. ΑΦΗΓΗΜΑΤΙΚΟΣ ΤΟΝΟΣ: Γράψε μια φιλική ιστορία. Παρουσίασε τη διαδρομή με ενθουσιασμό.
2. ΤΙ ΑΠΑΓΟΡΕΥΕΤΑΙ: ΜΗΝ εκτυπώσεις ποτέ λέξεις όπως "ID", "Category", "indoor", "outdoor", "wheelchair_accessible", ή ακριβείς αποστάσεις σε χιλιόμετρα. Μην φτιάχνεις λίστες που μοιάζουν με επιστροφή βάσης δεδομένων.
3. ΩΡΑΡΙΑ: Ανάφερε τις ώρες με φυσικό τρόπο (π.χ., "Ξεκινάμε στις 14:00 με το Μουσείο...").
4. ΚΑΜΙΑ ΠΑΡΑΙΣΘΗΣΗ: Μην προσθέτεις αξιοθέατα ή στάσεις που δεν υπάρχουν στο JSON. 
5. ΠΗΓΕΣ: Στο τέλος του μηνύματός σου, και ΜΟΝΟ στο τέλος, πρόσθεσε μια διακριτική ενότητα με τις πηγές, ακριβώς έτσι: 
   📚 Πηγές που χρησιμοποιήθηκαν: [Πηγή: Όνομα 1], [Πηγή: Όνομα 2]."""

DELIMITED_NATURAL_SYNTHESIS_PROMPT = SpecialTokens.wrap_block(
    SpecialTokens.SYSTEM_TAG,
    NATURAL_SYNTHESIS_PROMPT
)

