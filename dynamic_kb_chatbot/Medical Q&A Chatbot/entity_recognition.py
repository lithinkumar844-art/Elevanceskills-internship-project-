"""
entity_recognition.py — Rule-based medical entity recogniser.

Detects: symptoms, diseases, treatments, drugs, body parts, tests.
Uses keyword lists + regex patterns — no external NLP models needed.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# ── Entity dictionaries ────────────────────────────────────────────────────────

SYMPTOMS = {
    "fever", "fatigue", "pain", "nausea", "vomiting", "diarrhea",
    "headache", "cough", "shortness of breath", "dyspnea", "chest pain",
    "rash", "swelling", "bruising", "bleeding", "dizziness", "weakness",
    "weight loss", "loss of appetite", "night sweats", "chills", "seizures",
    "confusion", "numbness", "tingling", "joint pain", "back pain",
    "abdominal pain", "muscle pain", "sore throat", "runny nose",
    "depression", "anxiety", "insomnia", "tremors", "palpitations",
    "itching", "jaundice", "anemia", "inflammation", "edema",
}

DISEASES = {
    "cancer", "leukemia", "diabetes", "hypertension", "alzheimer",
    "parkinson", "asthma", "pneumonia", "influenza", "hiv", "aids",
    "tuberculosis", "hepatitis", "arthritis", "lupus", "multiple sclerosis",
    "epilepsy", "stroke", "heart disease", "kidney disease", "liver disease",
    "crohn's disease", "celiac disease", "fibromyalgia", "glaucoma",
    "cataracts", "osteoporosis", "anemia", "thyroid", "hypothyroidism",
    "hyperthyroidism", "psoriasis", "eczema", "melanoma", "lymphoma",
    "autism", "schizophrenia", "bipolar", "adhd", "copd", "sepsis",
    "meningitis", "endometriosis", "scoliosis", "sleep apnea",
}

TREATMENTS = {
    "chemotherapy", "radiation", "surgery", "transplant", "dialysis",
    "immunotherapy", "physical therapy", "occupational therapy",
    "cognitive therapy", "psychotherapy", "hormone therapy",
    "stem cell transplant", "bone marrow transplant", "vaccine",
    "vaccination", "immunization", "biopsy", "screening", "monitoring",
}

DRUGS = {
    "aspirin", "ibuprofen", "acetaminophen", "penicillin", "amoxicillin",
    "metformin", "insulin", "lisinopril", "atorvastatin", "metoprolol",
    "prednisone", "warfarin", "levothyroxine", "omeprazole", "sertraline",
    "fluoxetine", "amlodipine", "simvastatin", "losartan", "gabapentin",
    "morphine", "oxycodone", "hydrocodone", "codeine", "tramadol",
    "antibiotic", "antiviral", "antifungal", "anticoagulant", "antidepressant",
    "antihistamine", "diuretic", "steroid", "beta-blocker", "statin",
}

BODY_PARTS = {
    "brain", "heart", "lung", "kidney", "liver", "stomach", "intestine",
    "colon", "pancreas", "thyroid", "bone marrow", "spleen", "bladder",
    "prostate", "ovary", "uterus", "breast", "skin", "blood", "spine",
    "joint", "muscle", "nerve", "lymph node", "adrenal", "pituitary",
}

TESTS = {
    "biopsy", "mri", "ct scan", "x-ray", "ultrasound", "blood test",
    "ecg", "ekg", "colonoscopy", "endoscopy", "mammogram", "pap smear",
    "blood pressure", "cholesterol", "glucose", "cbc", "urinalysis",
    "genetic test", "pcr", "culture", "biopsy",
}

ENTITY_SETS = {
    "symptom":    SYMPTOMS,
    "disease":    DISEASES,
    "treatment":  TREATMENTS,
    "drug":       DRUGS,
    "body_part":  BODY_PARTS,
    "test":       TESTS,
}

ENTITY_COLORS = {
    "symptom":   "#FF6B6B",
    "disease":   "#4ECDC4",
    "treatment": "#45B7D1",
    "drug":      "#96CEB4",
    "body_part": "#DDA0DD",
    "test":      "#F4A460",
}


@dataclass
class Entity:
    text:        str
    label:       str
    start:       int
    end:         int
    color:       str = field(init=False)

    def __post_init__(self):
        self.color = ENTITY_COLORS.get(self.label, "#CCCCCC")


def recognise_entities(text: str) -> list[Entity]:
    """
    Find medical entities in text using dictionary matching.
    Returns a list of Entity objects sorted by position.
    """
    text_lower = text.lower()
    found: list[Entity] = []
    seen_spans: set[tuple[int, int]] = set()

    for label, entity_set in ENTITY_SETS.items():
        for term in sorted(entity_set, key=len, reverse=True):   # longest first
            pattern = r"\b" + re.escape(term) + r"\b"
            for match in re.finditer(pattern, text_lower):
                span = (match.start(), match.end())
                # Skip if this span overlaps with an already-found entity
                if any(
                    not (span[1] <= s[0] or span[0] >= s[1])
                    for s in seen_spans
                ):
                    continue
                seen_spans.add(span)
                found.append(Entity(
                    text=text[match.start():match.end()],
                    label=label,
                    start=match.start(),
                    end=match.end(),
                ))

    return sorted(found, key=lambda e: e.start)


def highlight_entities(text: str, entities: list[Entity]) -> str:
    """
    Return HTML with entity spans highlighted using coloured badges.
    """
    if not entities:
        return text

    parts = []
    last = 0
    for ent in sorted(entities, key=lambda e: e.start):
        parts.append(text[last:ent.start])
        parts.append(
            f'<span style="background-color:{ent.color};padding:1px 4px;'
            f'border-radius:3px;font-size:0.85em;font-weight:600;color:#fff;" '
            f'title="{ent.label}">{ent.text}</span>'
        )
        last = ent.end
    parts.append(text[last:])
    return "".join(parts)


def entity_summary(entities: list[Entity]) -> dict[str, list[str]]:
    """Group entity texts by label."""
    summary: dict[str, list[str]] = {}
    for ent in entities:
        summary.setdefault(ent.label, []).append(ent.text)
    return {k: list(dict.fromkeys(v)) for k, v in summary.items()}  # deduplicate
