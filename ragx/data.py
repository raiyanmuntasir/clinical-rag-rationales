"""Synthetic clinical notes + laboratory panels.

Each synthetic patient has one label from ``CLASSES``, a short free-text note
written from randomised templates (symptoms, history, negations and distractor
findings), and a lab panel. Notes are deliberately noisy: symptoms overlap
between conditions, some are missing, and some patients carry distractors, so
neither the text nor the labs alone is a perfect signal.

No real patient data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["diabetes", "ckd", "anemia", "hypothyroidism", "sepsis", "no_acute_finding"]

# lab name -> (unit, normal mean, normal sd)
LABS = {
    "glucose": ("mg/dL", 92, 10),
    "hba1c": ("%", 5.2, 0.3),
    "creatinine": ("mg/dL", 0.9, 0.15),
    "egfr": ("mL/min/1.73m2", 95, 12),
    "hemoglobin": ("g/dL", 14.0, 1.1),
    "mcv": ("fL", 90, 4),
    "tsh": ("mIU/L", 2.0, 0.8),
    "wbc": ("K/uL", 7.0, 1.5),
    "lactate": ("mmol/L", 1.2, 0.3),
}

# class -> {lab: (mean, sd)} shifts for affected patients
LAB_SHIFTS = {
    "diabetes": {"glucose": (190, 45), "hba1c": (8.3, 1.1)},
    "ckd": {"creatinine": (2.3, 0.5), "egfr": (38, 9)},
    "anemia": {"hemoglobin": (9.2, 1.0), "mcv": (74, 5)},
    "hypothyroidism": {"tsh": (11.0, 3.5)},
    "sepsis": {"wbc": (17.5, 4.0), "lactate": (3.4, 0.9)},
    "no_acute_finding": {},
}

SYMPTOMS = {
    "diabetes": ["polyuria", "polydipsia", "increased thirst", "frequent urination", "blurred vision",
                 "unintentional weight loss", "slow-healing foot ulcer"],
    "ckd": ["bilateral ankle edema", "decreased urine output", "fatigue", "nocturia", "pruritus",
            "foamy urine"],
    "anemia": ["fatigue", "pallor", "exertional dyspnea", "dizziness on standing", "palpitations",
               "brittle nails"],
    "hypothyroidism": ["cold intolerance", "weight gain", "constipation", "dry skin", "fatigue",
                       "hair thinning", "bradycardia"],
    "sepsis": ["fever", "rigors", "tachycardia", "hypotension", "confusion", "tachypnea"],
    "no_acute_finding": ["mild headache", "seasonal allergies", "routine follow-up", "minor back pain"],
}
HISTORY = {
    "diabetes": ["family history of diabetes", "obesity", "history of gestational diabetes"],
    "ckd": ["long-standing hypertension", "history of nephrolithiasis", "chronic NSAID use"],
    "anemia": ["heavy menstrual bleeding", "vegetarian diet", "history of GI bleeding"],
    "hypothyroidism": ["history of Hashimoto thyroiditis", "prior radioactive iodine therapy"],
    "sepsis": ["recent urinary tract infection", "indwelling catheter", "recent abdominal surgery"],
    "no_acute_finding": ["no significant past medical history"],
}
DISTRACTORS = ["mild headache", "seasonal allergies", "minor back pain", "insomnia", "knee pain"]
NEGATABLE = ["chest pain", "fever", "cough", "abdominal pain", "rash"]


FEMALE_ONLY = {"heavy menstrual bleeding", "history of gestational diabetes"}


def _note(rng, label: str) -> str:
    age, sex = int(rng.integers(25, 86)), str(rng.choice(["male", "female"]))
    k = int(rng.integers(1, 4))
    # 15% of notes are "silent": no condition-specific symptom is documented
    sx = [] if rng.random() < 0.15 else list(
        rng.choice(SYMPTOMS[label], size=min(k, len(SYMPTOMS[label])), replace=False))
    if rng.random() < 0.4 or not sx:
        sx.append(str(rng.choice(DISTRACTORS)))
    if rng.random() < 0.4:  # overlapping symptom from another condition
        other = str(rng.choice([c for c in CLASSES if c != label]))
        sx.append(str(rng.choice(SYMPTOMS[other])))
    sx = list(dict.fromkeys(sx))
    rng.shuffle(sx)
    hist = [h for h in HISTORY[label] if sex == "female" or h not in FEMALE_ONLY]
    hx = str(rng.choice(hist)) if rng.random() < 0.5 else "unremarkable past history"
    neg = [n for n in NEGATABLE if n not in sx and rng.random() < 0.3]
    opener = rng.choice([
        f"{age}-year-old {sex} presents with {', '.join(sx)}.",
        f"Patient is a {age} y/o {sex} reporting {', '.join(sx)}.",
        f"{age}yo {sex}, chief complaint: {', '.join(sx)}.",
    ])
    parts = [opener, f"PMH notable for {hx}."]
    if neg:
        parts.append(f"Denies {', '.join(neg)}.")
    return " ".join(parts)


def generate(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = np.array([0.2, 0.15, 0.18, 0.15, 0.12, 0.2])
    labels = rng.choice(CLASSES, size=n, p=probs / probs.sum())
    rows = []
    for i, label in enumerate(labels):
        labs = {}
        for name, (_unit, mu, sd) in LABS.items():
            if name in LAB_SHIFTS[label] and rng.random() > 0.25:  # 25% atypical: this lab looks normal
                m, s = LAB_SHIFTS[label][name]
                labs[name] = rng.normal(m, s)
            else:
                labs[name] = rng.normal(mu, sd)
        labs = {k: round(float(max(v, 0.01)), 2) for k, v in labs.items()}
        rows.append({"patient_id": f"P{i:05d}", "label": label, "note": _note(rng, label), **labs})
    return pd.DataFrame(rows)
