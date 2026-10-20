"""Grounded rationale generation.

Two backends share the same evidence bundle:

* ``template`` (default): deterministic, offline, and grounded by construction.
  Every number is copied from the record, every quote is a substring of the
  note, and every citation refers to a retrieved passage.
* ``llm``: sends the same evidence bundle to an LLM (Anthropic API, used when
  ``ANTHROPIC_API_KEY`` is set) with instructions to quote and cite. Its output is
  checked by :mod:`ragx.faithfulness`, which matters because LLMs can hallucinate.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from .data import HISTORY, LAB_SHIFTS, LABS, SYMPTOMS
from .retrieval import LAB_TERMS

# decision limits (independent of the generator's distributions)
REFERENCE = {
    "glucose": ("high", 125, "fasting < 100 mg/dL"),
    "hba1c": ("high", 6.5, "< 5.7 %"),
    "creatinine": ("high", 1.2, "0.6-1.2 mg/dL"),
    "egfr": ("low", 60, "> 90 mL/min/1.73m2"),
    "hemoglobin": ("low", 12.0, "12-17.5 g/dL"),
    "mcv": ("low", 80, "80-100 fL"),
    "tsh": ("high", 4.5, "0.4-4.0 mIU/L"),
    "wbc": ("high", 12.0, "4-11 K/uL"),
    "lactate": ("high", 2.0, "< 2 mmol/L"),
}
PRETTY_CLASS = {
    "diabetes": "diabetes mellitus", "ckd": "chronic kidney disease", "anemia": "anaemia",
    "hypothyroidism": "hypothyroidism", "sepsis": "sepsis", "no_acute_finding": "no acute finding",
}
LAB_LABEL = {"glucose": "Glucose", "hba1c": "HbA1c", "creatinine": "Creatinine", "egfr": "eGFR",
             "hemoglobin": "Haemoglobin", "mcv": "MCV", "tsh": "TSH", "wbc": "WBC", "lactate": "Lactate"}


def abnormal_labs(row: pd.Series) -> list[str]:
    out = []
    for lab, (direction, limit, _) in REFERENCE.items():
        v = row[lab]
        if (direction == "high" and v >= limit) or (direction == "low" and v < limit):
            out.append(lab)
    return out


def _note_phrases(note: str, label: str) -> list[str]:
    vocab = SYMPTOMS.get(label, []) + HISTORY.get(label, [])
    low = note.lower()
    return [p for p in vocab if p.lower() in low]


def _cite(passages: pd.DataFrame, words: str) -> list[str]:
    keys = [w.lower() for w in words.split()]
    return [r.passage_id for r in passages.itertuples() if any(k in r.text.lower() for k in keys)]


def evidence_bundle(row: pd.Series, explanation: dict, passages: pd.DataFrame) -> dict:
    pred = explanation["predicted"]
    abn = abnormal_labs(row)
    relevant = [lab for lab in abn if lab in LAB_SHIFTS.get(pred, {})]
    other = [lab for lab in abn if lab not in relevant]
    return {
        "patient_id": row["patient_id"],
        "predicted": pred,
        "probability": explanation["probability"],
        "note": row["note"],
        "note_phrases": _note_phrases(row["note"], pred),
        "labs": {lab: (row[lab], LABS[lab][0], REFERENCE[lab][2]) for lab in relevant},
        "other_abnormal": {lab: (row[lab], LABS[lab][0], REFERENCE[lab][2]) for lab in other},
        "passages": passages[["passage_id", "topic", "text"]].to_dict("records"),
    }


def template_rationale(b: dict) -> str:
    passages = pd.DataFrame(b["passages"])
    lines = [f"**Assessment:** {PRETTY_CLASS[b['predicted']]} (model probability {b['probability']:.2f})."]
    if b["note_phrases"]:
        quoted = ", ".join(f'"{p}"' for p in b["note_phrases"])
        cites = _cite(passages, " ".join(b["note_phrases"]))
        lines.append(f"**Note evidence:** the note documents {quoted}"
                     + (f" {' '.join(f'[{c}]' for c in cites)}" if cites else "") + ".")
    for lab, (v, unit, ref) in b["labs"].items():
        cites = _cite(passages, LAB_TERMS[lab].split()[0])
        lines.append(f"**Lab evidence:** {LAB_LABEL[lab]} {v:g} {unit} (reference {ref})"
                     + (f" {' '.join(f'[{c}]' for c in cites)}" if cites else "") + ".")
    if b["predicted"] == "no_acute_finding" and not b["other_abnormal"]:
        cites = [p["passage_id"] for p in b["passages"] if p["topic"] == "no_acute_finding"]
        lines.append("**Lab evidence:** all measured values fall within decision limits"
                     + (f" {' '.join(f'[{c}]' for c in cites)}" if cites else "") + ".")
    if b["other_abnormal"]:
        extra = "; ".join(f"{LAB_LABEL[lab]} {v:g} {unit}" for lab, (v, unit, _) in b["other_abnormal"].items())
        lines.append(f"**Other abnormal results (not explained by the assessment):** {extra}.")
    if b["probability"] < 0.6:
        lines.append(f"**Caveat:** low model confidence ({b['probability']:.2f}); consider the differential.")
    if not b["labs"] and b["predicted"] != "no_acute_finding":
        lines.append("**Caveat:** no laboratory value supports this assessment; it rests on the note alone "
                     "and should be confirmed.")
    return "\n".join(lines)


def llm_rationale(b: dict, model: str | None = None) -> str:
    import anthropic  # optional dependency

    model = model or os.environ.get("RAGX_MODEL", "claude-sonnet-5-5")
    kb = "\n".join(f"[{p['passage_id']}] {p['text']}" for p in b["passages"])
    labs = "\n".join(f"- {LAB_LABEL[k]}: {v} {u} (reference {r})" for k, (v, u, r) in
                     {**b["labs"], **b["other_abnormal"]}.items()) or "- none outside decision limits"
    prompt = (
        "You are writing a short clinical rationale for a model's assessment. Use ONLY the evidence below.\n"
        "Rules: quote note phrases verbatim in double quotes; copy lab values exactly; cite knowledge passages "
        "as [KBxx] using only the IDs listed; state uncertainty if evidence is weak. Max 120 words.\n\n"
        f"Assessment: {PRETTY_CLASS[b['predicted']]} (probability {b['probability']:.2f})\n"
        f"Clinical note: {b['note']}\nLaboratory values:\n{labs}\nKnowledge passages:\n{kb}\n"
    )
    msg = anthropic.Anthropic().messages.create(model=model, max_tokens=400,
                                                messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


def generate(b: dict, backend: str = "template") -> str:
    if backend == "llm":
        return llm_rationale(b)
    return template_rationale(b)


CITATION_RE = re.compile(r"\[(KB\d+)\]")
