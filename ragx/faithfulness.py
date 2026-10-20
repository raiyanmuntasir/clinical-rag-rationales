"""Automatic faithfulness checks for generated rationales.

The checks are backend-agnostic. For the template backend they verify the
implementation. For an LLM backend they measure hallucination.

* **value accuracy:** every lab value stated next to a lab name matches the record.
* **quote grounding:** every double-quoted phrase is a verbatim substring of the note.
* **citation validity:** every ``[KBxx]`` refers to a passage that was actually retrieved.
* **citation relevance:** cited passages belong to the asserted condition (or are reference ranges).
"""
from __future__ import annotations

import re

from .data import LABS
from .rationale import CITATION_RE, LAB_LABEL

NUM = r"(\d+(?:\.\d+)?)"


def check(rationale: str, bundle: dict, record: dict) -> dict:
    res = {"values_checked": 0, "values_wrong": 0, "quotes_checked": 0, "quotes_ungrounded": 0,
           "citations": 0, "citations_invalid": 0, "citations_irrelevant": 0, "problems": []}
    for lab in LABS:
        for m in re.finditer(rf"\b{re.escape(LAB_LABEL[lab])}\b[^\d\[]{{0,20}}?{NUM}", rationale, flags=re.I):
            res["values_checked"] += 1
            v = float(m.group(1))
            true = float(record[lab])
            if abs(v - true) > max(0.011, 0.01 * abs(true)):
                res["values_wrong"] += 1
                res["problems"].append(f"{LAB_LABEL[lab]} stated as {v}, record has {true}")
    note = bundle["note"].lower()
    for q in re.findall(r'"([^"]+)"', rationale):
        res["quotes_checked"] += 1
        if q.lower() not in note:
            res["quotes_ungrounded"] += 1
            res["problems"].append(f'quote "{q}" not found in note')
    retrieved = {p["passage_id"]: p["topic"] for p in bundle["passages"]}
    for c in CITATION_RE.findall(rationale):
        res["citations"] += 1
        if c not in retrieved:
            res["citations_invalid"] += 1
            res["problems"].append(f"citation [{c}] was not retrieved")
        elif retrieved[c] not in (bundle["predicted"], "no_acute_finding"):
            res["citations_irrelevant"] += 1
    return res


def summarise(results: list[dict]) -> dict:
    tot = {k: sum(r[k] for r in results) for k in results[0] if k != "problems"}

    def rate(bad, n):
        return 1 - tot[bad] / tot[n] if tot[n] else float("nan")

    return {
        "rationales": len(results),
        "value_accuracy": rate("values_wrong", "values_checked"),
        "quote_grounding": rate("quotes_ungrounded", "quotes_checked"),
        "citation_validity": rate("citations_invalid", "citations"),
        "citation_relevance": 1 - (tot["citations_irrelevant"] / (tot["citations"] - tot["citations_invalid"]))
        if tot["citations"] - tot["citations_invalid"] else float("nan"),
        "fully_faithful": sum(1 for r in results if not r["problems"]) / len(results),
        "mean_citations": tot["citations"] / len(results),
    }
