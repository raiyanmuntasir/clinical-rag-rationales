"""Sparse retrieval over the knowledge base, with an optional FAISS index.

Two query strategies are compared:

* **unguided**: the raw clinical note is the query (standard RAG).
* **classifier-guided**: the query is built from the classifier's prediction and
  the evidence features that drove it (decisive tokens and abnormal labs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# vocabulary bridging class labels and lab names to knowledge-base wording
CLASS_TERMS = {
    "diabetes": "diabetes hyperglycaemia HbA1c glucose",
    "ckd": "chronic kidney disease eGFR creatinine KDIGO",
    "anemia": "anaemia haemoglobin MCV iron deficiency",
    "hypothyroidism": "hypothyroidism TSH thyroid",
    "sepsis": "sepsis infection lactate white cell count",
    "no_acute_finding": "reference intervals routine follow-up within reference",
}
LAB_TERMS = {
    "glucose": "glucose", "hba1c": "HbA1c", "creatinine": "creatinine", "egfr": "eGFR",
    "hemoglobin": "haemoglobin", "mcv": "MCV mean corpuscular volume", "tsh": "TSH thyroid-stimulating hormone",
    "wbc": "white cell count leukocytosis", "lactate": "lactate",
}


class Retriever:
    def __init__(self, kb: pd.DataFrame, backend: str = "auto"):
        self.kb = kb.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        M = self.vec.fit_transform(self.kb.text).toarray().astype("float32")
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        self.M = M
        self.index = None
        if backend in ("auto", "faiss"):
            try:
                import faiss

                self.index = faiss.IndexFlatIP(M.shape[1])
                self.index.add(M)
            except ImportError:
                if backend == "faiss":
                    raise
        self.backend = "faiss" if self.index is not None else "numpy"

    def search(self, query: str, k: int = 3) -> pd.DataFrame:
        q = self.vec.transform([query]).toarray().astype("float32")
        q /= np.linalg.norm(q) + 1e-12
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            s = self.M @ q[0]
            idx = np.argsort(-s)[:k]
            scores = s[idx]
        out = self.kb.iloc[idx].copy()
        out["score"] = scores
        return out.reset_index(drop=True)


def guided_query(explanation: dict, abnormal_labs: list[str]) -> str:
    tokens = [f.split(":", 1)[1] for f, _ in explanation["evidence"] if f.startswith("text:")]
    labs = [LAB_TERMS[lab] for lab in abnormal_labs]
    return " ".join([CLASS_TERMS[explanation["predicted"]], *labs, *tokens])
