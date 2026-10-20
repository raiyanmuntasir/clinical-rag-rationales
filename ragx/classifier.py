"""Multimodal (note text + labs) classifier whose decisions can be decomposed into
per-feature evidence. That decomposition is what guides retrieval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import LABS


class NoteLabClassifier:
    """TF-IDF(note) ⊕ standardised labs → multinomial logistic regression.

    ``mode`` selects the modality: ``"text"``, ``"labs"`` or ``"fused"``.
    A linear model is used on purpose: each prediction decomposes exactly into
    token and lab contributions (coefficient × feature value).
    """

    def __init__(self, mode: str = "fused", C: float = 4.0, lab_weight: float = 1.0):
        self.mode, self.C, self.lab_weight = mode, C, lab_weight

    def _features(self, df: pd.DataFrame, fit: bool = False):
        blocks = []
        if self.mode in ("text", "fused"):
            if fit:
                self.vec_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
                blocks.append(self.vec_.fit_transform(df.note))
            else:
                blocks.append(self.vec_.transform(df.note))
        if self.mode in ("labs", "fused"):
            L = np.log(df[list(LABS)].values)
            if fit:
                self.scaler_ = StandardScaler().fit(L)
            blocks.append(csr_matrix(self.scaler_.transform(L) * self.lab_weight))
        return hstack(blocks).tocsr()

    def fit(self, df: pd.DataFrame, y) -> "NoteLabClassifier":
        X = self._features(df, fit=True)
        self.clf_ = LogisticRegression(C=self.C, max_iter=5000).fit(X, y)
        self.classes_ = self.clf_.classes_
        names = []
        if self.mode in ("text", "fused"):
            names += [f"text:{t}" for t in self.vec_.get_feature_names_out()]
        if self.mode in ("labs", "fused"):
            names += [f"lab:{n}" for n in LABS]
        self.feature_names_ = np.array(names)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._features(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(df).argmax(1)]

    def explain(self, row: pd.DataFrame, top: int = 6) -> dict:
        """Evidence for the predicted class of a single patient (1-row DataFrame)."""
        x = self._features(row)
        p = self.clf_.predict_proba(x)[0]
        k = int(p.argmax())
        contrib = x.multiply(self.clf_.coef_[k]).toarray()[0]
        order = np.argsort(-contrib)
        pos = [(self.feature_names_[i], float(contrib[i])) for i in order[:top] if contrib[i] > 0]
        return {
            "predicted": str(self.classes_[k]),
            "probability": float(p[k]),
            "probabilities": {str(c): float(v) for c, v in zip(self.classes_, p)},
            "evidence": pos,
        }
