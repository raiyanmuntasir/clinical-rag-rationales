"""End-to-end proof-of-concept: ``python -m ragx``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from . import data, knowledge_base  # noqa: E402
from .classifier import NoteLabClassifier  # noqa: E402
from .faithfulness import check, summarise  # noqa: E402
from .rationale import abnormal_labs, evidence_bundle, generate  # noqa: E402
from .retrieval import Retriever, guided_query  # noqa: E402


def cv_modalities(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for mode in ("text", "labs", "fused"):
        scores = []
        for tr, te in skf.split(df, df.label):
            m = NoteLabClassifier(mode).fit(df.iloc[tr], df.label.iloc[tr])
            scores.append(f1_score(df.label.iloc[te], m.predict(df.iloc[te]), average="macro"))
        rows.append({"modality": mode, "macro_f1": np.mean(scores), "macro_f1_sd": np.std(scores, ddof=1)})
    return pd.DataFrame(rows)


def retrieval_metrics(retrieved: pd.DataFrame, label: str) -> tuple[float, float]:
    rel = (retrieved.topic == label).values
    return float(rel.mean()), float(rel[0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ragx")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--backend", choices=["template", "llm"], default="template")
    ap.add_argument("--n-llm", type=int, default=25, help="rationales to generate with the LLM backend")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    df = data.generate(args.n, args.seed)
    train, test = train_test_split(df, test_size=0.3, stratify=df.label, random_state=args.seed)
    test = test.reset_index(drop=True)

    # 1. classifier --------------------------------------------------------------
    modal = cv_modalities(train, args.seed)
    clf = NoteLabClassifier("fused").fit(train, train.label)
    pred = clf.predict(test)
    test_f1 = f1_score(test.label, pred, average="macro")
    report = classification_report(test.label, pred, digits=3)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(test.label, pred, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Fused classifier, held-out test (macro-F1 {test_f1:.3f})")
    fig.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2. retrieval ----------------------------------------------------------------
    kb = knowledge_base.load()
    retr = Retriever(kb)
    rows, bundles = [], []
    for i, row in test.iterrows():
        one = test.iloc[[i]]
        ex = clf.explain(one)
        abn = abnormal_labs(row)
        res = {"unguided": retr.search(row.note, args.k),
               "guided": retr.search(guided_query(ex, abn), args.k),
               "guided_oracle_class": retr.search(guided_query({**ex, "predicted": row.label}, abn), args.k)}
        r = {"patient_id": row.patient_id, "label": row.label, "predicted": ex["predicted"]}
        for name, ret in res.items():
            r[f"{name}_p@{args.k}"], r[f"{name}_hit@1"] = retrieval_metrics(ret, row.label)
        rows.append(r)
        bundles.append(evidence_bundle(row, ex, res["guided"]))
    rdf = pd.DataFrame(rows)
    rdf.to_csv(args.out / "retrieval_per_patient.csv", index=False)
    retrieval = pd.DataFrame({
        "strategy": ["unguided (raw note)", "classifier-guided", "guided with oracle class (upper bound)"],
        f"precision@{args.k}": [rdf[f"{s}_p@{args.k}"].mean() for s in ("unguided", "guided", "guided_oracle_class")],
        "hit@1": [rdf[f"{s}_hit@1"].mean() for s in ("unguided", "guided", "guided_oracle_class")],
    })

    # 3. rationales + faithfulness ----------------------------------------------------
    n_gen = len(bundles) if args.backend == "template" else min(args.n_llm, len(bundles))
    rationales, checks = [], []
    for b, (_, row) in zip(bundles[:n_gen], test.iloc[:n_gen].iterrows()):
        text = generate(b, args.backend)
        rationales.append(text)
        checks.append(check(text, b, row.to_dict()))
    faith = summarise(checks)
    correct = (rdf.label == rdf.predicted).values[:n_gen]

    examples = ["# Example rationales", "",
                f"Backend: `{args.backend}`. Retrieval: classifier-guided, k={args.k}, index={retr.backend}.", ""]
    # one correctly classified example per class, plus one misclassified patient
    picks, seen = [], set()
    for j, (b, ok) in enumerate(zip(bundles[:n_gen], correct)):
        if ok and b["predicted"] not in seen:
            picks.append(j)
            seen.add(b["predicted"])
    wrong = [j for j, ok in enumerate(correct) if not ok]
    picks += wrong[:1]
    for j in picks:
        b, text, c, ok = bundles[j], rationales[j], checks[j], correct[j]
        true = test.loc[test.patient_id == b["patient_id"], "label"].iloc[0]
        examples += [f"## {b['patient_id']} (true label: {true}; {'correct' if ok else 'MISCLASSIFIED'})", "",
                     f"> {b['note']}", "",
                     "Retrieved: " + ", ".join(f"{p['passage_id']} ({p['topic']})" for p in b["passages"]), "",
                     text, "", f"Faithfulness problems: {c['problems'] or 'none'}", ""]
    (args.out / f"examples_{args.backend}.md").write_text("\n".join(examples))

    summary = {
        "data": {"patients": len(df), "train": len(train), "test": len(test),
                 "class_counts": df.label.value_counts().to_dict()},
        "modalities_cv_macro_f1": modal.round(4).to_dict("records"),
        "test_macro_f1_fused": round(test_f1, 4),
        "retrieval": retrieval.round(4).to_dict("records"),
        "retrieval_index": retr.backend,
        "faithfulness": {"backend": args.backend, **{k: round(v, 4) for k, v in faith.items()}},
    }
    (args.out / f"summary_{args.backend}.json").write_text(json.dumps(summary, indent=2))
    (args.out / "classification_report.txt").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
