import unittest

from ragx import data, knowledge_base
from ragx.classifier import NoteLabClassifier
from ragx.faithfulness import check
from ragx.rationale import abnormal_labs, evidence_bundle, template_rationale
from ragx.retrieval import Retriever, guided_query


class TestRagx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data.generate(600, seed=1)
        cls.clf = NoteLabClassifier("fused").fit(cls.df.iloc[:450], cls.df.label.iloc[:450])
        cls.retr = Retriever(knowledge_base.load())

    def _bundle(self, i):
        row = self.df.iloc[i]
        ex = self.clf.explain(self.df.iloc[[i]])
        passages = self.retr.search(guided_query(ex, abnormal_labs(row)), 3)
        return row, evidence_bundle(row, ex, passages)

    def test_no_female_only_history_in_male_notes(self):
        males = self.df[self.df.note.str.contains(" male", regex=False) & ~self.df.note.str.contains("female")]
        self.assertFalse(males.note.str.contains("menstrual|gestational").any())

    def test_classifier_beats_chance(self):
        acc = (self.clf.predict(self.df.iloc[450:]) == self.df.label.iloc[450:]).mean()
        self.assertGreater(acc, 0.8)

    def test_explanation_has_positive_evidence(self):
        ex = self.clf.explain(self.df.iloc[[460]])
        self.assertTrue(ex["evidence"])
        self.assertTrue(all(v > 0 for _, v in ex["evidence"]))

    def test_retrieval_returns_k_ranked(self):
        r = self.retr.search("HbA1c glucose diabetes", 3)
        self.assertEqual(len(r), 3)
        self.assertIn("diabetes", set(r.topic))
        self.assertTrue((r.score.diff().dropna() <= 1e-9).all())

    def test_template_rationale_is_faithful(self):
        for i in range(450, 500):
            row, b = self._bundle(i)
            res = check(template_rationale(b), b, row.to_dict())
            self.assertEqual(res["problems"], [], res["problems"])

    def test_checker_catches_hallucinations(self):
        row, b = self._bundle(455)
        fake = ('Assessment based on "crushing chest pain" with HbA1c 99.9 % [KB99]. '
                f"Glucose {row.glucose + 50:.1f} mg/dL.")
        res = check(fake, b, row.to_dict())
        self.assertEqual(res["quotes_ungrounded"], 1)
        self.assertEqual(res["citations_invalid"], 1)
        self.assertEqual(res["values_wrong"], 2)


if __name__ == "__main__":
    unittest.main()
