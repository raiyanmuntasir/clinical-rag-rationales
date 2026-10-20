"""A small curated knowledge base of clinical-criteria passages.

The passages are short paraphrased summaries of widely taught diagnostic
criteria (ADA, KDIGO, WHO, Sepsis-3 and standard reference intervals), written
for this proof of concept. They are illustrative, not authoritative, and should
not be used for clinical decisions. Distractor passages about unrelated topics
make retrieval non-trivial.
"""
from __future__ import annotations

import pandas as pd

PASSAGES = [
    # diabetes
    ("KB01", "diabetes", "Diabetes mellitus can be diagnosed when HbA1c is 6.5% or higher, reflecting average glycaemia over roughly three months (ADA criteria)."),
    ("KB02", "diabetes", "A random plasma glucose of 200 mg/dL or more in a patient with classic symptoms of hyperglycaemia such as polyuria and polydipsia supports a diagnosis of diabetes."),
    ("KB03", "diabetes", "Classic hyperglycaemic symptoms include increased thirst, frequent urination, blurred vision and unintentional weight loss; poor wound healing such as foot ulcers is common in long-standing diabetes."),
    ("KB04", "diabetes", "Family history of diabetes, obesity and prior gestational diabetes are major risk factors for type 2 diabetes."),
    # ckd
    ("KB05", "ckd", "Chronic kidney disease is defined by an estimated glomerular filtration rate (eGFR) below 60 mL/min/1.73m2 or markers of kidney damage persisting for more than three months (KDIGO)."),
    ("KB06", "ckd", "KDIGO GFR categories: G3a 45-59, G3b 30-44, G4 15-29 and G5 below 15 mL/min/1.73m2. Serum creatinine rises as filtration declines."),
    ("KB07", "ckd", "Signs of advancing kidney disease include peripheral oedema, nocturia, foamy urine (proteinuria), pruritus and reduced urine output."),
    ("KB08", "ckd", "Long-standing hypertension, diabetes and chronic NSAID exposure are common causes of progressive kidney damage."),
    # anemia
    ("KB09", "anemia", "Anaemia is commonly defined by haemoglobin below 13 g/dL in adult men and below 12 g/dL in non-pregnant adult women (WHO)."),
    ("KB10", "anemia", "A low mean corpuscular volume (MCV below 80 fL) indicates microcytic anaemia, most often due to iron deficiency."),
    ("KB11", "anemia", "Symptoms of anaemia include fatigue, pallor, exertional dyspnoea, palpitations and dizziness; brittle nails may accompany iron deficiency."),
    ("KB12", "anemia", "Chronic blood loss, including heavy menstrual bleeding and gastrointestinal bleeding, and low dietary iron intake are leading causes of iron-deficiency anaemia."),
    # hypothyroidism
    ("KB13", "hypothyroidism", "Primary hypothyroidism is suggested by an elevated thyroid-stimulating hormone (TSH) above the reference interval, typically above about 4.5 mIU/L."),
    ("KB14", "hypothyroidism", "Hypothyroid symptoms include cold intolerance, weight gain, constipation, dry skin, hair thinning, fatigue and bradycardia."),
    ("KB15", "hypothyroidism", "Autoimmune (Hashimoto) thyroiditis and prior radioactive iodine therapy are frequent causes of primary hypothyroidism."),
    # sepsis
    ("KB16", "sepsis", "Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection (Sepsis-3); serum lactate above 2 mmol/L indicates tissue hypoperfusion."),
    ("KB17", "sepsis", "Leukocytosis with a white cell count above 12 K/uL, fever, tachycardia and tachypnoea are signs of a systemic inflammatory response to infection."),
    ("KB18", "sepsis", "Hypotension, rigors and new confusion in a patient with a suspected infection should prompt urgent evaluation for sepsis."),
    ("KB19", "sepsis", "Urinary tract infection, indwelling catheters and recent surgery are common sources of infection leading to sepsis."),
    # no acute finding / reference
    ("KB20", "no_acute_finding", "Typical adult reference intervals: fasting glucose 70-99 mg/dL, HbA1c below 5.7%, creatinine 0.6-1.2 mg/dL and eGFR above 90 mL/min/1.73m2."),
    ("KB21", "no_acute_finding", "Typical adult reference intervals: haemoglobin 12-17.5 g/dL, MCV 80-100 fL, TSH 0.4-4.0 mIU/L, white cell count 4-11 K/uL and lactate below 2 mmol/L."),
    ("KB22", "no_acute_finding", "Routine follow-up visits with mild, self-limiting complaints and laboratory values within reference intervals do not by themselves indicate acute disease."),
    # distractors
    ("KB23", "other", "Migraine is a primary headache disorder characterised by recurrent unilateral throbbing headache, often with photophobia and nausea."),
    ("KB24", "other", "Seasonal allergic rhinitis presents with sneezing, nasal congestion and itchy eyes and is managed with antihistamines or intranasal corticosteroids."),
    ("KB25", "other", "Mechanical low back pain is common, usually self-limiting, and initial management includes activity and simple analgesia."),
    ("KB26", "other", "Essential hypertension is diagnosed after repeated blood pressure readings at or above 130/80 mmHg under current US guidelines."),
    ("KB27", "other", "Community-acquired pneumonia presents with cough, fever and focal chest signs and is confirmed with chest radiography."),
    ("KB28", "other", "Osteoarthritis of the knee causes activity-related pain and stiffness and is more common with age and obesity."),
]


def load() -> pd.DataFrame:
    return pd.DataFrame(PASSAGES, columns=["passage_id", "topic", "text"])
