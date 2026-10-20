# Example rationales

Backend: `template`. Retrieval: classifier-guided, k=3, index=numpy.

## P00299 (true label: no_acute_finding; correct)

> 50yo male, chief complaint: seasonal allergies, routine follow-up, mild headache. PMH notable for unremarkable past history. Denies cough, rash.

Retrieved: KB22 (no_acute_finding), KB20 (no_acute_finding), KB21 (no_acute_finding)

**Assessment:** no acute finding (model probability 0.96).
**Note evidence:** the note documents "mild headache", "seasonal allergies", "routine follow-up" [KB22].
**Lab evidence:** all measured values fall within decision limits [KB22] [KB20] [KB21].

Faithfulness problems: none

## P01439 (true label: ckd; correct)

> Patient is a 72 y/o male reporting foamy urine, bilateral ankle edema. PMH notable for history of nephrolithiasis. Denies abdominal pain, rash.

Retrieved: KB05 (ckd), KB07 (ckd), KB20 (no_acute_finding)

**Assessment:** chronic kidney disease (model probability 0.99).
**Note evidence:** the note documents "bilateral ankle edema", "foamy urine", "history of nephrolithiasis" [KB05] [KB07].
**Lab evidence:** eGFR 26.65 mL/min/1.73m2 (reference > 90 mL/min/1.73m2) [KB05] [KB20].

Faithfulness problems: none

## P00689 (true label: anemia; correct)

> Patient is a 40 y/o female reporting brittle nails. PMH notable for unremarkable past history. Denies cough, rash.

Retrieved: KB11 (anemia), KB10 (anemia), KB12 (anemia)

**Assessment:** anaemia (model probability 0.98).
**Note evidence:** the note documents "brittle nails" [KB11].
**Lab evidence:** Haemoglobin 10.41 g/dL (reference 12-17.5 g/dL).

Faithfulness problems: none

## P00844 (true label: diabetes; correct)

> Patient is a 60 y/o female reporting frequent urination, polydipsia, blurred vision. PMH notable for family history of diabetes. Denies chest pain, abdominal pain.

Retrieved: KB02 (diabetes), KB01 (diabetes), KB20 (no_acute_finding)

**Assessment:** diabetes mellitus (model probability 1.00).
**Note evidence:** the note documents "polydipsia", "frequent urination", "blurred vision", "family history of diabetes" [KB02] [KB01].
**Lab evidence:** Glucose 148.33 mg/dL (reference fasting < 100 mg/dL) [KB02] [KB20].
**Lab evidence:** HbA1c 8.91 % (reference < 5.7 %) [KB01] [KB20].

Faithfulness problems: none

## P00204 (true label: hypothyroidism; correct)

> Patient is a 59 y/o female reporting insomnia, weight gain, constipation, polydipsia, cold intolerance. PMH notable for unremarkable past history. Denies cough, rash.

Retrieved: KB13 (hypothyroidism), KB14 (hypothyroidism), KB21 (no_acute_finding)

**Assessment:** hypothyroidism (model probability 1.00).
**Note evidence:** the note documents "cold intolerance", "weight gain", "constipation" [KB14].
**Lab evidence:** TSH 6.8 mIU/L (reference 0.4-4.0 mIU/L) [KB13] [KB21].

Faithfulness problems: none

## P00531 (true label: sepsis; correct)

> 69-year-old female presents with hypotension, rigors. PMH notable for unremarkable past history. Denies chest pain, cough, abdominal pain.

Retrieved: KB17 (sepsis), KB21 (no_acute_finding), KB18 (sepsis)

**Assessment:** sepsis (model probability 0.99).
**Note evidence:** the note documents "rigors", "hypotension" [KB18].
**Lab evidence:** WBC 15.63 K/uL (reference 4-11 K/uL) [KB17] [KB21].
**Lab evidence:** Lactate 3.6 mmol/L (reference < 2 mmol/L) [KB21].
**Other abnormal results (not explained by the assessment):** Creatinine 1.22 mg/dL.

Faithfulness problems: none

## P00934 (true label: hypothyroidism; MISCLASSIFIED)

> 71yo male, chief complaint: bradycardia, dry skin. PMH notable for prior radioactive iodine therapy.

Retrieved: KB17 (sepsis), KB21 (no_acute_finding), KB16 (sepsis)

**Assessment:** sepsis (model probability 0.45).
**Caveat:** low model confidence (0.45); consider the differential.
**Caveat:** no laboratory value supports this assessment; it rests on the note alone and should be confirmed.

Faithfulness problems: none
