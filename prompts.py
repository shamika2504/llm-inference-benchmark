"""20 short synthetic healthcare prompts. All content is made up; no real patient data."""

MEDICARE_QUESTIONS = [
    "What is the difference between Medicare Part A and Part B?",
    "Does Medicare Part B cover annual flu shots?",
    "When is the Medicare initial enrollment period for someone turning 65?",
    "What does Medicare Part D cover?",
    "Explain what a Medicare Advantage plan is in simple terms.",
    "Does Original Medicare cover routine dental cleanings?",
    "What is a Medigap policy and who might need one?",
    "How does the Part B late enrollment penalty work?",
    "Is skilled nursing facility care covered by Medicare Part A?",
    "What preventive screenings does Medicare cover for free?",
]

CLINICAL_NOTES = [
    "68-year-old male presents with 3 days of productive cough and fever of 101F. "
    "Lungs with right lower lobe crackles. Chest X-ray shows consolidation. "
    "Started on oral antibiotics, follow up in 1 week.",
    "72-year-old female with type 2 diabetes for routine follow-up. A1c 7.8%, up from 7.1%. "
    "Reports missing evening metformin doses. Counseled on adherence, recheck A1c in 3 months.",
    "55-year-old male with chest tightness on exertion for 2 weeks, relieved by rest. "
    "ECG normal sinus rhythm. Referred for outpatient stress test, started aspirin.",
    "80-year-old female after a fall at home, no loss of consciousness. Right wrist swelling, "
    "X-ray shows distal radius fracture. Splinted, referred to orthopedics.",
    "66-year-old male with hypertension, BP today 152/94 on lisinopril 10 mg. "
    "No side effects. Increased lisinopril to 20 mg, home BP log requested.",
    "70-year-old female with 2 days of burning urination and frequency, no fever. "
    "Urinalysis positive for nitrites. Started on a 5-day antibiotic course.",
    "62-year-old male with chronic low back pain, worse after yard work. No red flags. "
    "Recommended physical therapy, heat and acetaminophen as needed.",
    "75-year-old female with mild memory complaints per daughter. MoCA score 24/30. "
    "Labs ordered including B12 and TSH, follow up in 4 weeks.",
    "69-year-old male with COPD reporting increased shortness of breath for 5 days. "
    "O2 sat 92% on room air. Prednisone burst given, inhaler technique reviewed.",
    "64-year-old female for annual wellness visit. Up to date on mammogram, due for "
    "colonoscopy. Shingles vaccine administered today.",
]

PROMPTS = MEDICARE_QUESTIONS + [f"Summarize this short clinical note:\n{n}" for n in CLINICAL_NOTES]

assert len(PROMPTS) == 20
