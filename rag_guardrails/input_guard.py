"""
Input guardrail using guardrails-ai.
Blocks queries containing PII and confirms the query is clinically in-scope.
"""
from guardrails import Guard
from guardrails.hub import DetectPII

# Guard that fails if SSNs, phone numbers, emails, etc. are present in the query
pii_guard = Guard().use(
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "PERSON"],
        on_fail="exception",
    )
)

CLINICAL_KEYWORDS = [
    # =========================
    # General Clinical Terms
    # =========================
    "clinical", "patient", "doctor", "physician", "hospital",
    "medicine", "medication", "drug", "pharmaceutical",
    "therapy", "treatment", "management", "care",
    "diagnosis", "diagnose", "diagnostic", "prognosis",
    "disease", "condition", "disorder", "syndrome",
    "symptom", "sign", "finding", "complaint",
    "screening", "screen", "prevention", "preventive",
    "risk factor", "complication", "comorbidity",
    "history", "family history", "medical history",
    "follow-up", "guideline", "recommendation",
    "consultation", "clinical practice",

    "cbc", "cmp", "bmp", "lft", "kft",
    "ecg", "ekg", "echo", "ct", "mri",
    "pet", "usg", "cxr", "abg", "spo2",
    "bp", "hr", "rr", "temp",
    "fbs", "rbs", "hba1c", "ogtt",
    "tsh", "t3", "t4",
    "hdl", "ldl", "bun", "egfr",
    "pt", "inr", "aptt",
    "tb", "copd", "ckd", "aki",
    "cad", "chf", "mi", "uti",
    "uri", "icu", "opd", "er", "ed",

    # =========================
    # Vital Signs
    # =========================
    "blood pressure",
    "heart rate",
    "pulse",
    "respiratory rate",
    "oxygen saturation",
    "spo2",
    "temperature",
    "fever",
    "hypotension",
    "hypertension",
    "tachycardia",
    "bradycardia",
    "tachypnea",
    "bradypnea",

    # =========================
    # Symptoms
    # =========================
    "pain",
    "chest pain",
    "headache",
    "migraine",
    "dizziness",
    "fatigue",
    "weakness",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "cough",
    "shortness of breath",
    "dyspnea",
    "wheezing",
    "sore throat",
    "runny nose",
    "congestion",
    "abdominal pain",
    "back pain",
    "joint pain",
    "swelling",
    "rash",
    "itching",
    "bleeding",
    "weight loss",
    "weight gain",
    "loss of appetite",
    "anorexia",
    "palpitations",
    "syncope",
    "seizure",
    "confusion",

    # =========================
    # Diseases
    # =========================
    "diabetes",
    "diabetic",
    "type 1 diabetes",
    "type 2 diabetes",
    "hypertension",
    "hypotension",
    "heart disease",
    "coronary artery disease",
    "heart failure",
    "stroke",
    "myocardial infarction",
    "angina",
    "asthma",
    "copd",
    "pneumonia",
    "tuberculosis",
    "tb",
    "covid",
    "covid-19",
    "influenza",
    "hepatitis",
    "cirrhosis",
    "kidney disease",
    "ckd",
    "aki",
    "liver disease",
    "anemia",
    "thyroid disease",
    "hypothyroidism",
    "hyperthyroidism",
    "obesity",
    "cancer",
    "malignancy",
    "tumor",
    "leukemia",
    "lymphoma",
    "arthritis",
    "osteoarthritis",
    "rheumatoid arthritis",
    "osteoporosis",
    "depression",
    "anxiety",
    "epilepsy",
    "parkinson",
    "alzheimer",
    "dementia",
    "hiv",
    "aids",
    "malaria",
    "dengue",
    "typhoid",
    "sepsis",

    # =========================
    # Laboratory Tests
    # =========================
    "blood test",
    "urine test",
    "urinalysis",
    "cbc",
    "complete blood count",
    "hemoglobin",
    "hematocrit",
    "platelet",
    "wbc",
    "rbc",
    "esr",
    "crp",
    "electrolytes",
    "sodium",
    "potassium",
    "chloride",
    "calcium",
    "magnesium",
    "phosphate",
    "blood glucose",
    "fasting blood sugar",
    "fbs",
    "random blood sugar",
    "rbs",
    "hba1c",
    "oral glucose tolerance test",
    "ogtt",
    "creatinine",
    "bun",
    "egfr",
    "liver function test",
    "lft",
    "kidney function test",
    "kft",
    "bilirubin",
    "albumin",
    "ast",
    "alt",
    "alp",
    "lipid profile",
    "cholesterol",
    "hdl",
    "ldl",
    "triglycerides",
    "thyroid function test",
    "tsh",
    "t3",
    "t4",
    "vitamin d",
    "vitamin b12",
    "iron studies",
    "ferritin",
    "d dimer",
    "troponin",
    "bnp",
    "procalcitonin",
    "lactate",
    "coagulation",
    "pt",
    "inr",
    "aptt",
    "abg",
    "arterial blood gas",
    "culture",
    "blood culture",
    "urine culture",
    "sputum culture",

    # =========================
    # Imaging
    # =========================
    "x ray",
    "x-ray",
    "radiograph",
    "ct",
    "ct scan",
    "mri",
    "ultrasound",
    "usg",
    "doppler",
    "pet scan",
    "pet",
    "echocardiogram",
    "echo",
    "ecg",
    "ekg",
    "electrocardiogram",
    "stress test",
    "angiography",
    "angiogram",
    "mammography",
    "bone scan",
    "fluoroscopy",

    # =========================
    # Procedures
    # =========================
    "biopsy",
    "surgery",
    "operation",
    "transplant",
    "dialysis",
    "intubation",
    "ventilation",
    "catheter",
    "endoscopy",
    "colonoscopy",
    "bronchoscopy",
    "laparoscopy",
    "thoracentesis",
    "paracentesis",
    "lumbar puncture",
    "blood transfusion",

    # =========================
    # Medications
    # =========================
    "antibiotic",
    "antiviral",
    "antifungal",
    "antiparasitic",
    "vaccine",
    "vaccination",
    "insulin",
    "metformin",
    "statin",
    "aspirin",
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "antihypertensive",
    "beta blocker",
    "ace inhibitor",
    "arb",
    "calcium channel blocker",
    "diuretic",
    "steroid",
    "corticosteroid",
    "chemotherapy",
    "immunotherapy",
    "anticoagulant",
    "heparin",
    "warfarin",
    "analgesic",

    # =========================
    # Clinical Measurements
    # =========================
    "bmi",
    "body mass index",
    "blood sugar",
    "glucose",
    "oxygen",
    "spo2",
    "pulse",
    "weight",
    "height",

    # =========================
    # Medical Specialties
    # =========================
    "cardiology",
    "neurology",
    "oncology",
    "endocrinology",
    "nephrology",
    "pulmonology",
    "gastroenterology",
    "hematology",
    "infectious disease",
    "dermatology",
    "psychiatry",
    "radiology",
    "pathology",
    "urology",
    "gynecology",
    "obstetrics",
    "pediatrics",
    "orthopedics",
    "ophthalmology",
    "ent",

    # =========================
    # Clinical Documents
    # =========================
    "prescription",
    "medical record",
    "discharge summary",
    "progress note",
    "case report",
    "lab report",
    "radiology report",
    "pathology report",
]


def check_input(query: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    is_safe=False means the query should be blocked before it reaches retrieval.
    """
    # 1. PII check via guardrails-ai
    try:
        pii_guard.validate(query)
    except Exception:
        return False, "Query appears to contain personal identifying information (PII)."

    # 2. Scope check — keep it a clinical RAG, not a general chatbot
    if not any(kw in query.lower() for kw in CLINICAL_KEYWORDS):
        return False, "Query does not appear to be a clinical question."

    return True, ""