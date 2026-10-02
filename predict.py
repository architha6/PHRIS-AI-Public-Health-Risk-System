import re
import joblib


# ==========================================================
# PHRIS - PRIMARY DISEASE SIGNAL PREDICTION
# ==========================================================

MODEL_PATH = "models/disease_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LABEL_PATH = "models/label_binarizer.pkl"


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 65)
print("       PHRIS AI DISEASE SIGNAL PREDICTION")
print("=" * 65)

try:

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_binarizer = joblib.load(LABEL_PATH)

    print("\nModel loaded successfully.")

except Exception as error:

    print("\nERROR: Model could not be loaded.")
    print(error)

    print("\nPlease train the model first:")
    print("python src/train_model.py")

    exit()


# ==========================================================
# DISEASE NAMES
# ==========================================================

disease_names = {

    "AURI": "Acute Upper Respiratory Infection",

    "PN": "Pneumonia",

    "TB": "Pulmonary Tuberculosis",

    "COVID": "COVID-19"
}


# ==========================================================
# PREDICTION LOOP
# ==========================================================

while True:

    print("\n" + "-" * 65)

    text = input(
        "Enter symptoms or health-related text "
        "(type 'exit' to stop): "
    )

    if text.strip().lower() == "exit":

        print("\nPHRIS system closed.")
        break

    if not text.strip():

        print("Please enter some text.")
        continue


    # ======================================================
    # CLEAN TEXT
    # ======================================================

    cleaned_text = clean_text(text)


    # ======================================================
    # TF-IDF
    # ======================================================

    features = vectorizer.transform(
        [cleaned_text]
    )


    # ======================================================
    # SVM DECISION SCORES
    # ======================================================

    scores = model.decision_function(
        features
    )

    scores = scores[0]


    # ======================================================
    # CREATE DISEASE-SCORE LIST
    # ======================================================

    results = []

    for i, label in enumerate(
        label_binarizer.classes_
    ):

        disease = disease_names.get(
            label,
            label
        )

        score = float(scores[i])

        results.append(
            (label, disease, score)
        )


    # ======================================================
    # SORT BY SCORE
    # ======================================================

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )


    # ======================================================
    # PRIMARY DISEASE
    # ======================================================

    primary = results[0]

    primary_code = primary[0]
    primary_name = primary[1]
    primary_score = primary[2]


    # ======================================================
    # DISPLAY RESULT
    # ======================================================

    print("\n" + "=" * 65)
    print("                 PHRIS RESULT")
    print("=" * 65)

    print("\nInput:")
    print(text)

    print("\nPrimary Disease Signal:")
    print(">>>", primary_name)

    print(
        "Signal Score:",
        round(primary_score, 3)
    )


    # ======================================================
    # OTHER POSSIBLE DISEASE SIGNALS
    # ======================================================

    other_signals = []

    for result in results[1:]:

        if result[2] > 0:

            other_signals.append(result)


    if other_signals:

        print("\nOther Possible Disease Signals:")

        for result in other_signals:

            print(
                "•",
                result[1],
                "| Score:",
                round(result[2], 3)
            )

    else:

        print(
            "\nNo other strong disease signals detected."
        )


    # ======================================================
    # ALL MODEL SCORES
    # ======================================================

    print("\nModel Signal Summary:")

    for result in results:

        print(
            f"{result[1]:35} : "
            f"{result[2]:.3f}"
        )


    # ======================================================
    # DISCLAIMER
    # ======================================================

    print("\n" + "-" * 65)

    print(
        "The primary result represents the strongest "
        "AI-generated disease signal from the text."
    )

    print(
        "This system is for public-health risk-signal "
        "identification and is not a medical diagnosis."
    )

    print("-" * 65)