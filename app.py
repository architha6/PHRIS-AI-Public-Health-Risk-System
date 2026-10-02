# ==========================================================
# PHRIS - PUBLIC HEALTH RISK INTELLIGENCE SYSTEM
# STREAMLIT DASHBOARD
# ==========================================================

import streamlit as st
import joblib
import os
import sys


# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, SRC_DIR)


# ==========================================================
# IMPORT RISK ENGINE
# ==========================================================

from risk_engine import calculate_risk


# ==========================================================
# MODEL PATHS
# ==========================================================

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "disease_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

LABEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "label_binarizer.pkl"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    label_binarizer = joblib.load(LABEL_PATH)

    return model, vectorizer, label_binarizer


model, vectorizer, label_binarizer = load_models()


# ==========================================================
# DISEASE NAME CONVERSION
# ==========================================================

disease_names = {

    "AURI": "Acute Upper Respiratory Infection",

    "PN": "Pneumonia",

    "TB": "Pulmonary Tuberculosis",

    "COVID": "COVID-19"
}


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.lower()

    return text.strip()


# ==========================================================
# PAGE DESIGN
# ==========================================================

st.set_page_config(

    page_title="PHRIS",

    page_icon="🏥",

    layout="wide"
)


# ==========================================================
# HEADER
# ==========================================================

st.title("🏥 PHRIS")

st.subheader(
    "Public Health Risk Intelligence System"
)

st.write(
    "AI-based multilingual disease surveillance "
    "and early risk-signal identification."
)

st.info(
    "PHRIS provides digital risk signals from text data. "
    "It is not a medical diagnosis or a replacement for "
    "professional medical assessment."
)


# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown("## Enter Health-Related Text")

text = st.text_area(

    "Health-related text",

    placeholder=(
        "Example: I have fever, cough and "
        "difficulty breathing"
    ),

    height=150
)


# ==========================================================
# ANALYZE BUTTON
# ==========================================================

if st.button(
    "🔍 Analyze",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter health-related text."
        )

    else:

        # --------------------------------------------------
        # CLEAN TEXT
        # --------------------------------------------------

        cleaned_text = clean_text(text)


        # --------------------------------------------------
        # TF-IDF TRANSFORMATION
        # --------------------------------------------------

        features = vectorizer.transform(
            [cleaned_text]
        )


        # --------------------------------------------------
        # MODEL DECISION SCORES
        # --------------------------------------------------

        scores = model.decision_function(
            features
        )

        scores = scores[0]


        # --------------------------------------------------
        # CREATE DISEASE RESULTS
        # --------------------------------------------------

        results = []

        for label, score in zip(
            label_binarizer.classes_,
            scores
        ):

            disease = disease_names.get(
                label,
                label
            )

            results.append(
                (
                    label,
                    disease,
                    float(score)
                )
            )


        # --------------------------------------------------
        # SORT BY STRONGEST SIGNAL
        # --------------------------------------------------

        results.sort(
            key=lambda x: x[2],
            reverse=True
        )


        # ==================================================
        # PRIMARY DISEASE SIGNAL
        # ==================================================

        primary = results[0]

        primary_label = primary[0]

        primary_disease = primary[1]

        primary_score = primary[2]


        # ==================================================
        # DISPLAY PRIMARY SIGNAL
        # ==================================================

        st.markdown("## 🎯 Primary Disease Signal")

        st.success(
            primary_disease
        )

        st.write(
            f"Model signal score: "
            f"**{primary_score:.3f}**"
        )


        # ==================================================
        # OTHER POSSIBLE SIGNALS
        # ==================================================

        st.markdown(
            "## 🔎 Other Possible Disease Signals"
        )

        other_signals = [

            result

            for result in results[1:]

            if result[2] > 0
        ]


        if other_signals:

            for label, disease, score in other_signals:

                st.write(
                    f"• **{disease}** "
                    f"| Signal Score: `{score:.3f}`"
                )

        else:

            st.write(
                "No additional positive disease signals."
            )


        # ==================================================
        # MODEL SIGNAL SUMMARY
        # ==================================================

        st.markdown(
            "## 📊 Model Signal Summary"
        )

        for label, disease, score in results:

            st.write(
                f"**{disease}** : `{score:.3f}`"
            )


        # ==================================================
        # PREPARE DISEASE SCORES
        # ==================================================

        disease_scores = {}

        for label, disease, score in results:

            disease_scores[disease] = max(
                0,
                score
            )


        # ==================================================
        # PHRIS RISK ENGINE
        # ==================================================

        # Current prototype:
        # Symptom signal is activated when text
        # contains health-related symptom terms.

        symptom_words = [

            "fever",
            "cough",
            "cold",
            "breathing",
            "breath",
            "pain",
            "fatigue",
            "headache",
            "sore throat",
            "chest pain",
            "vomiting",
            "diarrhea"
        ]


        symptom_found = [

            word

            for word in symptom_words

            if word in cleaned_text
        ]


        if symptom_found:

            symptom_score = 1.0

        else:

            symptom_score = 0.0


        # --------------------------------------------------
        # CURRENT PLACEHOLDER SIGNALS
        # --------------------------------------------------

        sentiment_score = 0.0

        temporal_score = 0.0

        geographic_score = 0.0


        # ==================================================
        # CALCULATE RISK
        # ==================================================

        risk_score, risk_level = calculate_risk(

            disease_scores=disease_scores,

            symptom_score=symptom_score,

            sentiment_score=sentiment_score,

            temporal_score=temporal_score,

            geographic_score=geographic_score
        )


        # ==================================================
        # RISK ANALYSIS
        # ==================================================

        st.markdown("---")

        st.markdown(
            "## 🚨 PHRIS Risk Analysis"
        )


        # --------------------------------------------------
        # RISK SCORE
        # --------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.3f}"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        # --------------------------------------------------
        # RISK MESSAGE
        # --------------------------------------------------

        if risk_level == "LOW":

            st.info(
                "Low digital risk signal detected."
            )

        elif risk_level == "MODERATE":

            st.warning(
                "Moderate digital risk signal detected."
            )

        elif risk_level == "HIGH":

            st.warning(
                "High digital risk signal detected."
            )

        else:

            st.error(
                "Critical digital risk signal detected."
            )


        # ==================================================
        # SYMPTOM INFORMATION
        # ==================================================

        st.markdown(
            "## 🩺 Detected Symptom Signals"
        )

        if symptom_found:

            st.write(
                ", ".join(symptom_found)
            )

        else:

            st.write(
                "No predefined symptom keywords detected."
            )


        # ==================================================
        # RISK COMPONENTS
        # ==================================================

        st.markdown(
            "## ⚙️ Risk Components"
        )

        st.write(
            f"**Disease Signal:** "
            f"{max(disease_scores.values()):.3f}"
        )

        st.write(
            f"**Symptom Signal:** "
            f"{symptom_score:.3f}"
        )

        st.write(
            f"**Temporal Signal:** "
            f"{temporal_score:.3f}"
        )

        st.write(
            f"**Geographic Signal:** "
            f"{geographic_score:.3f}"
        )

        st.write(
            f"**Public Sentiment:** "
            f"{sentiment_score:.3f}"
        )


        # ==================================================
        # DISCLAIMER
        # ==================================================

        st.markdown("---")

        st.caption(
            "PHRIS is a research prototype for public-health "
            "surveillance and early digital risk-signal "
            "identification. Model outputs represent "
            "text-based signals and should not be interpreted "
            "as a medical diagnosis."
        )