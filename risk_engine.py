# ==========================================================
# PHRIS - PUBLIC HEALTH RISK INTELLIGENCE SYSTEM
# RISK DETECTION ENGINE
# ==========================================================

"""
PHRIS Risk Detection Engine

Risk Formula:

R = w1D + w2S + w3T + w4G + w5P

D = Disease signal
S = Symptom signal
T = Temporal signal
G = Geographic signal
P = Public sentiment signal

This module generates a configurable digital risk indicator.
It is NOT a medical diagnosis or clinical risk assessment.
"""


# ==========================================================
# RISK CALCULATION FUNCTION
# ==========================================================

def calculate_risk(
    disease_scores,
    symptom_score,
    sentiment_score=0,
    temporal_score=0,
    geographic_score=0
):
    """
    Calculate the PHRIS risk score.

    Parameters
    ----------
    disease_scores : dict
        Disease names and their model signal scores.

    symptom_score : float
        Symptom-related signal between 0 and 1.

    sentiment_score : float
        Public sentiment signal between 0 and 1.

    temporal_score : float
        Time-related signal between 0 and 1.

    geographic_score : float
        Location-related signal between 0 and 1.

    Returns
    -------
    risk_score : float
        Overall risk score between 0 and 1.

    risk_level : str
        LOW, MODERATE, HIGH or CRITICAL.
    """

    # ------------------------------------------------------
    # 1. DEFINE WEIGHTS
    # ------------------------------------------------------

    disease_weight = 0.40
    symptom_weight = 0.25
    temporal_weight = 0.15
    geographic_weight = 0.10
    sentiment_weight = 0.10

    # Total weight = 1.00


    # ------------------------------------------------------
    # 2. GET STRONGEST DISEASE SIGNAL
    # ------------------------------------------------------

    if disease_scores:

        disease_score = max(
            disease_scores.values()
        )

    else:

        disease_score = 0.0


    # ------------------------------------------------------
    # 3. LIMIT ALL VALUES BETWEEN 0 AND 1
    # ------------------------------------------------------

    disease_score = max(
        0.0,
        min(float(disease_score), 1.0)
    )

    symptom_score = max(
        0.0,
        min(float(symptom_score), 1.0)
    )

    sentiment_score = max(
        0.0,
        min(float(sentiment_score), 1.0)
    )

    temporal_score = max(
        0.0,
        min(float(temporal_score), 1.0)
    )

    geographic_score = max(
        0.0,
        min(float(geographic_score), 1.0)
    )


    # ------------------------------------------------------
    # 4. CALCULATE RISK SCORE
    # ------------------------------------------------------

    risk_score = (

        disease_weight * disease_score

        + symptom_weight * symptom_score

        + temporal_weight * temporal_score

        + geographic_weight * geographic_score

        + sentiment_weight * sentiment_score
    )


    # ------------------------------------------------------
    # 5. DETERMINE RISK LEVEL
    # ------------------------------------------------------

    if risk_score < 0.25:

        risk_level = "LOW"

    elif risk_score < 0.50:

        risk_level = "MODERATE"

    elif risk_score < 0.75:

        risk_level = "HIGH"

    else:

        risk_level = "CRITICAL"


    # ------------------------------------------------------
    # 6. RETURN RESULT
    # ------------------------------------------------------

    return round(risk_score, 3), risk_level


# ==========================================================
# TEST THE RISK ENGINE
# ==========================================================

if __name__ == "__main__":

    # Example disease model signals

    disease_scores = {

        "COVID-19": 0.85,

        "Acute Upper Respiratory Infection": 0.70,

        "Pneumonia": 0.55,

        "Pulmonary Tuberculosis": 0.20
    }


    # Example signal values

    symptom_score = 1.0

    sentiment_score = 0.5

    temporal_score = 0.3

    geographic_score = 0.2


    # Calculate risk

    risk_score, risk_level = calculate_risk(

        disease_scores=disease_scores,

        symptom_score=symptom_score,

        sentiment_score=sentiment_score,

        temporal_score=temporal_score,

        geographic_score=geographic_score
    )


    # ------------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("             PHRIS RISK DETECTION ENGINE")
    print("=" * 60)

    print()
    print("Disease Signals:")

    for disease, score in disease_scores.items():

        print(
            f"{disease:<40} : {score:.3f}"
        )

    print()

    print("Symptom Score     :", symptom_score)

    print("Sentiment Score   :", sentiment_score)

    print("Temporal Score    :", temporal_score)

    print("Geographic Score  :", geographic_score)

    print()
    print("-" * 60)

    print("Overall Risk Score:", risk_score)

    print("Risk Level        :", risk_level)

    print("=" * 60)