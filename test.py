"""Utility to predict the default probability for a single client."""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/default_probability_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
ENCODER_PATH = Path("models/encoder.pkl")


def predict_default_probability(client_data: dict) -> float:
    """Predict the probability of a client defaulting on a loan.

    Parameters
    ----------
    client_data : dict
        Dictionary with the following keys:
        ``age``, ``income``, ``home_ownership_type``, ``employment_length``,
        ``loan_amount``, ``loan_interest_rate``, ``loan_percent_income``,
        ``has_defaulted_before``, ``credit_history_length``.

    Returns
    -------
    float
        Predicted probability of default.
    """

    # Load preprocessing objects and trained model
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Convert input to DataFrame for easy manipulation
    df = pd.DataFrame([client_data])

    # Apply transformations consistent with model training
    categorical = encoder.transform(df[["home_ownership_type"]]).toarray()
    numerical_columns = [
        "age",
        "income",
        "employment_length",
        "loan_amount",
        "loan_interest_rate",
        "loan_percent_income",
        "has_defaulted_before",
        "credit_history_length",
    ]
    numerical = scaler.transform(df[numerical_columns])

    # Combine numerical and encoded categorical features
    x = np.hstack([numerical, categorical])

    # Predict probability for class "default"
    return float(model.predict_proba(x)[0, 1])

if __name__ == "__main__":
    # Example input data for demonstration purposes
    sample_client = {
        "age": 35,
        "income": 50000,
        "home_ownership_type": "OWN",  # OWN, MORTGAGE, RENT, OTHER
        "employment_length": 5,
        "loan_amount": 15000,
        "loan_interest_rate": 0.08,
        "loan_percent_income": 0.3,
        "has_defaulted_before": 0,
        "credit_history_length": 7,
    }

    probability = predict_default_probability(sample_client)
    print(f"Default Probability: {probability:.2%}")
