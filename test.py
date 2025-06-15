import pickle
import numpy as np

def predict_default_probability(client_data):
    """
    Predicts the default probability for a client using a pre-trained model.

    Args:
        client_data (list or np.ndarray): The input features for the client.
            Must follow this order:
            [age, income, employment_length, loan_amount, loan_interest_rate,
             loan_percent_income, has_defaulted_before, credit_history_length]

    Returns:
        float: The predicted default probability.
    """

    # Load model (ensure model file exists at specified path)
    with open("models/default_probability_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Ensure input is 2D array
    x = np.array(client_data).reshape(1, -1)

    # Predict probability (class 1 = default)
    prob = model.predict_proba(x)[0][1]

    return prob

# Client data in the specified order:
# age, income, home_ownership_type, employment_length,
# loan_amount, loan_interest_rate, loan_percent_income,
# has_defaulted_before, credit_history_length

client_data = [
    35,      # age
    50000,   # income
    'OWN',   # home_ownership_type (OWN, MORTGAGE, RENT, OTHER)
    5,       # employment_length
    15000,   # loan_amount
    0.08,    # loan_interest_rate
    0.3,     # loan_percent_income
    0,       # has_defaulted_before
    7        # credit_history_length
]

print(f"Default Probability: {predict_default_probability(client_data):.2%}")