import joblib
import pandas as pd

from config import MODEL_PATH


# Load the trained model
model = joblib.load(MODEL_PATH)


def predict_price(car_data):
    """
    Predict car price using the saved pipeline.
    car_data: Dictionary containing car features.
    """

    # Convert input data to DataFrame
    car_df = pd.DataFrame([car_data])

    # Predict price
    prediction = model.predict(car_df)

    return prediction[0]