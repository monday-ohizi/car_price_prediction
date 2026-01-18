import joblib
import pandas as pd
import numpy as np
from typing import List, Dict

MODEL_PATH = "model/final_car_price_pipeline.joblib"

# Load the trained sklearn pipeline from disk
def load_model():
    return joblib.load(MODEL_PATH)


def predict_prices(cars: List[Dict]) -> List[float]:
    if not isinstance(cars, list) or len(cars) == 0:
        raise ValueError("cars must be a non-empty list of dictionaries")

    model = load_model()
    df = pd.DataFrame(cars)

    predictions = model.predict(df)

    # Clamp negative predictions to 0
    return np.maximum(predictions, 0.0).tolist()
