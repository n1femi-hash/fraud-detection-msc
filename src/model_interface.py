from unittest.mock import inplace

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

from src.compile_file_path import get_file_path
from src.data_cleaning import clean_dataframe
from src.data_preprocessing import data_vectorisation, dataframe_reindex
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

model = load_model(get_file_path('model/fraud_detection_model'))


def process_input(data):
    dataframe = pd.read_json(data)
    cleaned_dataframe = clean_dataframe(dataframe)
    vectorised_data = data_vectorisation(cleaned_dataframe)
    reindexed_data = dataframe_reindex(vectorised_data)
    reindexed_data.drop("is_fraud", axis=1, inplace=True)
    return np.asarray(reindexed_data, dtype=float)


def run_prediction(data, threshold):
    prediction = model.predict(data)
    percentage = float(prediction[0][0] * 100)
    crosses_threshold = percentage > threshold
    verdict = "not fraudulent"
    if crosses_threshold:
        verdict = "fraudulent"
    return f"This transaction with a rating of {round(percentage, 2)}% is {verdict} based on your threshold of {threshold}%", crosses_threshold
