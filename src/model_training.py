import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataframe_preprocessed = pd.read_csv('../assets/preprocessed_fraud_train.csv')
dataframe_test_preprocessed = pd.read_csv('../assets/preprocessed_fraud_test.csv')

dataframe_preprocessed = dataframe_preprocessed.sample(frac=1, random_state=42)

x_train = dataframe_preprocessed.drop('is_fraud', axis=1)
y_train = dataframe_preprocessed["is_fraud"]

x_test = dataframe_test_preprocessed.drop('is_fraud', axis=1)
y_test = dataframe_test_preprocessed["is_fraud"]

model = Sequential(
    [
        Conv1D(kernel_size=2, filters=2, input_shape=(48, 1), activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', metrics=['accuracy', Precision(), Recall(), AUC()], loss='binary_crossentropy')
print("Model Training")
print("======================")
model.fit(x_train, y_train, validation_split=0.4, epochs=15, batch_size=128)
print("======================")
print("Model Evaluation")
print("======================")
model.evaluate(x_test, y_test)
print("Model Summary")
print("======================")
model.summary()
