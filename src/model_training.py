import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout

import os

from src.compile_file_path import get_file_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataframe_preprocessed = pd.read_csv(get_file_path('assets/preprocessed_fraud_train.csv'))
dataframe_test_preprocessed = pd.read_csv(get_file_path('assets/preprocessed_fraud_test.csv'))

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
history = model.fit(x_train, y_train, validation_split=0.4, epochs=15, batch_size=128)
model.save(get_file_path('model/fraud_detection_model'))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converted_model = converter.convert()
with open(get_file_path('model/fraud_detection_model.tflite'), 'wb') as f:
    f.write(converted_model)
    f.close()

print("======================")
print("Model Evaluation")
print("======================")
model.evaluate(x_test, y_test)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Fraud Detection CNN Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Fraud Detection CNN Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Fraud Detection CNN Precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# summarize history for recall
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_recall'])
plt.title('Fraud Detection CNN Recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# summarize history for auc
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Fraud Detection CNN AUC')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# extract actual results from the dataset
cm_labels = np.reshape(np.array(y_test), (1, y_test.shape[0]))[0]
# predict the labels from the corresponding training set and round them to result to 0 or 1
predictions = np.round(model.predict(x_test))
# flatten the prediction to match the shape of the labels
predictions = np.reshape(np.array(predictions), (1, predictions.shape[0]))[0]

# plot confusion matrix with 500 sample data
cm = tf.math.confusion_matrix(cm_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Fraud Detection CNN Confusion Matrix')
plt.show()
print("Model Summary")
print("======================")
model.summary()
