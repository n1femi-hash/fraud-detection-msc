from tensorflow.keras.models import load_model

model = load_model('../model/fraud_detection_model')
print(model.summary())