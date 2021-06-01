from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting="hello")

@app.get("/predict")
def predict():
    # pipeline = joblib.load('model.joblib')
    # results = pipeline.predict(X)
    # pred = float(results[0])
    # return pred
    # model = tf.keras.models.load_model('sv_model.h5')
    # test_image = np.zeros(307200).reshape(1, 320, 320, 3)
    # pred = model.predict_classes(test_image)[0][0]
    return dict(test = "worked")