from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import models
import numpy as np
from PIL import Image
from google.cloud import storage

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

    bucket_name = "solarvision-test"
    source_blob_name = "data/data/test_subset/0/5.919714944045901,51.04547096048374.png"
    
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename("test_image.png")
    
    image = Image.open("test_image.png").convert('RGB')
    image = np.array(image).reshape(1,320,320,3)/255

    model = models.load_model('sv_model.h5')
    pred = model.predict_classes(image)[0][0].tolist()
    
    return dict(test = pred)
