from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import models
import numpy as np
from PIL import Image
from google.cloud import storage

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def model():
    inputs = Input(shape=(320, 320, 3), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model

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
def predict():#upload):

    bucket_name = "solarvision-test"
    
    #source_blob_name = upload
    source_blob_name = "/data/data/test_set/1/6.0091980000000005,51.810108.png"
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename("test_image.png")
    
    image = Image.open("test_image.png").convert('RGB')
    image = np.array(image).reshape(320,320,3)/255

    pretrained_model = model()
    pretrained_model.load_weights('seg_model_weights.h5')
    prediction = pretrained_model.predict(np.expand_dims(image, axis=0))[0]> 0.5
    prediction = prediction * 255

    im_array = prediction.reshape(320,320).astype(np.uint8)
    test = Image.fromarray(im_array)
    test.save("test.png")

    return dict(mask = 0)