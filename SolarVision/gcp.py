import os
from google.cloud import storage

def upload_model_to_gcp(rm=False):
    
    client = storage.Client().bucket('solarvision-test')
    
    storage_location = 'models/solarvision-test/model.h5'
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.h5')
    # print(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}")
    if rm:
        os.remove('model.h5')
