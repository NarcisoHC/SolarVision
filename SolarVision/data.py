import numpy as np
import os
from PIL import Image
from google.cloud import storage

def get_data():
  '''get image data'''
  storage_client = storage.Client()
  bucket_name = "solarvision-test"

  bucket = storage_client.bucket(bucket_name)
  blobs = list(storage_client.list_blobs(bucket))
  
  X_train = []
  X_test = []
  y_train = []
  y_test = []

  for b in blobs:
    file_name = b.name
    if 'data/data/train_subset' in file_name or 'data/data/val_subset' in file_name:
      blob = bucket.blob(file_name)
      blob.download_to_filename("downloaded_image.png")
      image = Image.open("downloaded_image.png").convert('RGB')
      X_train.append(np.array(image))
      # print(f'X_train len = {len(X_train)}')
      if '/0/' in file_name: 
          y_train.append(0)
      else:
          y_train.append(1)
    elif 'data/data/test_subset' in file_name:
      blob = bucket.blob(file_name)
      blob.download_to_filename("downloaded_image.png")
      image = Image.open("downloaded_image.png").convert('RGB')
      X_test.append(np.array(image))
      # print(f'X_test_len = {len(X_test)}')
      if '/0/' in file_name: 
          y_test.append(0)
      else:
          y_test.append(1)

  X_train = np.array(X_train) / 255. # normalize
  X_test = np.array(X_test) / 255. # normalize
  y_test = np.array(y_test) 
  y_train = np.array(y_train) 
  
  return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))
    