import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def get_data():
  '''get image data'''

  path_0 = 'data/train_subset/0/'
  path_1 = 'data/train_subset/1/'
  path_2 = 'data/test_subset/0/'
  path_3 = 'data/test_subset/1/'
  path_4 = 'val_subset/0/'
  path_5 = 'val_subset/1/'

  path = [path_0, path_1, path_2, path_3, path_4, path_5]
  X_train = []
  X_test = []
  y_train = []
  y_test = []

  for p in path:
    for img in os.listdir(p):
      if img.startswith('.'):
        continue
      else:
        image = Image.open(os.path.join(p,img)).convert('RGB')
        # image = image.resize((320, 320)) 
        if p[:2] == 'tr' or p[:2] == 'va': 
            X_train.append(np.array(image)) 
            if p[-2] == '0': 
                y_train.append(0)
            else:
                y_train.append(1)
        else: 
            X_test.append(np.array(image)) 
            if p[-2] == '0': 
                y_test.append(0)
            else:
                y_test.append(1)

  X_train = np.array(X_train)/ 255.
  X_test = np.array(X_test)/ 255.
  y_test = np.array(y_test) 
  y_train = np.array(y_train) 
  
  return X_train, X_test, y_train, y_test