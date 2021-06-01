import pandas as pd
import numpy as np
import os
from PIL import Image

def get_data():
  '''get image data'''

  path_0 = 'SolarVision/data/train_subset/0/'
  path_1 = 'SolarVision/data/train_subset/1/'
  path_2 = 'SolarVision/data/test_subset/0/'
  path_3 = 'SolarVision/data/test_subset/1/'
  #path_4 = 'SolarVision/data/val_subset/0/'
  #path_5 = 'SolarVision/data/val_subset/1/'

  path = [path_0, path_1, path_2, path_3] #path_4, path_5]
  
  X_train = []
  X_test = []
  y_train = []
  y_test = []

  for p in path:
    for img in os.listdir(p)[:200]:
      if img.startswith('.'):
        continue
      else:
        image = Image.open(os.path.join(p,img)).convert('RGB')
        # image = image.resize((320, 320)) 
        if p[:19] == 'SolarVision/data/tr' or p[:19] == 'SolarVision/data/va': 
            X_train.append(np.array(image))
            # print(f'X_train len = {len(X_train)}')
            if p[-2] == '0': 
                y_train.append(0)
            else:
                y_train.append(1)
        elif p[:19] == 'SolarVision/data/te': 
            X_test.append(np.array(image))
            # print(f'X_test len = {len(X_test)}')
            if p[-2] == '0': 
                y_test.append(0)
            else:
                y_test.append(1)

  X_train = np.array(X_train) / 255.
  X_test = np.array(X_test) / 255.
  y_test = np.array(y_test) 
  y_train = np.array(y_train) 
  
  # print(X_train)
  # print(X_test)
  # print(y_train)
  # print(y_test)
  
  return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    # all_data = get_data()
    # X_train = all_data[0]
    # X_test = all_data[1]
    # y_train = all_data[2]
    # y_test = all_data[3]
    # print(X_train.shape)
    # print(X_test.shape)
    # print(len(y_train))
    # print(len(y_test))