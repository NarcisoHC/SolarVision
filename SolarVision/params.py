### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-606-caduff'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
BUCKET_TRAIN_DATA_PATH = 'SolarVision/data/train_subset/0/' # change to real train data set

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'solarvision'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
STORAGE_LOCATION = 'models/solarvision/model.joblib'


### MLFLOW - - - - - - - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = 'https://mlflow.lewagon.co'

EXPERIMENT_NAME = '[DE] [Berlin] [caduffn] SolarVison v1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -