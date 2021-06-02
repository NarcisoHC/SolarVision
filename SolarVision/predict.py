## COPY FROM TFM_PREDINPROD, to be done with Jan/Wolfgang/Narciso

# import os
# from math import sqrt

# import joblib
# import pandas as pd
# from SolarVisioin.params import MODEL_NAME
# from google.cloud import storage
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# PATH_TO_LOCAL_MODEL = 'model.joblib'

# AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"

# BUCKET_NAME = "XXX"  # ⚠️ replace with your BUCKET NAME


# def get_test_data(nrows, data="s3"):
#     """method to get the test data (or a portion of it) from google cloud bucket
#     To predict we can either obtain predictions from train data or from test data"""
#     # Add Client() here
#     path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

#     if data == "local":
#         df = pd.read_csv(path)
#     elif data == "full":
#         df = pd.read_csv(AWS_BUCKET_TEST_PATH)
#     else:
#         df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
#     return df


# def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=False):
#     client = storage.Client().bucket(bucket)

#     storage_location = 'models/{}/versions/{}/{}'.format(
#         MODEL_NAME,
#         model_directory,
#         'model.joblib')
#     blob = client.blob(storage_location)
#     blob.download_to_filename('model.joblib')
#     print("=> pipeline downloaded from storage")
#     model = joblib.load('model.joblib')
#     if rm:
#         os.remove('model.joblib')
#     return model


# def get_model(path_to_joblib):
#     pipeline = joblib.load(path_to_joblib)
#     return pipeline


# def evaluate_model(y, y_pred):
#     MAE = round(mean_absolute_error(y, y_pred), 2)
#     RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
#     res = {'MAE': MAE, 'RMSE': RMSE}
#     return res

# if __name__ == '__main__':

#     # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
#     nrows = 100
#     generate_submission_csv(nrows, kaggle_upload=False)
