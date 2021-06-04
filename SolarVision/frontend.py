import streamlit as st
import requests
import os
import uuid
from google.cloud import storage

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('SolarVision V1.0')


st.markdown('''
Please, upload an image with a rooftop and we will classify it depending if it has or not a solar panel.
''')

uploaded_file = st.file_uploader(label='upload .jpg or .png file', type=['jpg', 'png']) 

if uploaded_file is not None: 

        st.image(uploaded_file, caption='Your image', use_column_width='auto')
        uploaded_file.name = str(uuid.uuid4())
        with open(os.path.join('SolarVision/tempDir', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        storage_client = storage.Client()
        bucket = storage_client.bucket('solarvision-test')
        blob = bucket.blob(os.path.join('data/predict_image', uploaded_file.name))
        blob.upload_from_filename(os.path.join('SolarVision/tempDir', uploaded_file.name)) 

else:
    st.write('Please, upload a file') 



#then, after this file is sent to GCP, and is processed for prediction, we retreive the prediction from the API:

#url = '' 

#params={'key':value}

#response = requests.get(url, params).json()

#if response[] == 0:
#    st.write('This rooftop has a solar panel')

#else:
#    st.write('This rooftop does not have a solar panel')

#remove files from tempDir and GCP 
#os.remove(os.path.join('SolarVision/tempDir', uploaded_file.name))