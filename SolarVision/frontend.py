import streamlit as st
#import pltoly_express as px


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('SolarVision V1.0')


st.markdown('''
Please, upload an image with a rooftop and we will classify it depending if it has or not a solar panel.
''')

#st.sidebar.subheader('Visualization settings')

uploaded_file = st.file_uploader(label='upload .jpg or .png file', type=['jpg', 'png'])

if uploaded_file is not None:

    st.image(uploaded_file, caption='Your image', use_column_width=False)

#url = ''

#dictionary containing the parameters for our API...

#3. Let's call our API using the `requests` package...

#4. Let's retrieve the prediction from the **JSON** returned by the API...

## Finally, we can display the prediction to the user
