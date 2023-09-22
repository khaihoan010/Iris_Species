# iris_slider_app.py

import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('tuned_svm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create a function for the model prediction
def predict_species(data):
    prediction = loaded_model.predict(data)
    return prediction[0]

# Design the webpage
st.title('Iris Species Prediction using Sliders')
st.write("""
Use the sliders below to specify the attributes of the iris flower and predict its species.
""")

# Sliders for user input
sepal_length = st.slider('Sepal Length (in cm)', min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.slider('Sepal Width (in cm)', min_value=2.0, max_value=5.0, value=2.3, step=0.1)
petal_length = st.slider('Petal Length (in cm)', min_value=1.0, max_value=7.0, value=3.3, step=0.1)
petal_width = st.slider('Petal Width (in cm)', min_value=0.1, max_value=3.0, value=1.1, step=0.1)

# Predict button
if st.button('Predict'):
    input_data = {
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    }
    new_sample = pd.DataFrame(input_data)
    result = predict_species(new_sample)
    st.write('Predicted Species:', result)

# Running the Streamlit app: In your terminal, navigate to the directory containing `iris_slider_app.py` and run `streamlit run iris_slider_app.py`.
