import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)



st.title("Blood Donation Prediction App")
st.write("""
### Predict the likelihood of a blood donation based on several features.
""")

recency = st.number_input('Recency (months since last donation)', min_value=0, max_value=100, value=2)
frequency = st.number_input('Frequency (times)', min_value=0, max_value=50, value=5)
monetary = st.number_input('Monetary (c.c. blood)', min_value=0, max_value=10000, value=500)
time = st.number_input('Time (months)', min_value=0, max_value=100, value=20)


input_data = pd.DataFrame({
    'Recency (months)': [recency],
    'Frequency (times)': [frequency],
    'Monetary (c.c. blood)': [monetary],
    'Time (months)': [time]
})

input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.write("### The model predicts that the individual **will donate blood.**")
    else:
        st.write("### The model predicts that the individual **will not donate blood.**")
