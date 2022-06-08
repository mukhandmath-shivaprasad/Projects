import streamlit as st
import pandas as pd
import joblib

st.header("Machine Learning App")
# ['year',  'km_traveled',  'tax',  'engineSize',  'km_per_liters',  'model',  'transmission',  'fuel_type']
year = st.text_input("Enter Year")
km_traveled = st.number_input("Enter km_traveled", format="%.2f")
tax = st.number_input("Enter tax")
engineSize = st.number_input("Enter Engine Size")
km_per_liters = st.number_input("Enter kmpl")
model = st.text_input("Enter vehicle model")
transmission = st.text_input("Enter transmission type ")
fuel_type = st.text_input("Enter fuel type")
if st.button("Submit"):
    model = joblib.load("final_car_prediction.pkl")
    cols = joblib.load("column_name.pkl")
    input = pd.DataFrame(data=[['year',  'km_traveled',  'tax',  'engineSize',
                                'km_per_liters',  'model',  'transmission',  'fuel_type']],
                         columns=cols)
    pred = model.predict(input)
    st.text(f"The price of the model {pred}")
