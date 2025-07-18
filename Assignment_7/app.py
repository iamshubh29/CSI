import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Load model from same folder
with open("Assignment_7/Trained_model.sav", "rb") as file:
    model = pickle.load(file)

st.title("Diabetes Prediction App")

# Input features
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# Make prediction
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]],
                          columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"The patient is predicted to be: **{result}**")
