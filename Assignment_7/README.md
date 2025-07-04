# 🩺 Diabetes Prediction Model

This project leverages a machine learning model to predict whether a person is diabetic based on medical features.

👉 **Live App:** [Click here to try the Streamlit App](https://6kuxclj5kyuma8wxs3fjye.streamlit.app/)

---

## 🚀 Project Overview

This web-based application allows users to input health-related metrics and receive an instant diabetes risk prediction. The model behind this app is trained on a labeled dataset containing diagnostic outcomes and medical measurements.

---

## 📊 Input Features

The prediction is based on the following 8 medical inputs:

1. **Pregnancies** – Number of times pregnant  
2. **Glucose** – Plasma glucose concentration  
3. **BloodPressure** – Diastolic blood pressure (mm Hg)  
4. **SkinThickness** – Triceps skinfold thickness (mm)  
5. **Insulin** – 2-Hour serum insulin (mu U/ml)  
6. **BMI** – Body Mass Index (weight in kg / (height in m)^2)  
7. **DiabetesPedigreeFunction** – Score based on family history  
8. **Age** – Age in years  

---

## 🧠 Model Explanation

- The input is structured into a format compatible with the trained model.
- Preprocessing includes reshaping and scaling using a pre-fitted scaler.
- A saved machine learning model (`Trained_model.sav`) is loaded using `pickle`.
- The model predicts:
  - `0` → **Not Diabetic**
  - `1` → **Diabetic**

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit (for deployment)

---

## 📎 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
