# Diabetes Prediction Model

This project uses a machine learning model to predict whether a person is diabetic based on their medical data.

## Project Overview

This script uses a pre-trained model to predict diabetes from input data. The input data consists of various medical measurements such as glucose levels, blood pressure, BMI, and age. The model has been trained using a dataset of similar data points and their corresponding diabetes diagnosis.

### Input Data:
The input data for prediction consists of the following features:
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration after 2 hours in an oral glucose tolerance test
3. **BloodPressure** - Diastolic blood pressure (mm Hg)
4. **SkinThickness** - Triceps skinfold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction** - A function that scores likelihood of diabetes based on family history
8. **Age** - Age in years

### Explanation:

- The input data is converted into a NumPy array to work with machine learning models.
- The data is reshaped to ensure that it matches the expected input size for the model (one instance of multiple features).
- The data is standardized using a pre-fitted scaler before passing it to the model for prediction.
- The model (`classifier`) makes a prediction, returning **0** for non-diabetic and **1** for diabetic.


