
# 🏠 **House Price Prediction – Documentation**  

---

## 📌 **Objective**
The goal of this project is to develop a **machine learning regression model** that predicts house prices based on various property features using a dataset inspired by the Kaggle House Prices challenge.  

---

## 📂 **Files & Outputs**
| File | Description |
|-------|-------------|
| `train.csv` | Training dataset with features and `SalePrice` as target |
| `test.csv` | Test dataset with features (no target) |
| `house_price_predictions_rf.xlsx` | Test predictions using Random Forest |
| `house_price_predictions_linreg.xlsx` | Test predictions using Linear Regression |
| `house_price_predictions_ridge.xlsx` | Test predictions using Ridge Regression |
| `house_price_predictions_lasso.xlsx` | Test predictions using Lasso Regression |  

👉 **All prediction files are uploaded to Google Drive as part of this project.**  

---

## ⚡ **Project Workflow**
### 1️⃣ **Data Loading**
- Datasets loaded from Google Drive.
- Shapes:
  - `train.csv`: 1460 rows, 81 columns
  - `test.csv`: 1459 rows, 80 columns  

---

### 2️⃣ **Exploratory Data Analysis (EDA)**
- **SalePrice distribution** → Right-skewed distribution noted.  
- **Correlation heatmap** → Top features:
  - `OverallQual` (correlation ~0.79)
  - `GrLivArea` (correlation ~0.71)
  - `GarageCars`, `GarageArea`, `TotalBsmtSF` (all >0.6)  
- **Missing data** → Dropped columns with excessive missingness:
  - `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`  

---

### 3️⃣ **Preprocessing**
- **Missing values**:  
  - Numeric → filled with median  
  - Categorical → filled with mode  
- **Feature encoding**:  
  - One-hot encoded categorical features  
- **Feature alignment**:  
  - Ensured train/test have the same features after encoding  

---

### 4️⃣ **Modeling**
Models used:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**

#### 📊 **Cross-Validation RMSE (5-fold)**  
| Model | RMSE |
|--------|------|
| Linear Regression | 33,224.00 |
| Ridge Regression | 33,162.13 |
| Lasso Regression | 33,269.97 *(convergence warnings)* |
| Random Forest | 29,894.82 ✅ Best |

---

### 5️⃣ **Final Predictions**
- Trained on the full train data  
- Predicted on test data  
- Saved predictions to Excel:
  - `house_price_predictions_rf.xlsx`
  - `house_price_predictions_linreg.xlsx`
  - `house_price_predictions_ridge.xlsx`
  - `house_price_predictions_lasso.xlsx`  

---

## 📝 **Technologies**
- Python 3
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Google Colab + Google Drive  

---

## ✅ **Deliverables**
📂 **Uploaded Prediction Files:**
- `house_price_predictions_rf.xlsx`
- `house_price_predictions_linreg.xlsx`
- `house_price_predictions_ridge.xlsx`
- `house_price_predictions_lasso.xlsx`  

Each file contains:
| Id | SalePrice |
|-----|-----------|
| (House ID) | (Predicted Price) |  

---

## 💡 **Future Improvements**
- Apply log-transform to `SalePrice` to address skewness  
- Hyperparameter tuning (`GridSearchCV`)  
- Feature selection to reduce dimensionality  
- Feature importance visualization for model interpretability  

---

### ✨ **Summary**
👉 A robust machine learning pipeline was built that:
- Performed EDA, preprocessing, and modeling  
- Compared multiple regression models  
- Exported results for external use  
