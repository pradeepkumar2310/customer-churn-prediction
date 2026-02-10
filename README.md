# Customer Churn Prediction using Machine Learning

## 📌 Project Overview
This project predicts customer churn using Machine Learning techniques.  
The goal is to identify customers who are likely to leave a service based on historical data.

## 🧠 Problem Statement
Customer churn is a major challenge for businesses.  
By predicting churn early, companies can take preventive actions to retain customers.

## 📊 Dataset
- File: `churn_data.csv`
- Contains customer demographic and service usage information
- Target variable: `Churn` (Yes / No)

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression

## 🔄 Workflow
1. Data loading and preprocessing  
2. Encoding categorical variables  
3. Feature selection  
4. Train-test split  
5. Model training (Logistic Regression)  
6. Model evaluation using:
   - Confusion Matrix
   - Classification Report

## 📈 Model Output
The model predicts whether a customer is likely to churn or not based on input features.

## Results
The Logistic Regression model achieved 100% accuracy on the test dataset.
Due to the small dataset size, results are illustrative and intended for learning purposes.

## ▶️ How to Run
```bash
python Churn_proj.py
