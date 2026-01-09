# 🛍️ Customer Purchase Prediction  
### Machine Learning Classification Project

![ML](https://img.shields.io/badge/Machine%20Learning-Classification-blueviolet)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Level](https://img.shields.io/badge/Level-Industry%20Grade-orange)

---

## 🚀 Project Overview

This project focuses on predicting whether a customer will make a purchase based on demographic and behavioral data. It follows a **complete machine learning pipeline** — from data preprocessing and exploratory analysis to model training, evaluation, and business insights.

**Goal:** Predict customer purchase behavior  
**Type:** Supervised Classification  
**Output:** Binary (Purchase / No Purchase)

---

## ✨ Key Highlights

- End-to-end machine learning workflow  
- Advanced exploratory data analysis (EDA)  
- Feature engineering and feature selection  
- Multiple ML models with comparison  
- Ensemble learning for higher accuracy  
- Business-oriented insights and recommendations  

---

## 📂 Dataset Description

**Input Features**
- Age  
- Gender  
- Annual Income  
- Number of Purchases  
- Time Spent on Website  
- Product Category  
- Loyalty Program Status  
- Discounts Availed  

**Target Variable**
- `PurchaseStatus`  
  - `0` → No Purchase  
  - `1` → Purchase  

---



## 🔄 Machine Learning Pipeline

1. Data Loading  
2. Data Cleaning & Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Feature Selection  
6. Model Training  
7. Hyperparameter Tuning  
8. Ensemble Modeling  
9. Model Evaluation  
10. Business Insights  

---

## 🧠 Models Implemented

| Model | Category |
|------|---------|
| Logistic Regression | Linear |
| Decision Tree | Tree-Based |
| Random Forest | Ensemble |
| K-Nearest Neighbors | Distance-Based |
| Support Vector Machine | Kernel-Based |
| XGBoost | Boosting |
| LightGBM | Boosting |
| CatBoost | Boosting |
| Voting Classifier | Ensemble |
| Stacking Classifier | Meta-Learning |

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

**Primary Metric Used:** F1-Score (handles class imbalance effectively)

---

## 🏆 Best Performing Model

**Ensemble Models (Voting / Stacking)** achieved the best overall performance with:
- High accuracy  
- Strong generalization  
- Robust predictions  

---

## 💡 Business Insights

- High-income customers have a higher probability of purchasing  
- Loyalty program members convert significantly better  
- Time spent on the website is a strong predictor  
- Discounts positively influence purchase decisions  

**Recommendation:**  
Use prediction probabilities to target high-intent customers and optimize marketing campaigns.

---

## 🧪 Sample Prediction

```python
prediction = model.predict(new_customer_data)
