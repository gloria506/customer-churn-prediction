# Customer Churn Prediction

## Project Overview
This project was completed as part of the NexAfrica Machine Learning Internship (5-week program). It builds an end-to-end machine learning classification model to predict whether a telecom customer is likely to churn (leave the company), based on demographic information, account details, service usage, and billing history.

## Business Problem
Customer churn refers to a subscriber discontinuing their relationship with a service provider - cancelling a line, letting a subscription go inactive, or switching to a competing network. In a competitive telecom market, retaining existing customers is significantly cheaper than acquiring new ones, so predicting churn before it happens allows a company to intervene proactively with targeted retention offers rather than reacting after a customer has already left.

## Dataset
- **Source:** [Telco Customer Churn dataset, Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 original columns
- **Target variable:** `Churn` (Yes/No)
- **Class distribution:** 73.46% No / 26.54% Yes (imbalanced)

## Tools and Technologies
- Python
- Pandas, NumPy
- Scikit-learn, imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Jupyter Notebook (VS Code)

## Project Workflow

**Week 1 - Business Understanding, Data Collection & Preparation**
Defined the business problem, loaded and inspected the raw dataset, cleaned column-name whitespace, converted `TotalCharges` to numeric, filled 11 missing values (new customers with zero tenure), confirmed zero duplicates, built a data dictionary, and analyzed the target variable's class imbalance.

**Week 2 - Exploratory Data Analysis & Feature Engineering**
Identified 6 key churn drivers (contract type, tenure, monthly charges, internet service type, payment method, senior citizen status) across 8+ visualizations. Engineered 3 new features (TenureGroup, NumServices, SpendingTier), each validated against churn before inclusion. Preprocessed the dataset (encoding, scaling) into a machine-learning-ready format.

**Week 3 - Model Development**
Trained and compared three classification models: Logistic Regression (baseline), Decision Tree, and Random Forest. Logistic Regression performed best untuned (ROC-AUC 0.843); all three models shared a common weakness of ~50% recall on the churn class.

**Week 4 - Model Optimization**
Addressed class imbalance via class weighting and SMOTE oversampling, applied 5-fold cross-validation to confirm result stability, and used GridSearchCV to tune a Random Forest model. The final tuned model achieved 77.3% recall (up from ~50%), while maintaining strong ROC-AUC (0.842).

**Week 5 - Model Interpretation & Business Recommendations**
Evaluated the final model in business terms, extracted feature importances (which closely matched the Week 2 EDA insights), analyzed false positive/negative error patterns, and produced 6 actionable business recommendations.

## Key Findings
- **Contract type** is the strongest behavioral churn driver: month-to-month customers churn at 42.7%, versus 2.8% for two-year contracts.
- **Tenure** is the single strongest predictor overall (highest feature importance): churned customers had roughly half the average tenure of retained customers.
- **Fiber optic customers** churn at nearly double the rate of DSL customers, despite being the premium product.
- **Electronic check users** churn at roughly 3x the rate of customers on automatic payment methods.
- **Senior citizens** churn at nearly double the rate of non-seniors.
- The final model's feature importances independently confirmed nearly every insight found through manual exploratory analysis.

## Final Model Performance
| Metric | Score |
|---|---|
| Accuracy | 0.7580 |
| Precision | 0.5303 |
| Recall | 0.7727 |
| F1-Score | 0.6289 |
| ROC-AUC | 0.8423 |

**Final model:** Tuned Random Forest Classifier (`n_estimators=200, max_depth=10, min_samples_split=5, class_weight='balanced'`)

## Business Recommendations
1. Incentivize migration off month-to-month contracts toward longer-term commitments.
2. Build a dedicated onboarding/retention program for a customer's first 12 months.
3. Investigate fiber optic service quality and pricing directly with subscribers.
4. Incentivize a shift from electronic check to automatic payment methods.
5. Deploy the model with a lowered decision threshold to catch more borderline at-risk customers.
6. Prioritize senior citizens in retention campaigns.

## Repository Structure
```
customer-churn-prediction/
├── Week1/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn (1).csv   # raw dataset
│   ├── telco_churn_cleaned.csv                     # cleaned dataset
│   └── week1_data_preparation.ipynb
├── Week2/
│   ├── week2_eda_feature_engineering.ipynb
│   ├── telco_churn_feature_engineered.csv
│   └── Week2_Report_Customer_Churn_Prediction.docx
├── Week3/
│   ├── week3_model_development.ipynb
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── X_train.csv / X_test.csv / y_train.csv / y_test.csv
│   └── Week3_Report_Customer_Churn_Prediction.docx
├── Week4/
│   ├── week4_optimization.ipynb
│   ├── final_model_random_forest_tuned.pkl
│   └── Week4_Report_Customer_Churn_Prediction.docx
├── Week5/
│   ├── week5_final_evaluation.ipynb
│   └── Final_Capstone_Report_Customer_Churn_Prediction.docx
├── Machine Learning Project Guide.pdf
├── Machine Learning Weekly task.pdf
├── .gitignore
└── README.md
```

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib`
3. Open notebooks in order (Week1 through Week5) in Jupyter Notebook or VS Code
4. Run all cells in each notebook sequentially

## Status
Project complete - all 5 weeks (business understanding through final model interpretation and business recommendations).
