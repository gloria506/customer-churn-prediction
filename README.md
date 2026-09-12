# Customer Churn Prediction

## Project Overview
This project is part of the NexAfrica Machine Learning Internship (5-week program). It builds a machine learning classification model to predict whether a telecom customer is likely to churn (leave the company) based on their demographic information, account details, service usage, and billing history.

## Business Problem
Customer churn refers to a subscriber discontinuing their relationship with a service provider — cancelling a line, letting a subscription go inactive, or switching to a competing network. In a competitive telecom market, retaining existing customers is significantly cheaper than acquiring new ones, so predicting churn before it happens allows a company to intervene proactively with targeted retention offers rather than reacting after a customer has already left.

## Dataset
- **Source:** [Telco Customer Churn dataset, Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns
- **Target variable:** `Churn` (Yes/No)
- **Class distribution:** 73.46% No / 26.54% Yes (imbalanced)

## Tools and Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook (VS Code)

## Project Workflow (Week 1)
1. Business understanding — defined the churn problem and its relevance
2. Data collection — obtained the raw Telco Customer Churn dataset
3. Data inspection — reviewed structure, columns, and data types
4. Data cleaning — stripped whitespace from column names, converted `TotalCharges` to numeric, filled 11 missing values (new customers with zero tenure), confirmed zero duplicate rows and zero duplicate customer IDs, verified all categorical values were clean
5. Data dictionary — documented every column, its meaning, and its type
6. Target variable analysis — confirmed class imbalance (73.46% / 26.54%)
7. Train/test split — 80/20 stratified split preserving the churn ratio in both sets

## Key Findings (Week 1)
- The dataset required no major structural repair beyond the `TotalCharges` type conversion — it is otherwise clean and well-formed.
- Roughly 1 in 4 customers in the dataset churned, confirming this is an imbalanced classification problem that will require careful metric selection (precision, recall, F1, ROC-AUC) rather than raw accuracy in later modeling stages.

## Repository Structure
```
customer-churn-prediction/
├── WA_Fn-UseC_-Telco-Customer-Churn (1).csv   # raw dataset
├── telco_churn_cleaned.csv                     # cleaned dataset
├── week1_data_preparation.ipynb                # Week 1 notebook
├── Machine Learning Project Guide.pdf
├── Machine Learning Weekly task.pdf
├── .gitignore
└── README.md
```

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `week1_data_preparation.ipynb` in Jupyter Notebook or VS Code
4. Run all cells in order

## Status
Week 1 complete. Weeks 2–5 (EDA, feature engineering, model development, optimization, and final capstone) in progress.
