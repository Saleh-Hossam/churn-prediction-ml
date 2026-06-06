# Customer Churn Prediction 

Churn is expensive. Acquiring a new customer costs significantly more than keeping an existing one,
which makes early detection valuable for any subscription-based business.

This project builds a machine learning pipeline to predict which customers are likely to churn,
understand why, and translate the findings into actionable business recommendations.

---

## Dataset

7,043 customers, 20 features covering demographics, account information, and subscribed services.
Target variable: whether the customer churned within the last month.

---

## What this project covers

- Data cleaning and type correction (TotalCharges was stored as a string)
- Encoding binary and multi-class categorical features
- Exploratory analysis — churn rate by contract type, internet service, tenure, and monthly charges
- Training and comparing five models: Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM
- Hyperparameter tuning with GridSearchCV
- Feature engineering — added AvgMonthlySpend (TotalCharges / tenure)
- Handling class imbalance using class_weight='balanced'
- Evaluation with Accuracy, Precision, Recall, F1, ROC-AUC, and Precision-Recall curves
- Logistic Regression coefficients analysis to explain which features drive churn

---

## Model comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.800 | 0.620 | 0.510 | 0.560 |
| Decision Tree | 0.728 | 0.480 | 0.490 | 0.485 |
| Random Forest (default) | 0.793 | 0.630 | 0.460 | 0.532 |
| Random Forest (tuned) | 0.798 | 0.656 | 0.473 | 0.550 |
| XGBoost | 0.797 | 0.635 | 0.490 | 0.554 |
| **Logistic Regression (balanced)** | **0.730** | **0.500** | **0.790** | **0.610** |

> Replace these numbers with your actual printed outputs before publishing.

---

## Why the balanced Logistic Regression was selected

The dataset has a 73/27 class split — most customers did not churn.
Without addressing this, models consistently favored the majority class and produced low Recall.

After applying `class_weight='balanced'`, Recall improved from 0.51 to 0.79.
Precision dropped from 0.62 to 0.50, but that tradeoff is acceptable here.

In a churn context, missing an at-risk customer is more costly than a false alarm.
The model also achieved an ROC-AUC of 0.83, which means it can meaningfully
separate churners from non-churners across all classification thresholds.

---

## Key findings

**What increases churn risk:**
- Fiber Optic internet service showed the highest churn rate among all service types
- Month-to-month contracts had significantly higher churn than annual or two-year contracts
- Customers paying via electronic check churned more than those using other payment methods
- No online security or tech support subscription was associated with higher churn

**What reduces churn risk:**
- Longer customer tenure was the strongest retention signal — loyal customers stay
- One-year and two-year contracts showed much lower churn
- Having dependents correlated with lower churn probability
- Online security and tech support subscriptions were both protective factors

---

## Business recommendations

**1. Prioritize the first few months**
New customers churn the most. Early engagement programs — onboarding calls, first-month offers —
would target the highest-risk window.

**2. Push annual contracts**
The gap in churn rate between month-to-month and annual contracts is large enough
that even a modest incentive to upgrade would likely pay off.

**3. Investigate Fiber Optic satisfaction**
This segment has a churn rate that stands out from the rest.
It may indicate a service quality or pricing issue worth a dedicated investigation.

**4. Bundle security and support services**
Online Security and Tech Support both reduce churn risk.
Offering them as a discounted bundle could improve retention without heavy discounting on price alone.

---

## Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter Notebook

---

## Files

├── Customer_Churn.ipynb    # Full notebook — cleaning, EDA, modeling, evaluation

├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Raw dataset

└── README.md

---

## Author

**Saleh Hossam** · Data Analyst  
[LinkedIn](https://linkedin.com/in/saleh-hossam/) · [GitHub](https://github.com/Saleh-Hossam)
