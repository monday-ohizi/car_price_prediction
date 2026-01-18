# Car Price Prediction – End-to-End Machine Learning Pipeline

## Project Overview

This project builds a robust, interpretable, and deployment-ready machine learning pipeline for predicting car prices using structured vehicle data.

The work is split into two notebooks to reflect a real-world ML workflow:

- **Notebook 1:** Data understanding, cleaning, feature engineering, and baseline modeling
- **Notebook 2:** Pipeline engineering, model selection, stability validation, and explainability

The final output is a single validated pipeline ready for production.

## Problem Definition

- **Task:** Supervised Learning (Regression)
- **Target Variable:** Price
- **Objective:** Predict realistic car prices while maintaining stability across time

**Evaluation Metrics:**
- Primary: RMSE
- Secondary: MAE

**Validation Strategy:**
- Notebook 1: K-Fold Cross-Validation
- Notebook 2: TimeSeriesSplit to prevent temporal leakage

## Dataset

- **Source:** `data/car_price_with_errors.csv`
- **Issues included intentionally:**
  - Inconsistent categorical values
  - Invalid numeric ranges
  - Formatting errors

All issues are handled explicitly in the pipeline.

## Repository Structure
```
car-price-prediction-ml/
│
├── data/               # CSV dataset
├── notebooks/          # EDA and modeling notebooks
├── models/             # Trained pipeline (.pkl)
├── src/                # Production-ready scripts
├── requirements.txt
├── README.md
└── .gitignore
```
## Notebook 1 – EDA, Cleaning & Baseline Modeling

**Key Steps:**
- Exploratory Data Analysis
- Domain-driven categorical cleaning
- Numerical sanity checks (Year, Mileage, Engine Size)
- Log transformation of skewed target
- Baseline Linear Regression & manual feature engineering
- 5-Fold Cross-Validated RMSE evaluation

**Outcome:**
Established baseline performance and justified need for tree-based models.

## Notebook 2 – Model Selection & Validation

**Pipeline Engineering:**
- All cleaning and feature engineering moved into pipelines
- Custom `FunctionTransformer` steps for:
  - Categorical normalization
  - Numerical validation
  - Feature engineering (Car Age, Log Mileage)

**Models Evaluated:**
- ElasticNet (log-price)
- Random Forest
- XGBoost
- LightGBM

**Validation & Hyperparameter Tuning:**
- TimeSeriesSplit for temporal consistency
- GridSearchCV for ElasticNet
- RandomizedSearchCV for tree models
- Stability gate: CV_std / CV_mean ≤ 0.15

**Model Selection Criteria:**
A model is deployable if it:
1. Has lowest CV RMSE
2. Passes stability threshold
3. Performs consistently on a hold-out set

## Model Evaluation & Explainability

**Performance Analysis:**
- Hold-out RMSE evaluation
- Error analysis by price segments: Cheap, Mid, High, Luxury

**Explainability Techniques:**
- Tree-based feature importance
- Permutation Importance
- SHAP (global explanation)

**Key Insights:**
- Engine Size: strongest price driver
- Car Age: captures depreciation
- Specific models (C-Class, Civic, Corolla) command premiums
- Brand influence is secondary

## Deployment Readiness

- Entire preprocessing + model stored in **one pipeline**
- Saved artifact: `models/final_car_price_pipeline.joblib`
- Can be directly loaded and used for inference without re-training

## Installation & Usage

```bash
git clone https://github.com/monday-ohizi/car_price_prediction.git
cd car-price-prediction-ml
pip install -r requirements.txt
python src/test_predict.py # Run the test script to predict sample cars
example: cars = [
    {"Make": "Toyota", "Model": "Corolla", "Year": 2018,
     "Engine Size": 1.8, "Mileage": 45000, "Fuel Type": "Petrol",
     "Transmission": "Automatic"}
]
```

**Run notebooks in order:**
1. `notebooks/notebook_1_eda_baseline.ipynb`
2. `notebooks/notebook_2_modeling.ipynb`

## Key Takeaways

- Emphasis on stability, not leaderboard scores
- Strict separation of EDA and modeling
- Realistic validation strategy
- Strong focus on interpretability and deployment

## Next Steps (Optional Extensions)

- REST API with FastAPI
- Monitoring prediction drift
- Confidence intervals for predictions
- Model retraining strategy

## Author
**Monday Imoudu**  

Machine Learning Practitioner

