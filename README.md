# Aviation Safety Data Projects

This repository contains a collection of data analysis and data science projects built with the Transportation Safety Board of Canada (TSB) aviation occurrence dataset.  Each project explores a different aspect of the data, from predictive modeling to geospatial analysis and natural‑language processing.  Together they demonstrate a comprehensive approach to understanding aviation safety and building a robust data portfolio.

## Table of Contents

1. [Accident Severity Prediction Model](#1-accident-severity-prediction-model)
2. [Unsupervised Clustering of Accident Profiles](#2-unsupervised-clustering-of-accident-profiles)
3. [Geospatial Hotspot & Risk-Score Analysis](#3-geospatial-hotspot--risk-score-analysis)
4. [Survival Analysis of Evacuation & Rescue](#4-survival-analysis-of-evacuation--rescue)
5. [Topic Modeling for Accident Narratives](#5-topic-modeling-for-accident-narratives)
6. [Aviation Accident Trend & Seasonality Analysis](#6-aviation-accident-trend--seasonality-analysis)
7. [Aircraft Type & Operation Breakdown](#7-aircraft-type--operation-breakdown)

---

## 1. Accident Severity Prediction Model

### Description
This project builds a machine-learning classifier to predict whether an aviation accident will result in fatalities.  By combining information from multiple TSB tables—occurrence details, aircraft characteristics:contentReference[oaicite:0]{index=0}, weather conditions:contentReference[oaicite:1]{index=1}, flight phases:contentReference[oaicite:2]{index=2} and survivability factors:contentReference[oaicite:3]{index=3}—the model estimates the probability that an accident will be fatal.

### Data
- **Occurrence table**: date, time, location, occurrence type, classification, weather:contentReference[oaicite:4]{index=4}.
- **Aircraft table**: type, make/model, operator type, flight plan, departure/destination:contentReference[oaicite:5]{index=5}.
- **Events & Phases table**: flight phases and events:contentReference[oaicite:6]{index=6}.
- **Survivability table**: evacuation, survival devices, locator systems:contentReference[oaicite:7]{index=7}.
- **Injuries table**: injury counts and severity:contentReference[oaicite:8]{index=8}.

### Methodology
1. **Merge tables** on the occurrence identifier to create a rich dataset.
2. **Feature engineering**: encode categorical variables, scale numeric features and create a binary target (`fatal`).
3. **Handle class imbalance** using SMOTE or class weights.
4. **Model training**: train logistic regression, random forest and XGBoost models; perform hyperparameter tuning.
5. **Evaluation**: calculate accuracy, precision, recall, F1‑score and ROC‑AUC; plot confusion matrices and ROC curves.
6. **Interpretation**: extract feature importance and SHAP values to understand which variables drive fatal outcomes.
7. **Deployment**: save the model (`.pkl`) and expose it via a REST API (e.g., FastAPI) or interactive web app.

### Results
- Best model: [x] (e.g., Random Forest with ROC‑AUC = [x]).
- Key predictors: operator type, flight phase, presence of survival devices, weather conditions.
- See `/severity_model/` for notebooks, code and saved models.

### Usage
```bash
cd severity_model
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/severity_model.ipynb
# To run the API:
uvicorn app.main:app --reload
