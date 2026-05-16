# Product Requirements Document (PRD)

# AI-Powered Hospital Readmission Risk Prediction System

---

# 1. Project Overview

## Project Name

Hospital Readmission Risk Scorer

## Project Type

AI-Powered Clinical Decision Support System

## Primary Goal

Build an AI healthcare platform that predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days after discharge.

The system should:

* Predict readmission risk accurately using machine learning
* Generate AI-powered clinical summaries using OpenRouter AI API
* Provide recommendations to healthcare professionals
* Display predictions through a modern healthcare dashboard

---

# 2. Problem Statement

Hospital readmissions are expensive and dangerous.

Many diabetic patients return to the hospital within 30 days due to:

* Poor discharge planning
* Unstable glucose conditions
* Medication issues
* Lack of follow-up care

Hospitals need a system that can:

* Identify high-risk patients early
* Help doctors take preventive actions
* Reduce healthcare costs
* Improve patient outcomes

---

# 3. Proposed Solution

Develop a machine learning system that:

1. Accepts patient hospital data
2. Predicts readmission risk
3. Generates AI clinical explanations
4. Displays recommendations visually
5. Helps healthcare staff make better decisions

---

# 4. Core Objectives

## Primary Objectives

* Predict hospital readmission risk accurately
* Reduce false negatives in healthcare predictions
* Provide explainable AI summaries
* Build a scalable ML pipeline
* Deploy the system using FastAPI
* Create a modern AI healthcare dashboard

## Secondary Objectives

* Add AI-generated recommendations
* Enable PDF export of reports
* Visualize patient risk levels
* Support future real-time hospital integration

---

# 5. Learning Type

| Category      | Type                |
| ------------- | ------------------- |
| Learning      | Supervised Learning |
| Problem Type  | Classification      |
| Training Type | Batch Learning      |
| Domain        | Healthcare AI       |

---

# 6. Dataset Information

## Dataset

Diabetic Patient Readmission Dataset

## Files

### Main Dataset

* diabetic_data.csv

### Mapping Dataset

* IDs_mapping.csv

---

# 7. Target Variable

## Target Column

readmitted

## Original Values

| Value | Meaning                   |
| ----- | ------------------------- |
| <30   | Readmitted within 30 days |
| >30   | Readmitted after 30 days  |
| NO    | Not readmitted            |

## Binary Conversion

| Original | Converted |
| -------- | --------- |
| <30      | 1         |
| >30      | 0         |
| NO       | 0         |

---

# 8. Important Features

## Patient Information

* race
* gender
* age

## Admission Details

* admission_type_id
* discharge_disposition_id
* admission_source_id
* time_in_hospital

## Clinical Features

* num_lab_procedures
* num_procedures
* num_medications
* number_outpatient
* number_emergency
* number_inpatient
* number_diagnoses

## Diabetes Care Features

* A1Cresult
* max_glu_serum
* insulin
* change
* diabetesMed
* diag_1
* diag_2
* diag_3

---

# 9. Machine Learning Requirements

## Functional Requirements

The ML system must:

* Train on diabetic patient data
* Predict readmission risk probability
* Handle missing values
* Handle categorical variables
* Handle class imbalance
* Save trained models
* Support API-based prediction

---

# 10. Data Preprocessing Requirements

## Missing Values

Replace:

* ? → NaN

## Remove Low-Value Columns

Drop:

* encounter_id
* patient_nbr
* weight
* payer_code
* medical_specialty

## Encoding

Use:

* Label Encoding
  or
* OneHotEncoding

## Scaling

Optional:

* StandardScaler

## Imbalanced Data Handling

Use:

* SMOTE
* class_weight

---

# 11. Recommended ML Models

| Model               | Priority    |
| ------------------- | ----------- |
| Logistic Regression | Baseline    |
| Random Forest       | Good        |
| XGBoost             | Recommended |
| LightGBM            | Advanced    |

## Final Recommended Model

XGBoost Classifier

Reason:

* Excellent tabular performance
* High recall
* Good healthcare prediction capability
* Strong classification performance

---

# 12. Model Evaluation Metrics

## Important Metrics

| Metric   | Importance |
| -------- | ---------- |
| Recall   | Very High  |
| F1-Score | High       |
| ROC-AUC  | High       |
| Accuracy | Medium     |

## Why Recall Matters

Healthcare systems should avoid false negatives.

A false negative means:

* high-risk patient predicted as low risk

This can be dangerous.

---

# 13. Expected Output

## Prediction Output

Example:

{
"risk_score": 0.89,
"risk_level": "High"
}

---

# 14. AI Summary Generation

## AI Provider

OpenRouter AI API

## Purpose

Generate:

* Clinical summaries
* Risk explanations
* Recommendations
* Follow-up suggestions

---

# 15. OpenRouter AI Requirements

## Input to OpenRouter

* Patient details
* ML prediction result
* Risk probability
* Important features

## Output from OpenRouter

* Risk explanation
* Contributing factors
* Care recommendations
* Monitoring suggestions

---

# 16. Example AI Summary

Patient demonstrates elevated readmission risk due to:

* multiple prior inpatient visits
* elevated A1C levels
* insulin dosage changes

Recommended Actions:

* schedule follow-up within 7 days
* monitor blood glucose regularly
* review medication adherence

---

# 17. System Architecture

## Recommended Architecture

Frontend (React)
↓
FastAPI Backend
↓
XGBoost ML Model
↓
Prediction Result
↓
OpenRouter API
↓
AI Summary Response

---

# 18. Backend Requirements

## Framework

FastAPI

## Responsibilities

* Load ML model
* Accept patient input
* Run prediction
* Generate probability score
* Call OpenRouter API
* Return JSON response

---

# 19. Frontend Requirements

## Framework

React + Tailwind CSS

## UI Style

Inspired by Sixth AI design system.

## Required Pages

### 1. Landing Page

* Hero section
* Features
* Architecture
* CTA button

### 2. Prediction Dashboard

* Patient input form
* Multi-step tabs
* Predict button

### 3. Result Dashboard

* Risk score
* AI summary
* Recommendations
* Confidence score

### 4. Analytics Dashboard

* Charts
* Risk trends
* Patient statistics

---

# 20. UI Design Requirements

## Design Style

* Dark modern dashboard
* Glassmorphism cards
* Orange accent glow
* Rounded corners
* AI healthcare theme

## Color System

| Purpose     | Color   |
| ----------- | ------- |
| Background  | #0A0A0A |
| Cards       | #111111 |
| Accent      | Orange  |
| High Risk   | Red     |
| Medium Risk | Yellow  |
| Low Risk    | Green   |

---

# 21. Risk Visualization Requirements

## Required Components

* Circular risk meter
* Risk percentage
* Risk level badge
* AI summary panel
* Recommendation panel
* Confidence score

---

# 22. Explainable AI Features

## Future Scope

Use SHAP explainability.

Display:

* top contributing features
* feature importance
* risk factor visualization

---

# 23. Deployment Requirements

## Frontend Deployment

Vercel

## Backend Deployment

Render or Railway

## Model Hosting

FastAPI server

---

# 24. MLOps Requirements

## Tools

| Tool           | Purpose             |
| -------------- | ------------------- |
| GitHub         | Version control     |
| DVC            | Dataset versioning  |
| MLflow         | Experiment tracking |
| Docker         | Containerization    |
| GitHub Actions | CI/CD               |

---

# 25. Security Requirements

## Important Requirements

* Do not store sensitive patient data
* Use HTTPS APIs
* Protect API keys
* Add medical disclaimer

---

# 26. Disclaimer

This system is intended for clinical assistance only and should not replace professional medical judgment.

---

# 27. Success Criteria

The project is successful if:

* Model achieves strong recall
* Predictions are reliable
* AI summaries are meaningful
* Frontend is responsive
* API integration works correctly
* System demonstrates real healthcare value

---

# 28. Recommended Development Phases

## Phase 1

Dataset preprocessing

## Phase 2

EDA and feature engineering

## Phase 3

Model training

## Phase 4

Model optimization

## Phase 5

FastAPI integration

## Phase 6

OpenRouter API integration

## Phase 7

Frontend development

## Phase 8

Deployment and testing

---

# 29. Recommended Folder Structure

hospital-readmission-ai/
│
├── data/
├── notebooks/
├── models/
├── src/
├── backend/
├── frontend/
├── reports/
├── requirements.txt
└── README.md

---

# 30. Antigravity Development Prompt

Build an AI-powered healthcare web application called “Hospital Readmission Risk Scorer”.

The platform should:

* predict diabetic patient readmission risk within 30 days
* use supervised machine learning classification
* use XGBoost for prediction
* use FastAPI backend
* use React frontend
* generate AI summaries using OpenRouter AI API
* display risk scores visually
* provide recommendations and explanations
* use modern dark SaaS UI inspired by Sixth AI
* include dashboards, analytics, and prediction pages
* support future deployment and scalability

The system should prioritize:

* high recall
* explainable AI
* healthcare usability
* modern UI/UX
* scalable architecture
* clean API design

---

# 31. Final Project Positioning

This project should be presented as:

AI-Powered Clinical Decision Support System for Predicting Diabetic Patient Readmission Risk.

Not just:

* prediction model
* hospital form app
* simple ML project

This positioning sounds more professional and industry-level during hackathons and evaluations.
