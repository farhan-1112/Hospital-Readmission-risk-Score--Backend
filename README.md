# Hospital Readmission Risk Scorer

## AI-Powered Clinical Decision Support System

Predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days of discharge. Uses machine learning on the Diabetic Patient Readmission Dataset to identify high-risk patients so healthcare professionals can take preventive action.

---

## Project Structure

```
hospital-readmission-ai/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── diabetic_data.csv
│   │   └── IDs_mapping.csv
│   └── processed/                    # Cleaned, model-ready data
│       └── diabetic_clean.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Phase 1: Data cleaning & feature engineering
│   ├── 02_eda.ipynb                  # Phase 2: Exploratory data analysis
│   ├── 03_model_training.ipynb       # Phase 3: Model training & comparison
│   └── 04_optimization.ipynb        # Phase 4: Hyperparameter tuning
├── models/
│   ├── xgboost_readmission_model.pkl # Final trained model
│   ├── scaler.pkl                    # StandardScaler for inference
│   ├── encoders.pkl                  # Label encoders for inference
│   ├── feature_names.pkl            # Feature name list
│   └── best_params.pkl              # Best hyperparameters
├── reports/
│   ├── figures/                      # All generated plots
│   └── model_comparison.csv          # Model comparison results
├── src/
│   └── predict.py                    # Standalone prediction pipeline
├── requirements.txt
├── prd.md                            # Product Requirements Document
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (in order)
python notebooks/run_preprocessing.py      # Phase 1: Clean data
python notebooks/run_eda.py                # Phase 2: EDA plots
python notebooks/run_model_training.py     # Phase 3: Train 4 models
python notebooks/run_optimization.py       # Phase 4: Optimize XGBoost

# Test the prediction pipeline
python src/predict.py

# Start the FastAPI server
python -m uvicorn backend.app:app --reload --port 8000

# Or open notebooks interactively
jupyter notebook
```

### FastAPI Endpoints

Once the server is running at `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (interactive docs) |
| GET | `/model/info` | Model metrics & details |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch prediction (up to 50) |
| POST | `/analyze` | Clinical risk analysis (OpenRouter AI) |
| POST | `/treatment_plan` | Personalized care plan (OpenRouter AI) |
| POST | `/chat` | Conversational clinical assistance (OpenRouter AI) |

## Dataset

- **Source:** Diabetic Patient Readmission Dataset (101,766 records, 50 features)
- **Target:** `readmitted` → binary (readmitted within 30 days = 1, else 0)
- **Class Imbalance:** ~8:1 ratio (handled via class weighting and F2-score optimization)

## ML Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data Preprocessing | ✅ Complete |
| Phase 2 | Exploratory Data Analysis | ✅ Complete |
| Phase 3 | Model Training (4 models) | ✅ Complete |
| Phase 4 | Model Optimization (XGBoost) | ✅ Complete |
| Phase 5 | Prediction Pipeline | ✅ Complete |

## Model Performance

### Comparison (Phase 3)

| Model | Recall | F1 | ROC-AUC | Accuracy |
|-------|--------|-----|---------|----------|
| **XGBoost** | **0.6156** | 0.2490 | 0.6424 | 0.5857 |
| Logistic Regression | 0.0762 | 0.1016 | 0.5715 | 0.8496 |
| LightGBM | 0.0269 | 0.0506 | 0.6678 | 0.8875 |
| Random Forest | 0.0220 | 0.0409 | 0.6239 | 0.8849 |

### Final Optimized Model (Phase 4)

| Metric | Score |
|--------|-------|
| **Recall** | **0.8230** |
| F1-Score | 0.2519 |
| F2-Score | 0.4316 |
| ROC-AUC | 0.6871 |
| Accuracy | 0.4547 |

> **Note:** The model prioritizes **recall** (catching high-risk patients) over accuracy. In healthcare, missing a high-risk patient (false negative) is far more dangerous than a false alarm (false positive).

## Prediction API

```python
from src.predict import predict_readmission

result = predict_readmission({
    'time_in_hospital': 5,
    'num_lab_procedures': 44,
    'num_procedures': 1,
    'num_medications': 13,
    'number_outpatient': 0,
    'number_emergency': 0,
    'number_inpatient': 2,
    'number_diagnoses': 7,
    'age': '[50-60)',
    'admission_type_id': 1,
    'discharge_disposition_id': 1,
    'admission_source_id': 7,
    'diag_1': '250.83',
    'diag_2': '276',
    'diag_3': '250',
    'insulin': 'Up',
    'A1Cresult': '>8',
})

# Output: {'risk_score': 0.5077, 'risk_level': 'Medium'}
```

### Risk Levels
- **High** (≥ 0.7): Immediate intervention recommended
- **Medium** (0.4 - 0.69): Enhanced monitoring suggested
- **Low** (< 0.4): Standard discharge protocol

## Key Design Decisions

1. **Recall is the primary metric** — missing a high-risk patient is far more dangerous than a false alarm
2. **XGBoost** selected as the primary model for its strong tabular data performance
3. **F2-score optimization** — weights recall 2x more than precision, prevents degenerate all-positive predictions
4. **scale_pos_weight** used instead of SMOTE for final model — more robust on test distribution
5. **ICD-9 codes** mapped to 19 clinical disease categories instead of raw encoding
6. **pathlib.Path** used for all file paths — cross-platform compatible

## Disclaimer

This system is intended for clinical assistance only and should not replace professional medical judgment.
