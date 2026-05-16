"""
Hospital Readmission Risk Scorer - Prediction Pipeline
Standalone prediction module for FastAPI deployment.

Usage:
    from predict import predict_readmission, validate_input
    
    patient = {
        'time_in_hospital': 5,
        'num_lab_procedures': 44,
        'num_procedures': 1,
        'num_medications': 13,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 2,
        'number_diagnoses': 7,
        'age': '[50-60)',
        'race': 'Caucasian',
        'gender': 'Female',
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
        'A1Cresult': '>8',
        'max_glu_serum': 'None',
        'insulin': 'Up',
        'change': 'Ch',
        'diabetesMed': 'Yes',
        'diag_1': '250.83',
        'diag_2': '276',
        'diag_3': '250',
        'metformin': 'No',
        'repaglinide': 'No',
        'nateglinide': 'No',
        'chlorpropamide': 'No',
        'glimepiride': 'No',
        'acetohexamide': 'No',
        'glipizide': 'No',
        'glyburide': 'No',
        'tolbutamide': 'No',
        'pioglitazone': 'No',
        'rosiglitazone': 'No',
        'acarbose': 'No',
        'miglitol': 'No',
        'troglitazone': 'No',
        'tolazamide': 'No',
        'examide': 'No',
        'citoglipton': 'No',
        'glyburide-metformin': 'No',
        'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No',
        'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No',
    }
    
    result = predict_readmission(patient)
    # {'risk_score': 0.72, 'risk_level': 'High'}
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
# Resolve model paths relative to this file's location
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_MODELS_DIR = _PROJECT_ROOT / 'models'

# Risk thresholds (configurable)
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# Required fields for a valid patient input
REQUIRED_FIELDS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'age',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'diag_1', 'diag_2', 'diag_3'
]

# Continuous features that need scaling
CONTINUOUS_FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

# Default values for optional medication fields
MEDICATION_DEFAULTS = {
    'metformin': 'No', 'repaglinide': 'No', 'nateglinide': 'No',
    'chlorpropamide': 'No', 'glimepiride': 'No', 'acetohexamide': 'No',
    'glipizide': 'No', 'glyburide': 'No', 'tolbutamide': 'No',
    'pioglitazone': 'No', 'rosiglitazone': 'No', 'acarbose': 'No',
    'miglitol': 'No', 'troglitazone': 'No', 'tolazamide': 'No',
    'examide': 'No', 'citoglipton': 'No', 'insulin': 'No',
    'glyburide-metformin': 'No', 'glipizide-metformin': 'No',
    'glimepiride-pioglitazone': 'No', 'metformin-rosiglitazone': 'No',
    'metformin-pioglitazone': 'No',
    'change': 'No', 'diabetesMed': 'No',
    'race': 'Caucasian', 'gender': 'Female',
    'A1Cresult': 'None', 'max_glu_serum': 'None'
}


# ============================================================
# Model Loading (lazy singleton pattern)
# ============================================================
_model = None
_scaler = None
_encoders = None
_feature_names = None


def _load_artifacts():
    """Load model, scaler, encoders, and feature names from disk."""
    global _model, _scaler, _encoders, _feature_names
    
    if _model is not None:
        return  # Already loaded
    
    model_path = _MODELS_DIR / 'model.pkl'
    scaler_path = _MODELS_DIR / 'scaler.pkl'
    encoders_path = _MODELS_DIR / 'encoder.pkl'
    features_path = _MODELS_DIR / 'feature_names.pkl'
    
    # Verify all artifacts exist
    for path, name in [(model_path, 'model'), (scaler_path, 'scaler'),
                       (encoders_path, 'encoders'), (features_path, 'feature_names')]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} artifact: {path}\n"
                f"Run the training pipeline (Phases 1-4) first."
            )
    
    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    _encoders = joblib.load(encoders_path)
    _feature_names = joblib.load(features_path)
    
    print(f"Loaded model from: {model_path}")
    print(f"Expected features: {len(_feature_names)}")


# ============================================================
# ICD Code Mapping (same as training pipeline)
# ============================================================
def _map_icd_to_category(code):
    """Map ICD-9 codes to high-level disease categories."""
    if pd.isna(code) or str(code).strip() == '':
        return 'Other'
    code = str(code).strip()
    if code.startswith(('V', 'v')):
        return 'Supplementary'
    if code.startswith(('E', 'e')):
        return 'External_Causes'
    try:
        n = float(code)
    except ValueError:
        return 'Other'
    
    if 390 <= n <= 459 or n == 785: return 'Circulatory'
    elif 460 <= n <= 519 or n == 786: return 'Respiratory'
    elif 520 <= n <= 579 or n == 787: return 'Digestive'
    elif 250 <= n < 251: return 'Diabetes'
    elif 800 <= n <= 999: return 'Injury'
    elif 710 <= n <= 739: return 'Musculoskeletal'
    elif 580 <= n <= 629 or n == 788: return 'Genitourinary'
    elif 140 <= n <= 239: return 'Neoplasms'
    elif 240 <= n < 250 or 251 <= n <= 279: return 'Endocrine_Other'
    elif 680 <= n <= 709 or n == 782: return 'Skin'
    elif 1 <= n <= 139: return 'Infectious'
    elif 280 <= n <= 289: return 'Blood'
    elif 290 <= n <= 319: return 'Mental'
    elif 320 <= n <= 389: return 'Nervous_System'
    elif 630 <= n <= 679: return 'Pregnancy'
    elif 740 <= n <= 759: return 'Congenital'
    elif 760 <= n <= 779: return 'Perinatal'
    elif 780 <= n <= 799: return 'Symptoms'
    else: return 'Other'


# ============================================================
# Input Validation
# ============================================================
def validate_input(patient_dict: dict) -> dict:
    """
    Validate a patient input dictionary.
    
    Returns:
        dict with 'valid' (bool) and 'errors' (list of strings)
    """
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in patient_dict:
            errors.append(f"Missing required field: '{field}'")
    
    # Validate numeric fields
    numeric_fields = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                      'num_medications', 'number_outpatient', 'number_emergency',
                      'number_inpatient', 'number_diagnoses',
                      'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    
    for field in numeric_fields:
        if field in patient_dict:
            try:
                val = float(patient_dict[field])
                if val < 0:
                    errors.append(f"'{field}' must be non-negative, got {val}")
            except (ValueError, TypeError):
                errors.append(f"'{field}' must be numeric, got '{patient_dict[field]}'")
    
    # Validate age bracket
    valid_ages = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                  '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    if 'age' in patient_dict and patient_dict['age'] not in valid_ages:
        errors.append(f"'age' must be one of {valid_ages}, got '{patient_dict['age']}'")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }


# ============================================================
# Preprocessing (mirrors training pipeline)
# ============================================================
def _preprocess_patient(patient_dict: dict) -> pd.DataFrame:
    """
    Apply the same preprocessing steps used during training to a single patient record.
    Returns a DataFrame with features matching the training schema.
    """
    # Fill defaults for optional fields
    patient = {**MEDICATION_DEFAULTS, **patient_dict}
    
    # Create single-row DataFrame
    df = pd.DataFrame([patient])
    
    # Map ICD diagnosis codes to categories
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        col_name = f'{diag_col}_category'
        df[col_name] = df[diag_col].apply(_map_icd_to_category)
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'], errors='ignore')
    
    # Ordinal encoding using saved encoders
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    
    if 'age' in df.columns and 'age' in _encoders:
        df['age'] = df['age'].apply(lambda x: x if x in age_order else age_order[-1])
        df['age'] = _encoders['age'].transform(df['age'])
    
    if 'A1Cresult' in df.columns and 'A1Cresult' in _encoders:
        known_values = list(_encoders['A1Cresult'].classes_)
        df['A1Cresult'] = df['A1Cresult'].apply(lambda x: x if x in known_values else known_values[0])
        df['A1Cresult'] = _encoders['A1Cresult'].transform(df['A1Cresult'])
    
    if 'max_glu_serum' in df.columns and 'max_glu_serum' in _encoders:
        known_values = list(_encoders['max_glu_serum'].classes_)
        df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x: x if x in known_values else known_values[0])
        df['max_glu_serum'] = _encoders['max_glu_serum'].transform(df['max_glu_serum'])
    
    # One-hot encode remaining categorical columns
    # IMPORTANT: We do NOT use drop_first=True here because for a single-row DataFrame,
    # it would drop the only category present, leading to all 0s for that feature.
    # Instead, we generate all dummies and let the reindex step handle dropping
    # the reference categories (since they won't be in _feature_names).
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    
    # Ensure numeric types for continuous columns
    for col in CONTINUOUS_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Scale continuous features
    existing_continuous = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    if existing_continuous:
        df[existing_continuous] = _scaler.transform(df[existing_continuous])
    
    # Align columns to match training feature set exactly
    # reindex handles both adding missing columns (with 0) and removing extra columns
    df = df.reindex(columns=_feature_names, fill_value=0)
    
    return df


# ============================================================
# Main Prediction Function
# ============================================================
def predict_readmission(patient_dict: dict) -> dict:
    """
    Predict 30-day readmission risk for a single patient.
    
    Args:
        patient_dict: Dictionary of patient features
        
    Returns:
        {
            'risk_score': float (0.0 to 1.0),
            'risk_level': 'High' | 'Medium' | 'Low',
            'details': {
                'threshold_high': 0.7,
                'threshold_medium': 0.4,
                'model': 'XGBoost (optimized)'
            }
        }
    """
    # Load artifacts if not already loaded
    _load_artifacts()
    
    # Validate input
    validation = validate_input(patient_dict)
    if not validation['valid']:
        return {
            'error': True,
            'message': 'Invalid input',
            'errors': validation['errors']
        }
    
    # Preprocess
    features = _preprocess_patient(patient_dict)
    
    # Predict probability
    risk_score = float(_model.predict_proba(features)[:, 1][0])
    
    # Determine risk level
    if risk_score >= HIGH_RISK_THRESHOLD:
        risk_level = 'High'
    elif risk_score >= MEDIUM_RISK_THRESHOLD:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return {
        'risk_score': round(risk_score, 4),
        'risk_level': risk_level,
        'details': {
            'threshold_high': HIGH_RISK_THRESHOLD,
            'threshold_medium': MEDIUM_RISK_THRESHOLD,
            'model': 'XGBoost (optimized)'
        }
    }


# ============================================================
# CLI Entry Point (for testing)
# ============================================================
if __name__ == '__main__':
    print("Hospital Readmission Risk Scorer - Prediction Pipeline")
    print("=" * 55)
    
    # Test with a sample patient
    sample_patient = {
        'time_in_hospital': 5,
        'num_lab_procedures': 44,
        'num_procedures': 1,
        'num_medications': 13,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 2,
        'number_diagnoses': 7,
        'age': '[50-60)',
        'race': 'Caucasian',
        'gender': 'Female',
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
        'A1Cresult': '>8',
        'max_glu_serum': 'Norm',
        'insulin': 'Up',
        'change': 'Ch',
        'diabetesMed': 'Yes',
        'diag_1': '250.83',
        'diag_2': '276',
        'diag_3': '250',
    }
    
    print("\nSample patient input:")
    for k, v in sample_patient.items():
        print(f"  {k}: {v}")
    
    print("\nValidation:")
    validation = validate_input(sample_patient)
    print(f"  Valid: {validation['valid']}")
    if not validation['valid']:
        print(f"  Errors: {validation['errors']}")
    
    print("\nPrediction:")
    result = predict_readmission(sample_patient)
    print(f"  Risk Score: {result['risk_score']}")
    print(f"  Risk Level: {result['risk_level']}")
    
    # Test with a high-risk patient profile
    high_risk_patient = {
        'time_in_hospital': 12,
        'num_lab_procedures': 72,
        'num_procedures': 4,
        'num_medications': 21,
        'number_outpatient': 2,
        'number_emergency': 3,
        'number_inpatient': 5,
        'number_diagnoses': 9,
        'age': '[80-90)',
        'race': 'AfricanAmerican',
        'gender': 'Male',
        'admission_type_id': 1,
        'discharge_disposition_id': 3,
        'admission_source_id': 7,
        'A1Cresult': '>8',
        'max_glu_serum': '>300',
        'insulin': 'Up',
        'change': 'Ch',
        'diabetesMed': 'Yes',
        'diag_1': '250.01',
        'diag_2': '428',
        'diag_3': '276',
    }
    
    print("\n\nHigh-risk patient profile:")
    result2 = predict_readmission(high_risk_patient)
    print(f"  Risk Score: {result2['risk_score']}")
    print(f"  Risk Level: {result2['risk_level']}")
    
    print("\n\nPrediction pipeline is working correctly!")
    print("Ready for FastAPI integration.")
