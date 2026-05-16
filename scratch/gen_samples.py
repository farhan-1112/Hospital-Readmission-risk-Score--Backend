
import os
import sys
from pathlib import Path

# Add root to path
PROJECT_ROOT = Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_readmission, _load_artifacts
_load_artifacts()

cases = {
    "High Risk": {
        'time_in_hospital': 8, 'num_lab_procedures': 65, 'num_procedures': 2,
        'num_medications': 25, 'number_outpatient': 2, 'number_emergency': 4,
        'number_inpatient': 6, 'number_diagnoses': 12, 'age': '[70-80)',
        'race': 'Caucasian', 'gender': 'Female', 'admission_type_id': 1,
        'discharge_disposition_id': 3, 'admission_source_id': 7,
        'diag_1': '428', 'diag_2': '250', 'diag_3': '276',
        'A1Cresult': '>8', 'max_glu_serum': 'None', 'insulin': 'Up',
        'change': 'Ch', 'diabetesMed': 'Yes'
    },
    "Medium Risk": {
        'time_in_hospital': 4, 'num_lab_procedures': 35, 'num_procedures': 1,
        'num_medications': 12, 'number_outpatient': 1, 'number_emergency': 0,
        'number_inpatient': 1, 'number_diagnoses': 6, 'age': '[50-60)',
        'race': 'AfricanAmerican', 'gender': 'Male', 'admission_type_id': 2,
        'discharge_disposition_id': 1, 'admission_source_id': 1,
        'diag_1': '250', 'diag_2': '276', 'diag_3': '401',
        'A1Cresult': 'Norm', 'max_glu_serum': 'None', 'insulin': 'Steady',
        'change': 'No', 'diabetesMed': 'Yes'
    },
    "Low Risk": {
        'time_in_hospital': 2, 'num_lab_procedures': 20, 'num_procedures': 0,
        'num_medications': 8, 'number_outpatient': 0, 'number_emergency': 0,
        'number_inpatient': 0, 'number_diagnoses': 3, 'age': '[30-40)',
        'race': 'Other', 'gender': 'Female', 'admission_type_id': 3,
        'discharge_disposition_id': 1, 'admission_source_id': 1,
        'diag_1': '250', 'diag_2': 'None', 'diag_3': 'None',
        'A1Cresult': 'None', 'max_glu_serum': 'None', 'insulin': 'No',
        'change': 'No', 'diabetesMed': 'No'
    }
}

for name, data in cases.items():
    res = predict_readmission(data)
    print(f"{name}: Score {res['risk_score']:.4f} -> {res['risk_level']}")
