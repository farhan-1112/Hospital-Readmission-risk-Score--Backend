"""
Phase 1 — Data Preprocessing Pipeline
Hospital Readmission Risk Scorer

Transforms raw diabetic patient data into a clean, model-ready dataset.
Run this script from the project root directory.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data dir: {DATA_RAW}")

# ============================================================
# Step 1: Load Raw Data
# ============================================================
print("\n[Step 1] Loading raw data...")
df = pd.read_csv(DATA_RAW / 'diabetic_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Total records: {df.shape[0]:,}")

# ============================================================
# Step 2: Handle Missing Values ('?' → NaN)
# ============================================================
print("\n[Step 2] Replacing '?' with NaN...")
df = df.replace('?', np.nan)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).query('missing_count > 0').sort_values('missing_pct', ascending=False)
print("Columns with missing values:")
print(missing_df)

# ============================================================
# Step 3: Drop Low-Value Columns
# ============================================================
print("\n[Step 3] Dropping low-value columns...")
cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
existing_drops = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_drops)
print(f"Shape after dropping: {df.shape}")

# ============================================================
# Step 4: Create Binary Target Variable
# ============================================================
print("\n[Step 4] Creating binary target variable...")
print("Original 'readmitted' distribution:")
print(df['readmitted'].value_counts())

df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
print(f"\nBinary target distribution:")
print(df['readmitted_binary'].value_counts())
imbalance = df['readmitted_binary'].value_counts()[0] / df['readmitted_binary'].value_counts()[1]
print(f"Imbalance ratio: {imbalance:.1f}:1")
df = df.drop(columns=['readmitted'])

# ============================================================
# Step 5: Fill Missing Values
# ============================================================
print("\n[Step 5] Handling remaining missing values...")

# Convert all object/str columns to proper string type for consistent handling
for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
        df[col] = df[col].astype('object')

# Identify column types after dtype normalization
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Fill numeric with median (CoW-safe: use assignment, not inplace)
for col in numeric_cols:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {col} (numeric) with median: {median_val}")

# Fill categorical with mode (CoW-safe)
for col in categorical_cols:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  Filled {col} (categorical, {null_count:,} missing) with mode: '{mode_val}'")

remaining_nulls = df.isnull().sum().sum()
assert remaining_nulls == 0, f"Still have {remaining_nulls} missing values!"
print(f"\n[OK] All missing values handled. Remaining NaN count: {remaining_nulls}")

# ============================================================
# Step 6: Map ICD Diagnosis Codes to Clinical Categories
# ============================================================
print("\n[Step 6] Mapping ICD diagnosis codes to clinical categories...")


def map_icd_to_category(code):
    """
    Map ICD-9 codes to high-level disease categories.
    Based on standard ICD-9-CM groupings used in clinical informatics.
    """
    if pd.isna(code) or str(code).strip() == '':
        return 'Other'

    code = str(code).strip()

    # V-codes: supplementary classification (aftercare, screening, etc.)
    if code.startswith(('V', 'v')):
        return 'Supplementary'
    # E-codes: external causes of injury
    if code.startswith(('E', 'e')):
        return 'External_Causes'

    try:
        n = float(code)
    except ValueError:
        return 'Other'

    # Standard ICD-9 numeric ranges
    if 390 <= n <= 459 or n == 785:
        return 'Circulatory'
    elif 460 <= n <= 519 or n == 786:
        return 'Respiratory'
    elif 520 <= n <= 579 or n == 787:
        return 'Digestive'
    elif 250 <= n < 251:
        return 'Diabetes'
    elif 800 <= n <= 999:
        return 'Injury'
    elif 710 <= n <= 739:
        return 'Musculoskeletal'
    elif 580 <= n <= 629 or n == 788:
        return 'Genitourinary'
    elif 140 <= n <= 239:
        return 'Neoplasms'
    elif 240 <= n < 250 or 251 <= n <= 279:
        return 'Endocrine_Other'
    elif 680 <= n <= 709 or n == 782:
        return 'Skin'
    elif 1 <= n <= 139:
        return 'Infectious'
    elif 280 <= n <= 289:
        return 'Blood'
    elif 290 <= n <= 319:
        return 'Mental'
    elif 320 <= n <= 389:
        return 'Nervous_System'
    elif 630 <= n <= 679:
        return 'Pregnancy'
    elif 740 <= n <= 759:
        return 'Congenital'
    elif 760 <= n <= 779:
        return 'Perinatal'
    elif 780 <= n <= 799:
        return 'Symptoms'
    else:
        return 'Other'


for diag_col in ['diag_1', 'diag_2', 'diag_3']:
    col_name = f'{diag_col}_category'
    df[col_name] = df[diag_col].apply(map_icd_to_category)
    print(f"\n{col_name} distribution (top 5):")
    print(df[col_name].value_counts().head(5))

df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
print(f"\nShape after ICD mapping: {df.shape}")

# ============================================================
# Step 7: Encode Categorical Features
# ============================================================
print("\n[Step 7] Encoding categorical features...")

label_encoders = {}

# --- Ordinal encoding ---
# Age brackets have a natural order
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
if 'age' in df.columns:
    le_age = LabelEncoder()
    le_age.fit(age_order)
    df['age'] = df['age'].apply(lambda x: x if x in age_order else age_order[-1])
    df['age'] = le_age.transform(df['age'])
    label_encoders['age'] = le_age
    print(f"age encoded: {dict(zip(age_order, le_age.transform(age_order)))}")

# A1Cresult
if 'A1Cresult' in df.columns:
    le_a1c = LabelEncoder()
    a1c_values = sorted(df['A1Cresult'].unique().tolist())
    le_a1c.fit(a1c_values)
    df['A1Cresult'] = le_a1c.transform(df['A1Cresult'])
    label_encoders['A1Cresult'] = le_a1c
    print(f"A1Cresult encoded: {dict(zip(a1c_values, le_a1c.transform(a1c_values)))}")

# max_glu_serum
if 'max_glu_serum' in df.columns:
    le_glu = LabelEncoder()
    glu_values = sorted(df['max_glu_serum'].unique().tolist())
    le_glu.fit(glu_values)
    df['max_glu_serum'] = le_glu.transform(df['max_glu_serum'])
    label_encoders['max_glu_serum'] = le_glu
    print(f"max_glu_serum encoded: {dict(zip(glu_values, le_glu.transform(glu_values)))}")

print(f"\nTotal label encoders: {len(label_encoders)}")

# --- One-hot encoding for nominal features ---
remaining_cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nOne-hot encoding {len(remaining_cat_cols)} columns...")
for col in remaining_cat_cols:
    print(f"  {col}: {df[col].nunique()} unique values")

df = pd.get_dummies(df, columns=remaining_cat_cols, drop_first=True, dtype=int)
print(f"\nShape after one-hot encoding: {df.shape}")

# ============================================================
# Step 8: Scale Continuous Numerical Features
# ============================================================
print("\n[Step 8] Scaling continuous numerical features...")

continuous_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

existing_continuous = [c for c in continuous_cols if c in df.columns]
print(f"Scaling {len(existing_continuous)} features: {existing_continuous}")

scaler = StandardScaler()
df[existing_continuous] = scaler.fit_transform(df[existing_continuous])

print(f"\nPost-scaling statistics:")
print(df[existing_continuous].describe().round(3).loc[['mean', 'std', 'min', 'max']])

# ============================================================
# Step 9: Save Processed Data and Artifacts
# ============================================================
print("\n[Step 9] Saving processed data and artifacts...")

output_path = DATA_PROCESSED / 'diabetic_clean.csv'
df.to_csv(output_path, index=False)
print(f"[OK] Cleaned dataset saved to: {output_path}")
print(f"   Shape: {df.shape}")
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

scaler_path = MODELS_DIR / 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"[OK] StandardScaler saved to: {scaler_path}")

encoders_path = MODELS_DIR / 'encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"[OK] Label encoders saved to: {encoders_path}")

# Save feature names for later use in prediction pipeline
feature_names = [c for c in df.columns if c != 'readmitted_binary']
feature_path = MODELS_DIR / 'feature_names.pkl'
joblib.dump(feature_names, feature_path)
print(f"[OK] Feature names saved to: {feature_path}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1 — DATA PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\nDataset Summary:")
print(f"   Original shape: (101766, 50)")
print(f"   Cleaned shape:  {df.shape}")
print(f"   Target column:  readmitted_binary")
print(f"   Class 0 (not readmitted <30d): {(df['readmitted_binary'] == 0).sum():,}")
print(f"   Class 1 (readmitted <30d):     {(df['readmitted_binary'] == 1).sum():,}")
print(f"   Imbalance ratio:               {imbalance:.1f}:1")
print(f"\nFiles created:")
print(f"   - data/processed/diabetic_clean.csv")
print(f"   - models/scaler.pkl")
print(f"   - models/encoders.pkl")
print(f"   - models/feature_names.pkl")
print(f"\nNext: Phase 2 -- Exploratory Data Analysis")
print(f"   We will analyze feature distributions, correlations,")
print(f"   and readmission patterns across clinical variables.")
