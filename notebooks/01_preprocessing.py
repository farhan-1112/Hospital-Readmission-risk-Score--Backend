# %% [markdown]
# # Phase 1 -- Data Preprocessing
# ## Hospital Readmission Risk Scorer
# 
# This notebook handles the entire data preprocessing pipeline for the diabetic patient
# readmission prediction system. We transform raw clinical data into a clean, model-ready
# dataset following healthcare ML best practices.
#
# **What we do:**
# 1. Load and inspect the raw data
# 2. Clean missing values and drop low-value columns
# 3. Engineer the binary target variable
# 4. Encode categorical features appropriately
# 5. Map ICD diagnosis codes to clinical categories
# 6. Scale continuous numerical features
# 7. Save the processed dataset

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import joblib

warnings.filterwarnings('ignore')

# Project paths -- using pathlib for cross-platform compatibility
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## Step 1 -- Load Raw Data
# 
# We load the main diabetic dataset (~101k records, 50 features) and the IDs mapping file.
# The mapping file contains human-readable descriptions for admission_type_id,
# discharge_disposition_id, and admission_source_id.

# %%
df = pd.read_csv(DATA_RAW / 'diabetic_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Total records: {df.shape[0]:,}")
print(f"Total features: {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")

# %%
print("Data types:")
print(df.dtypes.value_counts())
df.head(3)

# %%
# Parse IDs mapping file -- it has 3 sections separated by empty rows
def parse_ids_mapping(filepath):
    """Parse the IDs_mapping.csv which has multiple tables stacked vertically."""
    mappings = {}
    current_key = None
    current_data = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == ',':
                if current_key and current_data:
                    mappings[current_key] = current_data
                current_key = None
                current_data = {}
                continue
            
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            
            col1, col2 = parts[0].strip(), parts[1].strip()
            
            if col1.endswith('_id') and col2 == 'description':
                current_key = col1
                current_data = {}
            elif current_key:
                try:
                    current_data[int(col1)] = col2
                except ValueError:
                    continue
    
    if current_key and current_data:
        mappings[current_key] = current_data
    
    return mappings

id_mappings = parse_ids_mapping(DATA_RAW / 'IDs_mapping.csv')
for key, mapping in id_mappings.items():
    print(f"\n{key}: {len(mapping)} entries")
    for k, v in list(mapping.items())[:3]:
        print(f"  {k} -> {v}")

# %% [markdown]
# ## Step 2 -- Handle Missing Values
# 
# The dataset uses `?` as a placeholder for missing values instead of proper NaN.
# We convert all `?` to NaN first, then assess the damage.

# %%
df = df.replace('?', np.nan)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).query('missing_count > 0').sort_values('missing_pct', ascending=False)

print("Columns with missing values:")
print(missing_df)

# %% [markdown]
# ## Step 3 -- Drop Low-Value Columns
# 
# Several columns either have too many missing values, contain identifiers that
# carry no predictive signal, or would cause data leakage:
# 
# - `encounter_id`, `patient_nbr` -- identifiers, not features
# - `weight` -- ~97% missing, unreliable
# - `payer_code` -- insurance info, high missing rate
# - `medical_specialty` -- ~49% missing, too sparse

# %%
cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
existing_drops = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_drops)
print(f"Shape after dropping: {df.shape}")

# %% [markdown]
# ## Step 4 -- Create Binary Target Variable
# 
# | Original | Binary | Meaning |
# |----------|--------|---------|
# | `<30` | 1 | Readmitted within 30 days (high risk) |
# | `>30` | 0 | Readmitted after 30 days |
# | `NO` | 0 | Not readmitted |

# %%
print("Original 'readmitted' distribution:")
print(df['readmitted'].value_counts())

df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
print(f"\nBinary target distribution:")
print(df['readmitted_binary'].value_counts())
imbalance = df['readmitted_binary'].value_counts()[0] / df['readmitted_binary'].value_counts()[1]
print(f"Imbalance ratio: {imbalance:.1f}:1")

df = df.drop(columns=['readmitted'])

# %% [markdown]
# ## Step 5 -- Fill Remaining Missing Values
# 
# - **Numeric columns** -> fill with **median** (robust to outliers)
# - **Categorical columns** -> fill with **mode** (most frequent value)

# %%
# Normalize dtypes for consistent handling
for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
        df[col] = df[col].astype('object')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  Filled {col} ({df[col].isnull().sum()} remaining) with mode: '{mode_val}'")

print(f"\nRemaining NaN count: {df.isnull().sum().sum()}")

# %% [markdown]
# ## Step 6 -- Map ICD Diagnosis Codes to Clinical Categories
# 
# The `diag_1`, `diag_2`, `diag_3` columns contain ICD-9 diagnosis codes.
# Rather than one-hot encoding thousands of unique codes, we map them to
# clinically meaningful disease categories. This reduces dimensionality
# while preserving medical signal.

# %%
def map_icd_to_category(code):
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

for diag_col in ['diag_1', 'diag_2', 'diag_3']:
    col_name = f'{diag_col}_category'
    df[col_name] = df[diag_col].apply(map_icd_to_category)
    print(f"\n{col_name} distribution (top 5):")
    print(df[col_name].value_counts().head(5))

df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
print(f"\nShape after ICD mapping: {df.shape}")

# %% [markdown]
# ## Step 7 -- Encode Categorical Features
# 
# **Encoding strategy:**
# - **Ordinal features** (natural order): LabelEncoder -- `age`, `A1Cresult`, `max_glu_serum`
# - **Nominal features** (no order): OneHotEncoding via `pd.get_dummies`

# %%
label_encoders = {}

# Age has a natural bracket order
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
if 'age' in df.columns:
    le_age = LabelEncoder()
    le_age.fit(age_order)
    df['age'] = df['age'].apply(lambda x: x if x in age_order else age_order[-1])
    df['age'] = le_age.transform(df['age'])
    label_encoders['age'] = le_age

if 'A1Cresult' in df.columns:
    le_a1c = LabelEncoder()
    a1c_values = sorted(df['A1Cresult'].unique().tolist())
    le_a1c.fit(a1c_values)
    df['A1Cresult'] = le_a1c.transform(df['A1Cresult'])
    label_encoders['A1Cresult'] = le_a1c

if 'max_glu_serum' in df.columns:
    le_glu = LabelEncoder()
    glu_values = sorted(df['max_glu_serum'].unique().tolist())
    le_glu.fit(glu_values)
    df['max_glu_serum'] = le_glu.transform(df['max_glu_serum'])
    label_encoders['max_glu_serum'] = le_glu

print(f"Label encoders created: {list(label_encoders.keys())}")

# %%
# One-hot encoding for remaining categorical columns
remaining_cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"One-hot encoding {len(remaining_cat_cols)} columns")

df = pd.get_dummies(df, columns=remaining_cat_cols, drop_first=True, dtype=int)
print(f"Shape after encoding: {df.shape}")

# %% [markdown]
# ## Step 8 -- Scale Continuous Numerical Features
# 
# StandardScaler normalizes features to zero mean and unit variance.
# Applied only to continuous clinical measurements.

# %%
continuous_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

existing_continuous = [c for c in continuous_cols if c in df.columns]
scaler = StandardScaler()
df[existing_continuous] = scaler.fit_transform(df[existing_continuous])

print(f"Scaled {len(existing_continuous)} features")
print(df[existing_continuous].describe().round(3).loc[['mean', 'std', 'min', 'max']])

# %% [markdown]
# ## Step 9 -- Save Processed Data and Artifacts

# %%
# Save cleaned dataset
output_path = DATA_PROCESSED / 'diabetic_clean.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved: {output_path}")
print(f"Shape: {df.shape} | Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save scaler and encoders
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
joblib.dump(label_encoders, MODELS_DIR / 'encoders.pkl')

# Save feature names for prediction pipeline
feature_names = [c for c in df.columns if c != 'readmitted_binary']
joblib.dump(feature_names, MODELS_DIR / 'feature_names.pkl')

print(f"\nArtifacts saved to: {MODELS_DIR}")

# %%
print("\n" + "=" * 60)
print("PHASE 1 -- DATA PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\nDataset: (101766, 50) -> {df.shape}")
print(f"Target: readmitted_binary")
print(f"Class 0: {(df['readmitted_binary']==0).sum():,} | Class 1: {(df['readmitted_binary']==1).sum():,}")
print(f"Imbalance: {imbalance:.1f}:1")
print(f"\nNext: Phase 2 -- Exploratory Data Analysis")
