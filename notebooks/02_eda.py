# %% [markdown]
# # Phase 2 -- Exploratory Data Analysis
# ## Hospital Readmission Risk Scorer
#
# Analyzes feature distributions, correlations, and readmission patterns.
# Generates visualizations for clinical insights.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'figures'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# %%
df_clean = pd.read_csv(DATA_PROCESSED / 'diabetic_clean.csv')
df_raw = pd.read_csv(DATA_RAW / 'diabetic_data.csv')
df_raw = df_raw.replace('?', np.nan)
df_raw['readmitted_binary'] = (df_raw['readmitted'] == '<30').astype(int)
print(f"Clean shape: {df_clean.shape}, Raw shape: {df_raw.shape}")

# %% [markdown]
# ## 1. Class Distribution
# The target variable shows severe imbalance -- only ~11% of patients
# were readmitted within 30 days. This drives our entire modeling strategy.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df_clean['readmitted_binary'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = axes[0].bar(['Not Readmitted\nClass 0', 'Readmitted\nClass 1'],
                    counts.values, color=colors, edgecolor='white')
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
for bar, count in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                f'{count:,}', ha='center', fontsize=12, fontweight='bold')
axes[1].pie(counts.values, labels=['Not Readmitted', 'Readmitted'],
            colors=colors, autopct='%1.1f%%', explode=(0, 0.08), shadow=True)
plt.tight_layout()
plt.show()
print(f"Imbalance ratio: {counts[0]/counts[1]:.1f}:1")

# %% [markdown]
# ## 2. Feature Correlations with Target

# %%
correlations = df_clean.corr(numeric_only=True)['readmitted_binary'].drop('readmitted_binary')
top_10_features = correlations.abs().nlargest(10).index.tolist()
top_10_features.append('readmitted_binary')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_clean[top_10_features].corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Top 10 Features - Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Feature Distributions

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, (col, title) in enumerate([
    ('time_in_hospital', 'Time in Hospital'), ('num_medications', 'Num Medications'),
    ('number_inpatient', 'Inpatient Visits'), ('age', 'Age')]):
    ax = axes.flatten()[i]
    if col == 'age':
        df_raw[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='#9b59b6')
    else:
        ax.hist(pd.to_numeric(df_raw[col], errors='coerce').dropna(), bins=30, color='#3498db')
    ax.set_title(f'Distribution of {title}', fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Readmission Rates by Category

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, col in enumerate(['age', 'race', 'gender', 'insulin', 'diabetesMed', 'A1Cresult']):
    ax = axes.flatten()[i]
    rates = df_raw.groupby(col)['readmitted_binary'].mean().sort_values(ascending=False)
    rates.plot(kind='bar', ax=ax, color=plt.cm.RdYlGn_r(rates.values / max(rates.max(), 0.01)))
    ax.set_title(f'Readmission Rate by {col}', fontweight='bold')
    ax.axhline(y=df_raw['readmitted_binary'].mean(), color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Outlier Detection

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
for i, col in enumerate(['time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']):
    data = pd.to_numeric(df_raw[col], errors='coerce').dropna()
    axes.flatten()[i].boxplot(data, patch_artist=True, boxprops=dict(facecolor='#3498db', alpha=0.6))
    axes.flatten()[i].set_title(col.replace('_', ' ').title(), fontweight='bold')
plt.suptitle('Outlier Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary of Findings
#
# **Class Imbalance**: 8:1 ratio demands SMOTE + recall-focused models
#
# **Correlations**: Weak linear correlations typical for healthcare data.
# `number_inpatient` (0.165) is the strongest predictor.
# Tree-based models will capture non-linear interactions.
#
# **Demographics**: Readmission increases with age (peaks 70-80).
# Insulin users and patients with A1C > 8 show higher risk.
#
# **Clinical**: Prior inpatient visits strongly predict readmission.
# Emergency visit history signals patient instability.
#
# **Outliers**: Right-skewed utilization features have meaningful outliers
# (frequent hospital users) -- keep them, they carry clinical signal.
