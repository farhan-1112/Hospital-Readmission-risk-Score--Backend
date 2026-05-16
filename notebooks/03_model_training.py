# %% [markdown]
# # Phase 3 -- Model Training
# ## Hospital Readmission Risk Scorer
#
# Trains 4 models, handles imbalance with SMOTE, evaluates with recall as primary metric.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

# %%
df = pd.read_csv(DATA_PROCESSED / 'diabetic_clean.csv')
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## SMOTE -- Balance Training Data Only

# %%
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"After SMOTE: {X_train_sm.shape}, Balanced: {pd.Series(y_train_sm).value_counts().to_dict()}")

# %% [markdown]
# ## Train All Models

# %%
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    print(f"\n{name}: Recall={results[name]['Recall']:.4f}, F1={results[name]['F1']:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))

# %% [markdown]
# ## Model Comparison Table

# %%
comparison = pd.DataFrame(results).T.sort_values('Recall', ascending=False)
print(comparison.round(4))
print(f"\nBest by Recall: {comparison['Recall'].idxmax()}")

# %% [markdown]
# ## Confusion Matrices and ROC Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for i, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes.flatten()[i])
    axes.flatten()[i].set_title(f'{name}\nRecall={results[name]["Recall"]:.3f}', fontweight='bold')
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={results[name]["ROC-AUC"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.show()
