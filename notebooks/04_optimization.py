# %% [markdown]
# # Phase 4 -- Model Optimization
# ## Hospital Readmission Risk Scorer
#
# Hyperparameter tuning for XGBoost with RandomizedSearchCV.
# Optimizes for recall using 5-fold stratified cross-validation.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score, f1_score, accuracy_score, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
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
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
calculated_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Data ready. scale_pos_weight: {calculated_ratio:.2f}")

# %% [markdown]
# ## RandomizedSearchCV

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'scale_pos_weight': [calculated_ratio]
}
search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    param_grid, n_iter=30, scoring='recall',
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    random_state=42, n_jobs=-1, verbose=1
)
search.fit(X_train_sm, y_train_sm)
print(f"Best CV Recall: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# %% [markdown]
# ## Final Model Evaluation

# %%
best_model = xgb.XGBClassifier(**search.best_params_, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1)
best_model.fit(X_train_sm, y_train_sm)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))

# %% [markdown]
# ## Feature Importance (Top 20)

# %%
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(feat_imp)), feat_imp['importance'].values)
ax.set_yticks(range(len(feat_imp)))
ax.set_yticklabels(feat_imp['feature'].values)
ax.set_title('Top 20 Feature Importance', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ROC and Precision-Recall Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, color='#e74c3c', linewidth=2.5, label=f'AUC={roc_auc_score(y_test, y_proba):.3f}')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title('ROC Curve', fontweight='bold')
ax1.legend()
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ax2.plot(rec, prec, color='#2ecc71', linewidth=2.5, label=f'AP={average_precision_score(y_test, y_proba):.3f}')
ax2.set_title('Precision-Recall Curve', fontweight='bold')
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save Final Model

# %%
joblib.dump(best_model, MODELS_DIR / 'xgboost_readmission_model.pkl')
joblib.dump(search.best_params_, MODELS_DIR / 'best_params.pkl')
print("Final model and params saved!")
print("PHASE 4 COMPLETE -- Ready for Phase 5 (Prediction Pipeline)")
