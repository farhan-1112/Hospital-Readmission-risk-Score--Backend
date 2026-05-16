"""
Phase 4 -- Model Optimization (Corrected)
Hospital Readmission Risk Scorer

Key fix: Train on SMOTE data but use a custom scoring that evaluates on
non-synthetic distributions. We use f2-score (recall-weighted F-beta) as
the optimization target instead of pure recall, which prevents the model
from degenerating into "predict everything positive".
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, recall_score,
                             f1_score, accuracy_score, precision_recall_curve,
                             average_precision_score, make_scorer, fbeta_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'figures'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")

# ============================================================
# Load and Split Data
# ============================================================
print("\n[Step 1] Loading processed data...")
df = pd.read_csv(DATA_PROCESSED / 'diabetic_clean.csv')

X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")

# Calculate class weight ratio (for XGBoost, NOT combining with SMOTE)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
calculated_ratio = neg_count / pos_count
print(f"scale_pos_weight: {calculated_ratio:.2f}")

# ============================================================
# Hyperparameter Search
# Using original (non-SMOTE) training data with scale_pos_weight
# This avoids the double-counting problem of SMOTE + scale_pos_weight
# ============================================================
print("\n[Step 2] Running RandomizedSearchCV for XGBoost...")
print("  Strategy: Use scale_pos_weight for class imbalance (no SMOTE during CV)")
print("  Scoring: F2-score (recall-weighted F-beta, beta=2)")
print("  This prioritizes recall while preventing degenerate all-positive predictions")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'scale_pos_weight': [calculated_ratio, calculated_ratio * 0.5, calculated_ratio * 1.5]
}

xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    n_jobs=-1
)

# F2 score: beta=2 weights recall 2x more than precision
# This is a healthcare-appropriate metric -- prioritizes recall while
# still penalizing the model for predicting everything as positive
f2_scorer = make_scorer(fbeta_score, beta=2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=40,
    scoring=f2_scorer,
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit on ORIGINAL training data (not SMOTE) -- scale_pos_weight handles imbalance
search.fit(X_train, y_train)

print(f"\n  Best F2 score (CV): {search.best_score_:.4f}")
print(f"  Best parameters:")
for param, value in search.best_params_.items():
    print(f"    {param}: {value}")

# ============================================================
# Retrain with Best Params
# ============================================================
print("\n[Step 3] Retraining with best parameters on full training set...")

best_model = xgb.XGBClassifier(
    **search.best_params_,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    n_jobs=-1
)

# Train on original data with class weighting
best_model.fit(X_train, y_train)
print("  Model trained on original training data with class-weighted XGBoost")

# ============================================================
# Evaluate on Test Set
# ============================================================
print("\n[Step 4] Evaluating final model on test set...")

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)
roc_auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)

print(f"\n  FINAL MODEL METRICS (Test Set):")
print(f"  {'='*40}")
print(f"  Recall:   {recall:.4f}  <-- PRIMARY METRIC")
print(f"  F1:       {f1:.4f}")
print(f"  F2:       {f2:.4f}  <-- Optimization target")
print(f"  ROC-AUC:  {roc_auc:.4f}")
print(f"  Accuracy: {acc:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))

cm = confusion_matrix(y_test, y_pred)
print(f"  Confusion Matrix:")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# ============================================================
# Also try SMOTE + XGBoost (without scale_pos_weight) for comparison
# ============================================================
print("\n[Step 5] Comparing with SMOTE-based approach...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Use best params but without scale_pos_weight (SMOTE handles imbalance)
smote_params = {k: v for k, v in search.best_params_.items() if k != 'scale_pos_weight'}
smote_model = xgb.XGBClassifier(
    **smote_params,
    scale_pos_weight=1.0,  # Neutral since SMOTE balanced
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    n_jobs=-1
)
smote_model.fit(X_train_sm, y_train_sm)

y_pred_sm = smote_model.predict(X_test)
y_proba_sm = smote_model.predict_proba(X_test)[:, 1]

recall_sm = recall_score(y_test, y_pred_sm)
f1_sm = f1_score(y_test, y_pred_sm)
roc_auc_sm = roc_auc_score(y_test, y_proba_sm)
acc_sm = accuracy_score(y_test, y_pred_sm)

print(f"  SMOTE Model: Recall={recall_sm:.4f}, F1={f1_sm:.4f}, AUC={roc_auc_sm:.4f}, Acc={acc_sm:.4f}")
print(f"  Weighted Model: Recall={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}, Acc={acc:.4f}")

# Pick the model with better F2 score (recall-weighted balance)
f2_sm = fbeta_score(y_test, y_pred_sm, beta=2)
print(f"\n  F2 comparison: Weighted={f2:.4f} vs SMOTE={f2_sm:.4f}")

if f2_sm > f2:
    print("  --> SMOTE model has better F2 score. Using SMOTE model as final.")
    final_model = smote_model
    final_recall = recall_sm
    final_f1 = f1_sm
    final_auc = roc_auc_sm
    final_acc = acc_sm
    final_f2 = f2_sm
    y_pred_final = y_pred_sm
    y_proba_final = y_proba_sm
else:
    print("  --> Weighted model has better F2 score. Using weighted model as final.")
    final_model = best_model
    final_recall = recall
    final_f1 = f1
    final_auc = roc_auc
    final_acc = acc
    final_f2 = f2
    y_pred_final = y_pred
    y_proba_final = y_proba

# ============================================================
# Plot 1: Feature Importance (Top 20)
# ============================================================
print("\n[Step 6] Generating plots...")

feature_names = X.columns.tolist()
importance = final_model.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(feat_imp)), feat_imp['importance'].values,
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp))),
        edgecolor='white')
ax.set_yticks(range(len(feat_imp)))
ax.set_yticklabels([name.replace('_', ' ').title()[:35] for name in feat_imp['feature']], fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 20 Most Important Features (Optimized XGBoost)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'feature_importance_top20.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance_top20.png")

# ============================================================
# Plot 2: ROC Curve
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_proba_final)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2.5,
        label=f'Optimized XGBoost (AUC = {roc_auc_score(y_test, y_proba_final):.3f})')
ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve - Optimized XGBoost', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)

optimal_idx = np.argmax(tpr - fpr)
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ko', markersize=10)
ax.annotate(f'Optimal: thresh={thresholds[optimal_idx]:.2f}\n'
            f'TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f}',
            xy=(fpr[optimal_idx], tpr[optimal_idx]),
            xytext=(fpr[optimal_idx]+0.15, tpr[optimal_idx]-0.15),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'roc_curve_optimized.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: roc_curve_optimized.png")

# ============================================================
# Plot 3: Precision-Recall Curve
# ============================================================
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_final)
avg_precision = average_precision_score(y_test, y_proba_final)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(recall_vals, precision_vals, color='#2ecc71', linewidth=2.5,
        label=f'XGBoost (AP = {avg_precision:.3f})')
ax.fill_between(recall_vals, precision_vals, alpha=0.15, color='#2ecc71')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve - Optimized XGBoost', fontsize=14, fontweight='bold')
baseline = y_test.mean()
ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: precision_recall_curve.png")

# ============================================================
# Save Final Model
# ============================================================
print("\n[Step 7] Saving final model and artifacts...")

model_path = MODELS_DIR / 'xgboost_readmission_model.pkl'
joblib.dump(final_model, model_path)
print(f"  Final model saved: {model_path}")

params_path = MODELS_DIR / 'best_params.pkl'
joblib.dump(search.best_params_, params_path)
print(f"  Best params saved: {params_path}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4 -- MODEL OPTIMIZATION COMPLETE")
print("=" * 60)
print(f"""
FINAL OPTIMIZED MODEL:
  Model:     XGBoost Classifier
  Recall:    {final_recall:.4f}
  F1-Score:  {final_f1:.4f}
  F2-Score:  {final_f2:.4f}
  ROC-AUC:   {final_auc:.4f}
  Accuracy:  {final_acc:.4f}

BEST HYPERPARAMETERS:""")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

print(f"""
FILES SAVED:
  - models/xgboost_readmission_model.pkl  (final model)
  - models/best_params.pkl                (hyperparameters)
  - reports/figures/feature_importance_top20.png
  - reports/figures/roc_curve_optimized.png
  - reports/figures/precision_recall_curve.png

NEXT: Phase 5 -- Prediction Pipeline
  Test standalone src/predict.py for FastAPI deployment.
""")
