"""
Phase 3 -- Model Training
Hospital Readmission Risk Scorer

Trains 4 classification models, handles class imbalance with SMOTE,
and evaluates using recall as the primary metric.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, recall_score,
                             f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
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
# Load Processed Data
# ============================================================
print("\n[Step 1] Loading processed data...")
df = pd.read_csv(DATA_PROCESSED / 'diabetic_clean.csv')
print(f"Dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")

# ============================================================
# Train-Test Split (80/20, stratified)
# ============================================================
print("\n[Step 2] Splitting data (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")
print(f"Train target distribution:\n{y_train.value_counts()}")

# ============================================================
# Apply SMOTE (training data ONLY -- never on test)
# ============================================================
print("\n[Step 3] Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {X_train.shape[0]:,} samples")
print(f"After SMOTE:  {X_train_sm.shape[0]:,} samples")
print(f"SMOTE target distribution:\n{pd.Series(y_train_sm).value_counts()}")
print("  --> Classes are now balanced for training")

# Calculate class weight ratio for XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nscale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

# ============================================================
# Define Models
# ============================================================
print("\n[Step 4] Defining models...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
}

# ============================================================
# Train and Evaluate All Models
# ============================================================
print("\n[Step 5] Training and evaluating models...")
print("=" * 70)

results = {}
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(14, 12))
axes_cm = axes_cm.flatten()

fig_roc, ax_roc = plt.subplots(figsize=(10, 8))

for i, (name, model) in enumerate(models.items()):
    print(f"\n--- {name} ---")
    
    # Train on SMOTE-balanced data
    # (except for models with built-in class_weight handling,
    #  we still use SMOTE data for consistency in comparison)
    model.fit(X_train_sm, y_train_sm)
    
    # Predict on ORIGINAL test set (never SMOTE'd)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'Accuracy': acc,
        'model': model
    }
    
    # Print classification report
    print(f"  Recall:   {recall:.4f}  <-- PRIMARY METRIC")
    print(f"  F1:       {f1:.4f}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i],
                xticklabels=['Not Readmit', 'Readmit'],
                yticklabels=['Not Readmit', 'Readmit'])
    axes_cm[i].set_title(f'{name}\nRecall={recall:.3f} | F1={f1:.3f}',
                         fontsize=12, fontweight='bold')
    axes_cm[i].set_ylabel('Actual')
    axes_cm[i].set_xlabel('Predicted')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

# Save confusion matrices
fig_cm.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig_cm.savefig(REPORTS_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close(fig_cm)

# Save ROC curves
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.500)')
ax_roc.set_xlabel('False Positive Rate', fontsize=12)
ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax_roc.legend(fontsize=11, loc='lower right')
ax_roc.grid(True, alpha=0.3)
fig_roc.tight_layout()
fig_roc.savefig(REPORTS_DIR / 'roc_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig_roc)

# ============================================================
# Comparison Table
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)

comparison_df = pd.DataFrame({
    name: {k: v for k, v in metrics.items() if k != 'model'}
    for name, metrics in results.items()
}).T

comparison_df = comparison_df.sort_values('Recall', ascending=False)
print(comparison_df.round(4).to_string())

# Save comparison as CSV
comparison_df.to_csv(PROJECT_ROOT / 'reports' / 'model_comparison.csv')

# Highlight best model by recall
best_model_name = comparison_df['Recall'].idxmax()
print(f"\n--> Best model by Recall: {best_model_name}")
print(f"    Recall: {comparison_df.loc[best_model_name, 'Recall']:.4f}")
print(f"    F1:     {comparison_df.loc[best_model_name, 'F1']:.4f}")
print(f"    AUC:    {comparison_df.loc[best_model_name, 'ROC-AUC']:.4f}")

# Save the base XGBoost model (before optimization) for reference
xgb_model = results['XGBoost']['model']
joblib.dump(xgb_model, MODELS_DIR / 'xgboost_base_model.pkl')
print(f"\nBase XGBoost model saved to: {MODELS_DIR / 'xgboost_base_model.pkl'}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 -- MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"""
RESULTS SUMMARY:
  - All 4 models trained on SMOTE-balanced training data
  - Evaluated on original (non-SMOTE) test set
  - Recall is the primary metric (missing high-risk patients = dangerous)
  
PLOTS SAVED:
  - reports/figures/confusion_matrices.png
  - reports/figures/roc_curves_comparison.png
  - reports/model_comparison.csv

NEXT: Phase 4 -- Model Optimization
  - Hyperparameter tuning for XGBoost using RandomizedSearchCV
  - Optimize for recall with 5-fold stratified CV
""")
