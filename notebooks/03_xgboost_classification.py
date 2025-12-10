# %% [markdown]
# # XGBoost Classification for ICU Length of Stay
# 
# This script trains an XGBoost model to predict:
# - Binary LOS: Short (<4 days) vs Long (≥4 days)
# 
# Includes:
# - Baseline model
# - Hyperparameter tuning
# - Feature importance analysis
# - Model evaluation (AUROC, precision, recall, etc.)

# %% Imports
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, 
    average_precision_score, f1_score, accuracy_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# %% Configuration
OUTPUT_DIR = Path("../outputs")
RANDOM_STATE = 42

# %% Load preprocessed data
print("Loading preprocessed data...")
with open(OUTPUT_DIR / 'data_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

X_train = splits['X_train']
X_val = splits['X_val']
X_test = splits['X_test']
y_train = splits['y_train']
y_val = splits['y_val']
y_test = splits['y_test']
feature_columns = splits['feature_columns']

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {len(feature_columns)}")
print(f"\nClass distribution (train):")
print(f"  Short stay (0): {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)")
print(f"  Long stay (1): {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)")

# %% Calculate class weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
print(f"Using scale_pos_weight={scale_pos_weight:.2f} to handle imbalance")

# %% Baseline Model
print("\n" + "=" * 60)
print("BASELINE XGBOOST MODEL")
print("=" * 60)

baseline_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='auc'
)

# Train
baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predict
y_train_pred = baseline_model.predict_proba(X_train)[:, 1]
y_val_pred = baseline_model.predict_proba(X_val)[:, 1]

# Evaluate
train_auc = roc_auc_score(y_train, y_train_pred)
val_auc = roc_auc_score(y_val, y_val_pred)

print(f"\nBaseline Results:")
print(f"  Train AUROC: {train_auc:.4f}")
print(f"  Val AUROC:   {val_auc:.4f}")

# %% Hyperparameter Tuning
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# For faster tuning, use a smaller grid first
quick_param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
}

print("Running GridSearchCV (this may take a few minutes)...")

base_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='auc'
)

grid_search = GridSearchCV(
    base_model,
    quick_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV AUROC: {grid_search.best_score_:.4f}")

# %% Train final model with best parameters
print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

best_params = grid_search.best_params_
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = RANDOM_STATE
best_params['n_jobs'] = -1
best_params['eval_metric'] = 'auc'

final_model = xgb.XGBClassifier(**best_params)

# Train with early stopping
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# %% Evaluate on all sets
def evaluate_model(model, X, y, set_name=""):
    """Comprehensive model evaluation."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    # Metrics
    auroc = roc_auc_score(y, y_pred_proba)
    auprc = average_precision_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{set_name} Results:")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"  AUPRC:    {auprc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return {
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f1': f1
    }

train_results = evaluate_model(final_model, X_train, y_train, "Training")
val_results = evaluate_model(final_model, X_val, y_val, "Validation")
test_results = evaluate_model(final_model, X_test, y_test, "Test")

# %% Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Test Set)")
print("=" * 60)
print(classification_report(y_test, test_results['y_pred'], 
                           target_names=['Short Stay (<4d)', 'Long Stay (≥4d)']))

# %% Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, test_results['y_pred'])
print(cm)

# %% Feature Importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Features:")
print(importance_df.head(20).to_string(index=False))

# Save full importance list
importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'feature_importance.csv'}")

# %% Visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC Curve
ax = axes[0, 0]
for results, name, color in [(train_results, 'Train', 'blue'), 
                              (val_results, 'Val', 'green'),
                              (test_results, 'Test', 'red')]:
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
    ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={results["auroc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax = axes[0, 1]
for results, name, color in [(train_results, 'Train', 'blue'), 
                              (val_results, 'Val', 'green'),
                              (test_results, 'Test', 'red')]:
    precision, recall, _ = precision_recall_curve(results['y_true'], results['y_pred_proba'])
    ax.plot(recall, precision, color=color, label=f'{name} (AUPRC={results["auprc"]:.3f})')

ax.axhline(y=y_train.mean(), color='k', linestyle='--', label=f'Baseline ({y_train.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 3. Confusion Matrix Heatmap
ax = axes[1, 0]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
            xticklabels=['Short', 'Long'], yticklabels=['Short', 'Long'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Normalized)')

# Add raw counts
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.7, f'(n={cm[i, j]})', 
                ha='center', va='center', fontsize=9, color='gray')

# 4. Top 20 Feature Importance
ax = axes[1, 1]
top_20 = importance_df.head(20)
colors = ['#e74c3c' if 'invasive' in f or 'vasopressor' in f or 'sofa' in f 
          else '#3498db' for f in top_20['feature']]
ax.barh(range(len(top_20)), top_20['importance'].values, color=colors)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Top 20 Features')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_xgboost_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR / '04_xgboost_results.png'}")

# %% Predicted Probability Distribution
fig, ax = plt.subplots(figsize=(10, 6))

# Plot distributions for each class
for label, color, name in [(0, 'blue', 'Short Stay'), (1, 'red', 'Long Stay')]:
    mask = test_results['y_true'] == label
    ax.hist(test_results['y_pred_proba'][mask], bins=30, alpha=0.5, 
            color=color, label=name, density=True)

ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
ax.set_xlabel('Predicted Probability of Long Stay')
ax.set_ylabel('Density')
ax.set_title('Distribution of Predicted Probabilities (Test Set)')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_probability_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR / '05_probability_distribution.png'}")

# %% Save model and results
print("\n" + "=" * 60)
print("SAVING MODEL AND RESULTS")
print("=" * 60)

# Save model
final_model.save_model(OUTPUT_DIR / 'xgboost_model.json')
print(f"Saved: {OUTPUT_DIR / 'xgboost_model.json'}")

# Save results summary
results_summary = {
    'best_params': best_params,
    'train_auroc': train_results['auroc'],
    'val_auroc': val_results['auroc'],
    'test_auroc': test_results['auroc'],
    'test_auprc': test_results['auprc'],
    'test_accuracy': test_results['accuracy'],
    'test_f1': test_results['f1'],
    'confusion_matrix': cm.tolist(),
    'feature_importance': importance_df.to_dict('records')
}

with open(OUTPUT_DIR / 'xgboost_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print(f"Saved: {OUTPUT_DIR / 'xgboost_results.pkl'}")

# %% Summary for Report
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)

print(f"""
## XGBoost Classification Results

### Model Configuration
- Target: ICU LOS ≥4 days (binary classification)
- Features: {len(feature_columns)} (first 24h of ICU stay)
- Training samples: {len(X_train)}
- Class balance: {y_train.mean()*100:.1f}% positive (long stay)

### Best Hyperparameters
{best_params}

### Performance Metrics (Test Set)
| Metric | Value |
|--------|-------|
| AUROC | {test_results['auroc']:.4f} |
| AUPRC | {test_results['auprc']:.4f} |
| Accuracy | {test_results['accuracy']:.4f} |
| F1 Score | {test_results['f1']:.4f} |

### Comparison to Literature
- Hempel et al. (2023): AUROC = 0.80 on MIMIC-IV
- Our model: AUROC = {test_results['auroc']:.2f}

### Top 5 Predictive Features
""")

for i, row in importance_df.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")

# %% Next Steps
print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
1. Run on full dataset (50k patients) for final results
2. Build Cox survival model (04_survival_analysis.py)
3. Perform fairness analysis across demographic groups
4. Create final presentation figures

To load this model later:
```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model('../outputs/xgboost_model.json')
```
""")

# %%
print("\n✅ XGBoost classification complete!")
