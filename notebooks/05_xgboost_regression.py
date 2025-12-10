# %% [markdown]
# # XGBoost Regression for Continuous LOS Prediction
# 
# This script predicts the actual ICU length of stay in days (regression).
# Metrics: MAE, MSE, RMSE, R²

# %% Imports
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
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
meta_train = splits['meta_train']
meta_val = splits['meta_val']
meta_test = splits['meta_test']
feature_columns = splits['feature_columns']

# Target: continuous LOS in days
y_train_reg = meta_train['los_days'].values
y_val_reg = meta_val['los_days'].values
y_test_reg = meta_test['los_days'].values

print(f"Training set: {X_train.shape}")
print(f"Target (LOS days): mean={y_train_reg.mean():.2f}, median={np.median(y_train_reg):.2f}, std={y_train_reg.std():.2f}")

# %% Baseline Model
print("\n" + "=" * 60)
print("BASELINE XGBOOST REGRESSOR")
print("=" * 60)

baseline_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

baseline_reg.fit(X_train, y_train_reg, 
                 eval_set=[(X_val, y_val_reg)],
                 verbose=False)

# Predictions
y_train_pred = baseline_reg.predict(X_train)
y_val_pred = baseline_reg.predict(X_val)

# Metrics
print(f"\nBaseline Results:")
print(f"  Train MAE:  {mean_absolute_error(y_train_reg, y_train_pred):.3f} days")
print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train_reg, y_train_pred)):.3f} days")
print(f"  Val MAE:    {mean_absolute_error(y_val_reg, y_val_pred):.3f} days")
print(f"  Val RMSE:   {np.sqrt(mean_squared_error(y_val_reg, y_val_pred)):.3f} days")

# %% Hyperparameter Tuning
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
}

print("Running GridSearchCV...")

base_reg = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)

grid_search = GridSearchCV(
    base_reg,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train_reg)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_:.3f} days")

# %% Final Model
print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

best_params = grid_search.best_params_
best_params['random_state'] = RANDOM_STATE
best_params['n_jobs'] = -1

final_reg = xgb.XGBRegressor(**best_params)
final_reg.fit(X_train, y_train_reg,
              eval_set=[(X_val, y_val_reg)],
              verbose=False)

# %% Evaluation
def evaluate_regression(y_true, y_pred, set_name=""):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{set_name} Results:")
    print(f"  MAE:  {mae:.3f} days")
    print(f"  MSE:  {mse:.3f}")
    print(f"  RMSE: {rmse:.3f} days")
    print(f"  R²:   {r2:.4f}")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

y_train_pred = final_reg.predict(X_train)
y_val_pred = final_reg.predict(X_val)
y_test_pred = final_reg.predict(X_test)

train_metrics = evaluate_regression(y_train_reg, y_train_pred, "Training")
val_metrics = evaluate_regression(y_val_reg, y_val_pred, "Validation")
test_metrics = evaluate_regression(y_test_reg, y_test_pred, "Test")

# %% Error Analysis by LOS Category
print("\n" + "=" * 60)
print("ERROR ANALYSIS BY LOS CATEGORY")
print("=" * 60)

# Categorize by actual LOS
def categorize_los(los):
    if los < 2:
        return '<2 days'
    elif los < 4:
        return '2-4 days'
    elif los < 7:
        return '4-7 days'
    elif los < 14:
        return '7-14 days'
    else:
        return '>14 days'

test_df = pd.DataFrame({
    'actual': y_test_reg,
    'predicted': y_test_pred,
    'error': y_test_pred - y_test_reg,
    'abs_error': np.abs(y_test_pred - y_test_reg),
    'category': [categorize_los(x) for x in y_test_reg]
})

print("\nMAE by LOS Category:")
category_order = ['<2 days', '2-4 days', '4-7 days', '7-14 days', '>14 days']
for cat in category_order:
    subset = test_df[test_df['category'] == cat]
    if len(subset) > 0:
        print(f"  {cat}: MAE={subset['abs_error'].mean():.2f} days (n={len(subset)})")

# %% Visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test_reg, y_test_pred, alpha=0.3, s=20)
ax.plot([0, 30], [0, 30], 'r--', label='Perfect prediction')
ax.set_xlabel('Actual LOS (days)')
ax.set_ylabel('Predicted LOS (days)')
ax.set_title(f'Actual vs Predicted LOS (Test Set)\nMAE={test_metrics["mae"]:.2f}, R²={test_metrics["r2"]:.3f}')
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.legend()

# 2. Residuals Distribution
ax = axes[0, 1]
residuals = y_test_pred - y_test_reg
ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--')
ax.set_xlabel('Prediction Error (days)')
ax.set_ylabel('Count')
ax.set_title(f'Residuals Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}')

# 3. Error by LOS Category
ax = axes[1, 0]
category_mae = test_df.groupby('category')['abs_error'].mean().reindex(category_order)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(category_order)))
bars = ax.bar(range(len(category_order)), category_mae.values, color=colors)
ax.set_xticks(range(len(category_order)))
ax.set_xticklabels(category_order, rotation=45, ha='right')
ax.set_xlabel('Actual LOS Category')
ax.set_ylabel('Mean Absolute Error (days)')
ax.set_title('Prediction Error by LOS Category')

# Add count labels
category_counts = test_df.groupby('category').size().reindex(category_order)
for i, (bar, count) in enumerate(zip(bars, category_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'n={count}', ha='center', va='bottom', fontsize=9)

# 4. Feature Importance (Top 15)
ax = axes[1, 1]
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': final_reg.feature_importances_
}).sort_values('importance', ascending=False).head(15)

ax.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Top 15 Features (Regression)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_regression_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR / '09_regression_results.png'}")

# %% Save Results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save model
with open(OUTPUT_DIR / 'xgboost_regressor.pkl', 'wb') as f:
    pickle.dump(final_reg, f)
print(f"Saved: {OUTPUT_DIR / 'xgboost_regressor.pkl'}")

# Save results
regression_results = {
    'best_params': best_params,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'feature_importance': importance_df.to_dict('records')
}

with open(OUTPUT_DIR / 'regression_results.pkl', 'wb') as f:
    pickle.dump(regression_results, f)
print(f"Saved: {OUTPUT_DIR / 'regression_results.pkl'}")

# %% Summary for Report
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)

print(f"""
## XGBoost Regression Results

### Model Configuration
- Target: Continuous ICU LOS (days)
- Features: {len(feature_columns)} (first 24h of ICU stay)
- Training samples: {len(X_train)}

### Best Hyperparameters
{best_params}

### Performance Metrics

| Set | MAE (days) | RMSE (days) | R² |
|-----|------------|-------------|-----|
| Train | {train_metrics['mae']:.3f} | {train_metrics['rmse']:.3f} | {train_metrics['r2']:.4f} |
| Val | {val_metrics['mae']:.3f} | {val_metrics['rmse']:.3f} | {val_metrics['r2']:.4f} |
| Test | {test_metrics['mae']:.3f} | {test_metrics['rmse']:.3f} | {test_metrics['r2']:.4f} |

### Comparison to Literature
- Hempel et al. (2023): MAE ~1.1 days (stepwise approach on short stays only)
- Our model: MAE = {test_metrics['mae']:.2f} days (all stays)

### Error by LOS Category
""")

for cat in category_order:
    subset = test_df[test_df['category'] == cat]
    if len(subset) > 0:
        print(f"- {cat}: MAE = {subset['abs_error'].mean():.2f} days (n={len(subset)})")

print(f"""
### Key Observation
Prediction error increases with actual LOS — the model struggles more
with long-stay patients, which is consistent with literature findings.
""")

# %% Complete Model Comparison
print("\n" + "=" * 60)
print("COMPLETE MODEL COMPARISON")
print("=" * 60)

# Load other results
try:
    with open(OUTPUT_DIR / 'xgboost_results.pkl', 'rb') as f:
        clf_results = pickle.load(f)
    with open(OUTPUT_DIR / 'cox_results.pkl', 'rb') as f:
        cox_results = pickle.load(f)
    
    print("""
| Model | Task | Metric | Test Value |
|-------|------|--------|------------|
| XGBoost Classifier | Binary LOS (≥4d) | AUROC | {:.4f} |
| XGBoost Classifier | Binary LOS (≥4d) | AUPRC | {:.4f} |
| XGBoost Regressor | Continuous LOS | MAE | {:.3f} days |
| XGBoost Regressor | Continuous LOS | RMSE | {:.3f} days |
| XGBoost Regressor | Continuous LOS | R² | {:.4f} |
| Cox PH | Time to Discharge | C-index | {:.4f} |
""".format(
        clf_results['test_auroc'],
        clf_results['test_auprc'],
        test_metrics['mae'],
        test_metrics['rmse'],
        test_metrics['r2'],
        cox_results['test_cindex']
    ))
except Exception as e:
    print(f"Could not load all results for comparison: {e}")

# %%
print("\n✅ Regression analysis complete!")
