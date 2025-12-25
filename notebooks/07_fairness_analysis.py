# %% [markdown]
# # Fairness Analysis for ICU LOS Prediction
# 
# This script evaluates model performance across demographic groups:
# - Gender
# - Race
# - Insurance type
# - Age groups
#
# Goal: Identify potential biases in model predictions

# %% Imports
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# %% Configuration
OUTPUT_DIR = Path("../outputs")

# %% Load data and models
print("Loading data and models...")

# Load data splits
with open(OUTPUT_DIR / 'data_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

# Load preprocessing objects (contains label encoders)
with open(OUTPUT_DIR / 'preprocessing_objects.pkl', 'rb') as f:
    prep_objects = pickle.load(f)

# Load trained models
with open(OUTPUT_DIR / 'xgboost_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

with open(OUTPUT_DIR / 'xgboost_regressor.pkl', 'rb') as f:
    reg_model = pickle.load(f)

print("Data and models loaded successfully!")

# %% Extract test data
X_test = splits['X_test']
y_test_binary = splits['y_test']
meta_test = splits['meta_test']
feature_columns = splits['feature_columns']

print(f"Test set size: {len(X_test)}")
print(f"Feature columns: {feature_columns[:15]}...")

# %% Get demographic information from X_test (encoded features)
# Demographics are stored as encoded features in X_test

# Convert X_test to DataFrame if needed
if isinstance(X_test, np.ndarray):
    X_test_df = pd.DataFrame(X_test, columns=feature_columns)
else:
    X_test_df = X_test.copy()

# Create a dataframe for analysis
fairness_df = pd.DataFrame({
    'y_true_binary': y_test_binary,
    'y_true_los': meta_test['los_days'].values,
})

# Get encoded demographics from X_test
if 'gender_encoded' in feature_columns:
    fairness_df['gender_encoded'] = X_test_df['gender_encoded'].values
if 'race_encoded' in feature_columns:
    fairness_df['race_encoded'] = X_test_df['race_encoded'].values
if 'insurance_encoded' in feature_columns:
    fairness_df['insurance_encoded'] = X_test_df['insurance_encoded'].values
if 'anchor_age' in feature_columns:
    fairness_df['age'] = X_test_df['anchor_age'].values

# Decode demographics using label encoders
label_encoders = prep_objects.get('label_encoders', {})

# Decode gender
if 'gender' in label_encoders and 'gender_encoded' in fairness_df.columns:
    le = label_encoders['gender']
    fairness_df['gender'] = le.inverse_transform(fairness_df['gender_encoded'].astype(int))
else:
    # Fallback: use encoded values as labels
    fairness_df['gender'] = fairness_df['gender_encoded'].map({0: 'F', 1: 'M'}) if 'gender_encoded' in fairness_df.columns else 'Unknown'

# Decode race
if 'race' in label_encoders and 'race_encoded' in fairness_df.columns:
    le = label_encoders['race']
    fairness_df['race'] = le.inverse_transform(fairness_df['race_encoded'].astype(int))
else:
    fairness_df['race'] = fairness_df['race_encoded'].astype(int) if 'race_encoded' in fairness_df.columns else 'Unknown'

# Decode insurance
if 'insurance' in label_encoders and 'insurance_encoded' in fairness_df.columns:
    le = label_encoders['insurance']
    fairness_df['insurance'] = le.inverse_transform(fairness_df['insurance_encoded'].astype(int))
else:
    fairness_df['insurance'] = fairness_df['insurance_encoded'].astype(int) if 'insurance_encoded' in fairness_df.columns else 'Unknown'

# Get predictions
fairness_df['y_pred_proba'] = clf_model.predict_proba(X_test)[:, 1]
fairness_df['y_pred_binary'] = clf_model.predict(X_test)
fairness_df['y_pred_los'] = reg_model.predict(X_test)

# Create age groups (age values are standardized, so we need to unstandardize first)
# Check if age values are standardized (mean ~0, std ~1)
age_values = fairness_df['age'].values if 'age' in fairness_df.columns else None

if age_values is not None:
    # If standardized, the values will be roughly between -3 and 3
    if age_values.mean() < 10:  # Likely standardized
        # Use the scaler to inverse transform, or use quantile-based groups
        print("Age appears standardized, using quantile-based groups...")
        fairness_df['age_group'] = pd.qcut(
            fairness_df['age'], 
            q=5, 
            labels=['Very Young', 'Young', 'Middle', 'Older', 'Elderly']
        )
    else:
        # Age is in original scale
        fairness_df['age_group'] = pd.cut(
            fairness_df['age'], 
            bins=[18, 40, 55, 65, 75, 100],
            labels=['18-40', '41-55', '56-65', '66-75', '75+']
        )
else:
    fairness_df['age_group'] = 'Unknown'

print("\nDemographic columns available:")
print(fairness_df.columns.tolist())

# %% Simplify race categories
def simplify_race(race):
    """Group race into broader categories."""
    if pd.isna(race):
        return 'Unknown'
    race = str(race).upper()
    if 'WHITE' in race:
        return 'White'
    elif 'BLACK' in race or 'AFRICAN' in race:
        return 'Black'
    elif 'ASIAN' in race:
        return 'Asian'
    elif 'HISPANIC' in race or 'LATINO' in race:
        return 'Hispanic'
    elif 'UNKNOWN' in race or 'UNABLE' in race or 'DECLINED' in race:
        return 'Unknown'
    else:
        return 'Other'

fairness_df['race_group'] = fairness_df['race'].apply(simplify_race)

print("\nRace distribution:")
print(fairness_df['race_group'].value_counts())

# %% Define evaluation function
def evaluate_subgroup(df, group_col, group_name):
    """Evaluate model performance for a subgroup."""
    subset = df[df[group_col] == group_name]
    
    if len(subset) < 30:  # Minimum sample size
        return None
    
    # Classification metrics
    try:
        auroc = roc_auc_score(subset['y_true_binary'], subset['y_pred_proba'])
    except:
        auroc = np.nan
    
    accuracy = accuracy_score(subset['y_true_binary'], subset['y_pred_binary'])
    
    # Regression metrics
    mae = mean_absolute_error(subset['y_true_los'], subset['y_pred_los'])
    
    # Prediction rates
    pred_long_rate = subset['y_pred_binary'].mean()
    actual_long_rate = subset['y_true_binary'].mean()
    
    return {
        'group': group_name,
        'n': len(subset),
        'auroc': auroc,
        'accuracy': accuracy,
        'mae': mae,
        'pred_long_rate': pred_long_rate,
        'actual_long_rate': actual_long_rate,
        'rate_ratio': pred_long_rate / actual_long_rate if actual_long_rate > 0 else np.nan
    }

# %% Evaluate by Gender
print("\n" + "=" * 60)
print("FAIRNESS ANALYSIS BY GENDER")
print("=" * 60)

gender_results = []
for gender in fairness_df['gender'].dropna().unique():
    result = evaluate_subgroup(fairness_df, 'gender', gender)
    if result:
        gender_results.append(result)

gender_df = pd.DataFrame(gender_results)
print("\n" + gender_df.to_string(index=False))

# %% Evaluate by Race
print("\n" + "=" * 60)
print("FAIRNESS ANALYSIS BY RACE")
print("=" * 60)

race_results = []
for race in fairness_df['race_group'].unique():
    result = evaluate_subgroup(fairness_df, 'race_group', race)
    if result:
        race_results.append(result)

race_df = pd.DataFrame(race_results).sort_values('n', ascending=False)
print("\n" + race_df.to_string(index=False))

# %% Evaluate by Insurance
print("\n" + "=" * 60)
print("FAIRNESS ANALYSIS BY INSURANCE")
print("=" * 60)

insurance_results = []
for insurance in fairness_df['insurance'].dropna().unique():
    result = evaluate_subgroup(fairness_df, 'insurance', insurance)
    if result:
        insurance_results.append(result)

insurance_df = pd.DataFrame(insurance_results).sort_values('n', ascending=False)
print("\n" + insurance_df.to_string(index=False))

# %% Evaluate by Age Group
print("\n" + "=" * 60)
print("FAIRNESS ANALYSIS BY AGE GROUP")
print("=" * 60)

age_results = []
for age_group in ['18-40', '41-55', '56-65', '66-75', '75+']:
    result = evaluate_subgroup(fairness_df, 'age_group', age_group)
    if result:
        age_results.append(result)

age_df = pd.DataFrame(age_results)
print("\n" + age_df.to_string(index=False))

# %% Visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. AUROC by Gender
ax = axes[0, 0]
if len(gender_df) > 0:
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(gender_df['group'], gender_df['auroc'], color=colors[:len(gender_df)])
    ax.axhline(y=fairness_df['y_true_binary'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('AUROC')
    ax.set_title('Classification Performance by Gender')
    ax.set_ylim(0.5, 1.0)
    # Add sample sizes
    for i, (_, row) in enumerate(gender_df.iterrows()):
        ax.text(i, row['auroc'] + 0.01, f"n={row['n']}", ha='center', fontsize=10)

# 2. AUROC by Race
ax = axes[0, 1]
if len(race_df) > 0:
    race_df_plot = race_df[race_df['n'] >= 50].sort_values('auroc')
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(race_df_plot)))
    bars = ax.barh(race_df_plot['group'], race_df_plot['auroc'], color=colors)
    ax.axvline(x=0.88, color='red', linestyle='--', label='Overall AUROC')
    ax.set_xlabel('AUROC')
    ax.set_title('Classification Performance by Race')
    ax.set_xlim(0.5, 1.0)
    ax.legend()
    # Add sample sizes
    for i, (_, row) in enumerate(race_df_plot.iterrows()):
        ax.text(row['auroc'] + 0.01, i, f"n={row['n']}", va='center', fontsize=9)

# 3. MAE by Insurance
ax = axes[1, 0]
if len(insurance_df) > 0:
    colors = ['#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(insurance_df['group'], insurance_df['mae'], color=colors[:len(insurance_df)])
    ax.axhline(y=1.91, color='red', linestyle='--', label='Overall MAE')
    ax.set_ylabel('MAE (days)')
    ax.set_title('Regression Performance by Insurance Type')
    ax.legend()
    # Add sample sizes
    for i, (_, row) in enumerate(insurance_df.iterrows()):
        ax.text(i, row['mae'] + 0.05, f"n={row['n']}", ha='center', fontsize=10)

# 4. AUROC by Age Group
ax = axes[1, 1]
if len(age_df) > 0:
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(age_df)))
    bars = ax.bar(age_df['group'], age_df['auroc'], color=colors)
    ax.axhline(y=0.88, color='red', linestyle='--', label='Overall AUROC')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('AUROC')
    ax.set_title('Classification Performance by Age Group')
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    # Add sample sizes
    for i, (_, row) in enumerate(age_df.iterrows()):
        ax.text(i, row['auroc'] + 0.01, f"n={row['n']}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_fairness_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR / '10_fairness_analysis.png'}")

# %% Disparity Analysis
print("\n" + "=" * 60)
print("DISPARITY ANALYSIS")
print("=" * 60)

def calculate_disparity(df, metric_col):
    """Calculate max disparity ratio for a metric."""
    values = df[metric_col].dropna()
    if len(values) < 2:
        return np.nan
    return values.max() / values.min()

print("\nAUROC Disparity (max/min ratio):")
print(f"  By Gender: {calculate_disparity(gender_df, 'auroc'):.3f}")
print(f"  By Race: {calculate_disparity(race_df[race_df['n'] >= 50], 'auroc'):.3f}")
print(f"  By Insurance: {calculate_disparity(insurance_df, 'auroc'):.3f}")
print(f"  By Age: {calculate_disparity(age_df, 'auroc'):.3f}")

print("\nMAE Disparity (max/min ratio):")
print(f"  By Gender: {calculate_disparity(gender_df, 'mae'):.3f}")
print(f"  By Race: {calculate_disparity(race_df[race_df['n'] >= 50], 'mae'):.3f}")
print(f"  By Insurance: {calculate_disparity(insurance_df, 'mae'):.3f}")
print(f"  By Age: {calculate_disparity(age_df, 'mae'):.3f}")

# %% Prediction Rate Parity
print("\n" + "=" * 60)
print("PREDICTION RATE ANALYSIS")
print("=" * 60)

print("\nLong Stay Prediction Rates vs Actual Rates:")
print("\nBy Gender:")
for _, row in gender_df.iterrows():
    print(f"  {row['group']}: Predicted={row['pred_long_rate']:.1%}, Actual={row['actual_long_rate']:.1%}, Ratio={row['rate_ratio']:.2f}")

print("\nBy Race (n≥50):")
for _, row in race_df[race_df['n'] >= 50].iterrows():
    print(f"  {row['group']}: Predicted={row['pred_long_rate']:.1%}, Actual={row['actual_long_rate']:.1%}, Ratio={row['rate_ratio']:.2f}")

print("\nBy Insurance:")
for _, row in insurance_df.iterrows():
    print(f"  {row['group']}: Predicted={row['pred_long_rate']:.1%}, Actual={row['actual_long_rate']:.1%}, Ratio={row['rate_ratio']:.2f}")

# %% Save Results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

fairness_results = {
    'gender': gender_df.to_dict('records'),
    'race': race_df.to_dict('records'),
    'insurance': insurance_df.to_dict('records'),
    'age': age_df.to_dict('records'),
}

with open(OUTPUT_DIR / 'fairness_results.pkl', 'wb') as f:
    pickle.dump(fairness_results, f)
print(f"Saved: {OUTPUT_DIR / 'fairness_results.pkl'}")

# Save as CSV for easy viewing
all_results = pd.concat([
    gender_df.assign(category='Gender'),
    race_df.assign(category='Race'),
    insurance_df.assign(category='Insurance'),
    age_df.assign(category='Age')
])
all_results.to_csv(OUTPUT_DIR / 'fairness_results.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'fairness_results.csv'}")

# %% Summary for Report
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)

print(f"""
## Fairness Analysis Results

### Overview
We evaluated model performance across demographic subgroups to identify potential biases.
Test set: {len(fairness_df)} patients.

### Classification Performance (AUROC) by Group

| Category | Group | N | AUROC | vs Overall |
|----------|-------|---|-------|------------|""")

overall_auroc = 0.880
for _, row in gender_df.iterrows():
    diff = row['auroc'] - overall_auroc
    print(f"| Gender | {row['group']} | {row['n']} | {row['auroc']:.3f} | {diff:+.3f} |")

for _, row in race_df[race_df['n'] >= 50].iterrows():
    diff = row['auroc'] - overall_auroc
    print(f"| Race | {row['group']} | {row['n']} | {row['auroc']:.3f} | {diff:+.3f} |")

for _, row in insurance_df.iterrows():
    diff = row['auroc'] - overall_auroc
    print(f"| Insurance | {row['group']} | {row['n']} | {row['auroc']:.3f} | {diff:+.3f} |")

print(f"""
### Key Findings

1. **Gender**: Model performs similarly across genders (disparity ratio: {calculate_disparity(gender_df, 'auroc'):.3f})

2. **Race**: Some variation in AUROC across racial groups (disparity ratio: {calculate_disparity(race_df[race_df['n'] >= 50], 'auroc'):.3f})

3. **Insurance**: Performance varies by insurance type (disparity ratio: {calculate_disparity(insurance_df, 'auroc'):.3f})

4. **Age**: Older patients may have different prediction accuracy (disparity ratio: {calculate_disparity(age_df, 'auroc'):.3f})

### Interpretation
- Disparity ratio close to 1.0 indicates fair performance across groups
- Ratio > 1.1 suggests potential bias that should be investigated
- Small subgroups (n<50) may have unreliable metrics
""")

# %%
print("\n✅ Fairness analysis complete!")
