# %% [markdown]
# # Survival Analysis for ICU Length of Stay
# 
# This script applies survival analysis to model time-to-discharge:
# - Kaplan-Meier curves
# - Cox Proportional Hazards model
# - Time-dependent evaluation (C-index)
# - Hazard ratios for clinical interpretation
#
# Advantages over binary classification:
# - Models the full distribution of LOS
# - Handles censoring (patients who die or transfer)
# - Provides interpretable hazard ratios
# - Can predict discharge probability at any time point

# %% Imports
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
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

X_train = splits['X_train'].copy()
X_val = splits['X_val'].copy()
X_test = splits['X_test'].copy()
meta_train = splits['meta_train'].copy()
meta_val = splits['meta_val'].copy()
meta_test = splits['meta_test'].copy()
feature_columns = splits['feature_columns']

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# %% Prepare survival data
def prepare_survival_data(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for survival analysis.
    
    - duration: LOS in days (time to event)
    - event: 1 = discharged (event occurred), 0 = censored (died or still in ICU)
    
    For LOS prediction, we treat discharge as the event of interest.
    Death is treated as censoring (patient didn't experience discharge).
    """
    df = X.copy()
    
    # Duration is LOS in days
    df['duration'] = meta['los_days'].values
    
    # Event: 1 = discharged alive, 0 = died (censored)
    # In this dataset, mortality is in meta
    df['event'] = (meta['mortality'] == 0).astype(int).values
    
    # Cap extreme values
    df['duration'] = df['duration'].clip(lower=0.1, upper=90)
    
    return df

print("\nPreparing survival data...")
train_surv = prepare_survival_data(X_train, meta_train)
val_surv = prepare_survival_data(X_val, meta_val)
test_surv = prepare_survival_data(X_test, meta_test)

print(f"Train: {len(train_surv)} patients, {train_surv['event'].sum()} events ({train_surv['event'].mean()*100:.1f}%)")
print(f"Duration: mean={train_surv['duration'].mean():.2f}, median={train_surv['duration'].median():.2f} days")

# %% Kaplan-Meier Analysis
print("\n" + "=" * 60)
print("KAPLAN-MEIER ANALYSIS")
print("=" * 60)

# Overall survival curve
kmf = KaplanMeierFitter()
kmf.fit(train_surv['duration'], event_observed=train_surv['event'], label='All Patients')

print(f"\nMedian time to discharge: {kmf.median_survival_time_:.2f} days")
print(f"25th percentile: {kmf.percentile(0.25):.2f} days")
print(f"75th percentile: {kmf.percentile(0.75):.2f} days")

# %% Kaplan-Meier by subgroups
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Overall KM curve
ax = axes[0, 0]
kmf.plot_survival_function(ax=ax, ci_show=True)
ax.set_xlabel('Days')
ax.set_ylabel('Probability of Still Being in ICU')
ax.set_title('Overall ICU Stay Duration')
ax.set_xlim(0, 30)

# 2. By invasive ventilation (first 24h)
ax = axes[0, 1]
# Check if patient was ever on invasive vent in first 24h
vent_col = 'invasive_max' if 'invasive_max' in train_surv.columns else 'invasive_sum'
if vent_col in train_surv.columns:
    for vent_status, label, color in [(0, 'No Invasive Vent', 'green'), (1, 'Invasive Vent', 'red')]:
        mask = train_surv[vent_col] == vent_status
        if mask.sum() > 10:
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(train_surv.loc[mask, 'duration'], 
                        event_observed=train_surv.loc[mask, 'event'],
                        label=f'{label} (n={mask.sum()})')
            kmf_temp.plot_survival_function(ax=ax, ci_show=False, color=color)
    ax.set_xlabel('Days')
    ax.set_ylabel('Probability of Still Being in ICU')
    ax.set_title('ICU Stay by Invasive Ventilation Status')
    ax.set_xlim(0, 30)
    ax.legend()

# 3. By vasopressor use
ax = axes[1, 0]
vaso_col = 'vasopressor_max' if 'vasopressor_max' in train_surv.columns else 'vasopressor_sum'
if vaso_col in train_surv.columns:
    train_surv['vaso_binary'] = (train_surv[vaso_col] > 0).astype(int)
    for vaso_status, label, color in [(0, 'No Vasopressors', 'green'), (1, 'On Vasopressors', 'red')]:
        mask = train_surv['vaso_binary'] == vaso_status
        if mask.sum() > 10:
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(train_surv.loc[mask, 'duration'], 
                        event_observed=train_surv.loc[mask, 'event'],
                        label=f'{label} (n={mask.sum()})')
            kmf_temp.plot_survival_function(ax=ax, ci_show=False, color=color)
    ax.set_xlabel('Days')
    ax.set_ylabel('Probability of Still Being in ICU')
    ax.set_title('ICU Stay by Vasopressor Use')
    ax.set_xlim(0, 30)
    ax.legend()

# 4. By GCS (high vs low)
ax = axes[1, 1]
gcs_col = 'gcs_mean' if 'gcs_mean' in train_surv.columns else 'gcs_first'
if gcs_col in train_surv.columns:
    gcs_median = train_surv[gcs_col].median()
    train_surv['gcs_group'] = (train_surv[gcs_col] >= gcs_median).astype(int)
    for gcs_status, label, color in [(0, f'GCS < {gcs_median:.0f}', 'red'), 
                                      (1, f'GCS ≥ {gcs_median:.0f}', 'green')]:
        mask = train_surv['gcs_group'] == gcs_status
        if mask.sum() > 10:
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(train_surv.loc[mask, 'duration'], 
                        event_observed=train_surv.loc[mask, 'event'],
                        label=f'{label} (n={mask.sum()})')
            kmf_temp.plot_survival_function(ax=ax, ci_show=False, color=color)
    ax.set_xlabel('Days')
    ax.set_ylabel('Probability of Still Being in ICU')
    ax.set_title('ICU Stay by GCS Score')
    ax.set_xlim(0, 30)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_kaplan_meier.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR / '06_kaplan_meier.png'}")

# %% Feature Selection for Cox Model
print("\n" + "=" * 60)
print("COX PROPORTIONAL HAZARDS MODEL")
print("=" * 60)

# Cox models can struggle with many features
# Select top features based on XGBoost importance or clinical relevance

# Load XGBoost feature importance
try:
    importance_df = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')
    top_features = importance_df.head(30)['feature'].tolist()
    print(f"\nUsing top 30 features from XGBoost importance")
except:
    # Fallback: use clinically relevant features
    top_features = [
        'anchor_age', 'elixhauser_vanwalraven', 'gender_encoded',
        'heart_rate_mean', 'heart_rate_std', 'sbp_mean', 'mbp_mean',
        'spo2_mean', 'spo2_min', 'temperature_mean',
        'gcs_mean', 'gcs_min',
        'invasive_sum', 'invasive_max',
        'vasopressor_sum', 'vasopressor_max',
        'sofa_24hours_mean', 'sofa_24hours_max',
        'set_fio2_mean', 'set_peep_mean',
        'pH_mean', 'pO2_mean', 'pCO2_mean'
    ]
    top_features = [f for f in top_features if f in feature_columns]
    print(f"\nUsing {len(top_features)} clinically relevant features")

print(f"Selected features: {len(top_features)}")

# %% Prepare data for Cox model
def prepare_cox_data(surv_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Prepare data for Cox model with selected features."""
    
    # Select features + duration + event
    available_features = [f for f in features if f in surv_df.columns]
    cols = available_features + ['duration', 'event']
    
    df = surv_df[cols].copy()
    
    # Handle missing values (Cox model can't handle NaN)
    df = df.fillna(df.median())
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    return df, available_features

train_cox, cox_features = prepare_cox_data(train_surv, top_features)
val_cox, _ = prepare_cox_data(val_surv, top_features)
test_cox, _ = prepare_cox_data(test_surv, top_features)

print(f"\nCox model features: {len(cox_features)}")
print(f"Training samples: {len(train_cox)}")

# %% Fit Cox Proportional Hazards Model
print("\nFitting Cox PH model...")

cph = CoxPHFitter(penalizer=0.1)  # L2 regularization to handle collinearity

try:
    cph.fit(train_cox, duration_col='duration', event_col='event')
    print("Model fitted successfully!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COX MODEL SUMMARY")
    print("=" * 60)
    cph.print_summary(decimals=3, columns=['coef', 'exp(coef)', 'p', 'coef lower 95%', 'coef upper 95%'])
    
except Exception as e:
    print(f"Error fitting model: {e}")
    print("Trying with fewer features...")
    
    # Use even fewer features
    minimal_features = [f for f in [
        'anchor_age', 'gcs_verbal_last', 'set_fio2_count', 
        'vasopressor_sum', 'elixhauser_vanwalraven',
        'heart_rate_mean', 'spo2_min', 'invasive_max'
    ] if f in train_surv.columns]
    
    train_cox, cox_features = prepare_cox_data(train_surv, minimal_features)
    val_cox, _ = prepare_cox_data(val_surv, minimal_features)
    test_cox, _ = prepare_cox_data(test_surv, minimal_features)
    
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(train_cox, duration_col='duration', event_col='event')
    print("Model fitted with minimal features!")
    cph.print_summary(decimals=3)

# %% Evaluate Cox Model
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Calculate concordance index (C-index)
def calculate_cindex(model, df, duration_col='duration', event_col='event'):
    """Calculate concordance index for Cox model."""
    predictions = model.predict_partial_hazard(df)
    cindex = concordance_index(
        df[duration_col], 
        -predictions,  # Negative because higher hazard = shorter survival
        df[event_col]
    )
    return cindex

train_cindex = calculate_cindex(cph, train_cox)
val_cindex = calculate_cindex(cph, val_cox)
test_cindex = calculate_cindex(cph, test_cox)

print(f"\nConcordance Index (C-index):")
print(f"  Train: {train_cindex:.4f}")
print(f"  Val:   {val_cindex:.4f}")
print(f"  Test:  {test_cindex:.4f}")

# %% Hazard Ratios Visualization
print("\n" + "=" * 60)
print("HAZARD RATIOS")
print("=" * 60)

# Get hazard ratios
hr_df = pd.DataFrame({
    'feature': cph.summary.index,
    'hazard_ratio': cph.summary['exp(coef)'],
    'lower_ci': cph.summary['exp(coef) lower 95%'],
    'upper_ci': cph.summary['exp(coef) upper 95%'],
    'p_value': cph.summary['p']
})

# Sort by hazard ratio deviation from 1
hr_df['hr_deviation'] = np.abs(np.log(hr_df['hazard_ratio']))
hr_df = hr_df.sort_values('hr_deviation', ascending=False)

print("\nTop Hazard Ratios (sorted by effect size):")
print(hr_df[['feature', 'hazard_ratio', 'lower_ci', 'upper_ci', 'p_value']].head(15).to_string(index=False))

# %% Forest Plot of Hazard Ratios
fig, ax = plt.subplots(figsize=(10, max(8, len(cox_features) * 0.4)))

# Use top 20 features by effect size
plot_df = hr_df.head(20).copy()
plot_df = plot_df.sort_values('hazard_ratio')

y_pos = range(len(plot_df))

# Plot hazard ratios with confidence intervals
ax.errorbar(
    plot_df['hazard_ratio'], y_pos,
    xerr=[plot_df['hazard_ratio'] - plot_df['lower_ci'],
          plot_df['upper_ci'] - plot_df['hazard_ratio']],
    fmt='o', color='steelblue', capsize=3, capthick=1, markersize=6
)

# Reference line at HR=1
ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='HR=1 (no effect)')

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['feature'], fontsize=9)
ax.set_xlabel('Hazard Ratio (95% CI)')
ax.set_title('Cox Proportional Hazards - Feature Effects\n(HR > 1: faster discharge, HR < 1: longer stay)')
ax.legend(loc='upper right')

# Log scale if range is large
if plot_df['hazard_ratio'].max() / plot_df['hazard_ratio'].min() > 10:
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_hazard_ratios.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR / '07_hazard_ratios.png'}")

# %% Survival Curves for Specific Patient Profiles
print("\n" + "=" * 60)
print("PREDICTED SURVIVAL CURVES")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

# Create example patient profiles
# Low risk patient
low_risk = train_cox[cox_features].median().to_frame().T
low_risk['gcs_verbal_last'] = train_cox['gcs_verbal_last'].quantile(0.75) if 'gcs_verbal_last' in cox_features else low_risk['gcs_verbal_last']
if 'vasopressor_sum' in cox_features:
    low_risk['vasopressor_sum'] = 0
if 'invasive_max' in cox_features:
    low_risk['invasive_max'] = 0

# High risk patient
high_risk = train_cox[cox_features].median().to_frame().T
high_risk['gcs_verbal_last'] = train_cox['gcs_verbal_last'].quantile(0.25) if 'gcs_verbal_last' in cox_features else high_risk['gcs_verbal_last']
if 'vasopressor_sum' in cox_features:
    high_risk['vasopressor_sum'] = train_cox['vasopressor_sum'].quantile(0.9)
if 'invasive_max' in cox_features:
    high_risk['invasive_max'] = 1

# Plot survival curves
cph.plot_partial_effects_on_outcome(
    covariates='gcs_verbal_last' if 'gcs_verbal_last' in cox_features else cox_features[0],
    values=[train_cox['gcs_verbal_last'].quantile(0.25) if 'gcs_verbal_last' in cox_features else train_cox[cox_features[0]].quantile(0.25),
            train_cox['gcs_verbal_last'].quantile(0.75) if 'gcs_verbal_last' in cox_features else train_cox[cox_features[0]].quantile(0.75)],
    ax=ax,
    cmap='coolwarm'
)

ax.set_xlabel('Days')
ax.set_ylabel('Probability of Still Being in ICU')
ax.set_title('Predicted ICU Stay by Patient Risk Profile')
ax.set_xlim(0, 21)
ax.legend(['High Risk (Low GCS)', 'Low Risk (High GCS)'])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_survival_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR / '08_survival_predictions.png'}")

# %% Comparison with XGBoost
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Load XGBoost results
try:
    with open(OUTPUT_DIR / 'xgboost_results.pkl', 'rb') as f:
        xgb_results = pickle.load(f)
    
    print("\n| Model | Metric | Value |")
    print("|-------|--------|-------|")
    print(f"| XGBoost | AUROC | {xgb_results['test_auroc']:.4f} |")
    print(f"| XGBoost | AUPRC | {xgb_results['test_auprc']:.4f} |")
    print(f"| Cox PH | C-index | {test_cindex:.4f} |")
    
except:
    print("XGBoost results not found for comparison")

# %% Save Results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save Cox model
with open(OUTPUT_DIR / 'cox_model.pkl', 'wb') as f:
    pickle.dump(cph, f)
print(f"Saved: {OUTPUT_DIR / 'cox_model.pkl'}")

# Save results summary
cox_results = {
    'train_cindex': train_cindex,
    'val_cindex': val_cindex,
    'test_cindex': test_cindex,
    'features': cox_features,
    'hazard_ratios': hr_df.to_dict('records'),
    'model_summary': cph.summary.to_dict()
}

with open(OUTPUT_DIR / 'cox_results.pkl', 'wb') as f:
    pickle.dump(cox_results, f)
print(f"Saved: {OUTPUT_DIR / 'cox_results.pkl'}")

# Save hazard ratios to CSV
hr_df.to_csv(OUTPUT_DIR / 'hazard_ratios.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'hazard_ratios.csv'}")

# %% Summary for Report
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)

print(f"""
## Cox Proportional Hazards Results

### Model Configuration
- Outcome: Time to ICU discharge (days)
- Event: Discharge alive (death = censored)
- Features: {len(cox_features)} (top features from XGBoost)
- Regularization: L2 (penalizer=0.1)

### Performance
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| C-index | {train_cindex:.4f} | {val_cindex:.4f} | {test_cindex:.4f} |

### Clinical Interpretation
Hazard Ratio (HR) interpretation:
- HR > 1: Higher hazard of discharge → shorter ICU stay
- HR < 1: Lower hazard of discharge → longer ICU stay
- HR = 1: No effect on ICU duration

### Key Findings
""")

# Print significant hazard ratios
sig_hr = hr_df[hr_df['p_value'] < 0.05].head(10)
for _, row in sig_hr.iterrows():
    direction = "shorter" if row['hazard_ratio'] > 1 else "longer"
    print(f"- {row['feature']}: HR={row['hazard_ratio']:.2f} → {direction} stay (p={row['p_value']:.3f})")

# %% Next Steps
print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
Survival analysis complete! You now have:

1. XGBoost Classification
   - Binary LOS prediction (short vs long)
   - AUROC = 0.88

2. Cox Survival Model
   - Time-to-discharge modeling
   - Interpretable hazard ratios
   - C-index for evaluation

For your report draft (Dec 17):
- Include both models in Methods section
- Compare AUROC vs C-index
- Discuss clinical interpretation of hazard ratios
- Show Kaplan-Meier curves for patient subgroups

Remaining tasks:
- Fairness analysis (by demographics)
- Run on full 50k dataset
- Prepare presentation figures
""")

# %%
print("\n✅ Survival analysis complete!")
