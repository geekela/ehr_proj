# %% [markdown]
# # ICU Length of Stay Prediction - Data Exploration
# 
# This notebook explores the Temporal Respiratory Support dataset from PhysioNet.
# Dataset: ~50,920 adult ICU patients from MIMIC-IV with hourly data over 90 days.
import os
print("Current working directory:", os.getcwd())

# %% Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# %% Configuration

DATA_DIR = Path("../data/temporal-respiratory-support")

# Verify the path exists
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

print(f"Data directory: {DATA_DIR}")
print(f"Folders found: {sorted([f.name for f in DATA_DIR.iterdir() if f.is_dir()])}")

# %% Function to load all patient files
def load_all_patients(data_dir: Path, sample_size: int = None) -> pd.DataFrame:
    """
    Load all patient CSV files from the data directory.
    
    Args:
        data_dir: Path to the data directory containing numbered folders
        sample_size: If set, only load this many patients (for quick testing)
    
    Returns:
        DataFrame with all patient data
    """
    all_files = []
    
    # Find all CSV files in numbered subdirectories
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            csv_files = list(folder.glob("*.csv"))
            all_files.extend(csv_files)
    
    print(f"Found {len(all_files)} patient files")
    
    if sample_size:
        all_files = all_files[:sample_size]
        print(f"Loading sample of {sample_size} patients...")
    
    # Load all files
    dfs = []
    for f in tqdm(all_files, desc="Loading patients"):
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined):,}")
    
    return combined

# %% Load a sample first (faster for exploration)
# Start with 100 patients to understand the data structure
df_sample = load_all_patients(DATA_DIR, sample_size=100)

# %% Basic info about the dataset
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df_sample.shape}")
print(f"Columns: {len(df_sample.columns)}")
print(f"\nColumn names:\n{df_sample.columns.tolist()}")

# %% Data types
print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df_sample.dtypes)

# %% Extract unique patients and their static info
def get_patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per patient with static variables."""
    
    static_cols = [
        'subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime',
        'gender', 'anchor_age', 'race', 'insurance', 'language', 
        'marital_status', 'first_careunit', 'elixhauser_vanwalraven',
        'height_inch', 'pbw_kg', 'los',
        'discharge_outcome', 'icuouttime_outcome', 'death_outcome'
    ]
    
    # Get first row per patient (static vars are repeated)
    patient_df = df.groupby('subject_id').first().reset_index()
    
    # Select only columns that exist
    available_cols = [c for c in static_cols if c in patient_df.columns]
    
    return patient_df[available_cols]

patients = get_patient_summary(df_sample)
print(f"\nUnique patients in sample: {len(patients)}")

# %% Patient demographics
print("\n" + "=" * 60)
print("DEMOGRAPHICS")
print("=" * 60)

print(f"\nAge: mean={patients['anchor_age'].mean():.1f}, "
      f"std={patients['anchor_age'].std():.1f}, "
      f"range=[{patients['anchor_age'].min()}, {patients['anchor_age'].max()}]")

print(f"\nGender distribution:")
print(patients['gender'].value_counts())

print(f"\nRace distribution:")
print(patients['race'].value_counts())

print(f"\nInsurance distribution:")
print(patients['insurance'].value_counts())

print(f"\nCare unit distribution:")
print(patients['first_careunit'].value_counts())

# %% Length of Stay (TARGET VARIABLE)
print("\n" + "=" * 60)
print("LENGTH OF STAY (TARGET)")
print("=" * 60)

print(f"\nLOS (days):")
print(f"  Mean:   {patients['los'].mean():.2f}")
print(f"  Median: {patients['los'].median():.2f}")
print(f"  Std:    {patients['los'].std():.2f}")
print(f"  Min:    {patients['los'].min():.2f}")
print(f"  Max:    {patients['los'].max():.2f}")

# LOS categories (for classification)
patients['los_category'] = pd.cut(
    patients['los'], 
    bins=[0, 2, 4, 7, 14, float('inf')],
    labels=['<2d', '2-4d', '4-7d', '7-14d', '>14d']
)
print(f"\nLOS Categories:")
print(patients['los_category'].value_counts().sort_index())

# Binary classification (short vs long, threshold = 4 days)
patients['los_binary'] = (patients['los'] >= 4).astype(int)
print(f"\nBinary LOS (threshold=4 days):")
print(f"  Short (<4 days): {(patients['los_binary'] == 0).sum()} ({(patients['los_binary'] == 0).mean()*100:.1f}%)")
print(f"  Long (≥4 days):  {(patients['los_binary'] == 1).sum()} ({(patients['los_binary'] == 1).mean()*100:.1f}%)")

# %% Outcomes
print("\n" + "=" * 60)
print("OUTCOMES")
print("=" * 60)

print(f"\nDischarge outcome (1=discharged):")
print(patients['discharge_outcome'].value_counts())

print(f"\nDeath outcome (1=died):")
print(patients['death_outcome'].value_counts())

mortality_rate = patients['death_outcome'].mean() * 100
print(f"\nMortality rate: {mortality_rate:.1f}%")

# %% Comorbidity scores
print("\n" + "=" * 60)
print("COMORBIDITY")
print("=" * 60)

print(f"\nElixhauser-Van Walraven Score:")
print(f"  Mean:   {patients['elixhauser_vanwalraven'].mean():.2f}")
print(f"  Median: {patients['elixhauser_vanwalraven'].median():.2f}")
print(f"  Std:    {patients['elixhauser_vanwalraven'].std():.2f}")
print(f"  Range:  [{patients['elixhauser_vanwalraven'].min()}, {patients['elixhauser_vanwalraven'].max()}]")

# %% Analyze temporal data structure
print("\n" + "=" * 60)
print("TEMPORAL DATA STRUCTURE")
print("=" * 60)

# Hours per patient
hours_per_patient = df_sample.groupby('subject_id')['hr'].count()
print(f"\nHours per patient:")
print(f"  Mean:   {hours_per_patient.mean():.0f}")
print(f"  Median: {hours_per_patient.median():.0f}")
print(f"  Min:    {hours_per_patient.min()}")
print(f"  Max:    {hours_per_patient.max()}")

# Actual ICU stay vs 90-day window
print(f"\nNote: Each patient has data for 90 days (2160 hours)")
print(f"Actual ICU stay is stored in 'los' column")

# %% Respiratory support analysis
print("\n" + "=" * 60)
print("RESPIRATORY SUPPORT")
print("=" * 60)

resp_cols = ['invasive', 'noninvasive', 'highflow']

# Hours on each type of support (per patient)
for col in resp_cols:
    hours_on = df_sample.groupby('subject_id')[col].sum()
    patients_with = (hours_on > 0).sum()
    print(f"\n{col.upper()}:")
    print(f"  Patients ever on {col}: {patients_with} ({patients_with/len(patients)*100:.1f}%)")
    print(f"  Hours on {col} (if any): mean={hours_on[hours_on > 0].mean():.1f}")

# Any mechanical ventilation
df_sample['any_vent'] = (df_sample['invasive'] == 1) | (df_sample['noninvasive'] == 1)
vent_hours = df_sample.groupby('subject_id')['any_vent'].sum()
print(f"\nANY VENTILATION (invasive or noninvasive):")
print(f"  Patients ever ventilated: {(vent_hours > 0).sum()} ({(vent_hours > 0).mean()*100:.1f}%)")

# %% Missingness analysis
print("\n" + "=" * 60)
print("MISSINGNESS ANALYSIS")
print("=" * 60)

# Calculate missingness for each column
missingness = df_sample.isnull().mean() * 100
missingness_sorted = missingness.sort_values(ascending=False)

print("\nColumns with >50% missing:")
print(missingness_sorted[missingness_sorted > 50])

print("\nColumns with <10% missing:")
print(missingness_sorted[missingness_sorted < 10])

# %% Define feature groups for modeling
print("\n" + "=" * 60)
print("FEATURE GROUPS FOR MODELING")
print("=" * 60)

feature_groups = {
    'identifiers': ['hr', 'subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'],
    
    'demographics': ['gender', 'anchor_age', 'race', 'insurance', 'language', 
                     'marital_status', 'first_careunit'],
    
    'targets': ['los', 'discharge_outcome', 'icuouttime_outcome', 'death_outcome'],
    
    'comorbidity': ['elixhauser_vanwalraven'],
    
    'gcs': ['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gcs_unable'],
    
    'respiratory_status': ['invasive', 'noninvasive', 'highflow'],
    
    'ventilator_settings': ['pinsp_draeger', 'pinsp_hamilton', 'ppeak', 'set_peep', 
                           'total_peep', 'pcv_level', 'rr', 'set_rr', 'total_rr',
                           'set_tv', 'total_tv', 'set_fio2', 'set_ie_ratio', 
                           'set_pc_draeger', 'set_pc'],
    
    'blood_gas': ['calculated_bicarbonate', 'pCO2', 'pH', 'pO2', 'so2'],
    
    'vitals': ['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni', 'mbp_ni',
               'temperature', 'spo2', 'glucose'],
    
    'interventions': ['vasopressor', 'crrt'],
    
    'severity': ['sepsis3', 'sofa_24hours'],
    
    'body_metrics': ['height_inch', 'pbw_kg']
}

for group, cols in feature_groups.items():
    available = [c for c in cols if c in df_sample.columns]
    missing_pct = df_sample[available].isnull().mean().mean() * 100
    print(f"\n{group.upper()} ({len(available)} features, {missing_pct:.1f}% missing avg):")
    print(f"  {available}")

# %% Visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. LOS distribution
ax = axes[0, 0]
ax.hist(patients['los'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(x=4, color='red', linestyle='--', label='4-day threshold')
ax.set_xlabel('Length of Stay (days)')
ax.set_ylabel('Count')
ax.set_title('Distribution of ICU Length of Stay')
ax.legend()

# 2. Age distribution
ax = axes[0, 1]
ax.hist(patients['anchor_age'], bins=30, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Age Distribution')

# 3. Gender distribution
ax = axes[0, 2]
gender_counts = patients['gender'].value_counts()
ax.bar(gender_counts.index, gender_counts.values, color=['steelblue', 'coral'])
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.set_title('Gender Distribution')

# 4. Care unit distribution
ax = axes[1, 0]
unit_counts = patients['first_careunit'].value_counts()
ax.barh(unit_counts.index, unit_counts.values, color='purple', alpha=0.7)
ax.set_xlabel('Count')
ax.set_title('Care Unit Distribution')

# 5. LOS by mortality
ax = axes[1, 1]
alive = patients[patients['death_outcome'] == 0]['los']
dead = patients[patients['death_outcome'] == 1]['los']
ax.boxplot([alive, dead], labels=['Survived', 'Died'])
ax.set_ylabel('Length of Stay (days)')
ax.set_title('LOS by Mortality Outcome')

# 6. Comorbidity score distribution
ax = axes[1, 2]
ax.hist(patients['elixhauser_vanwalraven'], bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Elixhauser-Van Walraven Score')
ax.set_ylabel('Count')
ax.set_title('Comorbidity Score Distribution')

plt.tight_layout()
plt.savefig('../outputs/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ../outputs/01_eda_overview.png")

# %% Missingness heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Select key features for visualization
key_features = (
    feature_groups['vitals'] + 
    feature_groups['blood_gas'] + 
    feature_groups['gcs'] +
    feature_groups['respiratory_status']
)
key_features = [f for f in key_features if f in df_sample.columns]

# Calculate missingness per feature
miss_matrix = df_sample[key_features].isnull().mean() * 100
miss_df = pd.DataFrame({'Feature': key_features, 'Missing %': miss_matrix.values})
miss_df = miss_df.sort_values('Missing %', ascending=True)

ax.barh(miss_df['Feature'], miss_df['Missing %'], color='crimson', alpha=0.7)
ax.set_xlabel('Missing %')
ax.set_title('Missingness by Feature (Key Clinical Variables)')
ax.axvline(x=30, color='black', linestyle='--', alpha=0.5, label='30% threshold')
ax.legend()

plt.tight_layout()
plt.savefig('../outputs/02_missingness.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ../outputs/02_missingness.png")

# %% Single patient trajectory visualization
print("\n" + "=" * 60)
print("SINGLE PATIENT TRAJECTORY EXAMPLE")
print("=" * 60)

# Pick one patient
example_patient = df_sample[df_sample['subject_id'] == df_sample['subject_id'].iloc[0]].copy()
patient_los = example_patient['los'].iloc[0]
print(f"Patient ID: {example_patient['subject_id'].iloc[0]}")
print(f"ICU LOS: {patient_los:.2f} days ({patient_los*24:.0f} hours)")

# Limit to actual ICU stay
actual_hours = int(patient_los * 24)
example_icu = example_patient[example_patient['hr'] <= actual_hours]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Heart rate
ax = axes[0]
ax.plot(example_icu['hr'], example_icu['heart_rate'], 'b-', alpha=0.7)
ax.set_ylabel('Heart Rate')
ax.set_title(f'Patient Trajectory (LOS={patient_los:.1f} days)')

# SpO2
ax = axes[1]
ax.plot(example_icu['hr'], example_icu['spo2'], 'g-', alpha=0.7)
ax.set_ylabel('SpO2 (%)')

# Respiratory support
ax = axes[2]
ax.fill_between(example_icu['hr'], example_icu['invasive'], alpha=0.5, label='Invasive', color='red')
ax.fill_between(example_icu['hr'], example_icu['noninvasive'], alpha=0.5, label='Non-invasive', color='orange')
ax.fill_between(example_icu['hr'], example_icu['highflow'], alpha=0.5, label='High-flow', color='green')
ax.set_ylabel('Respiratory Support')
ax.set_xlabel('Hour')
ax.legend()

plt.tight_layout()
plt.savefig('../outputs/03_patient_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ../outputs/03_patient_trajectory.png")

# %% Summary statistics for report
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)

summary = {
    'Total patients (sample)': len(patients),
    'Total hourly records': len(df_sample),
    'Mean age': f"{patients['anchor_age'].mean():.1f} ± {patients['anchor_age'].std():.1f}",
    'Male %': f"{(patients['gender'] == 'M').mean() * 100:.1f}%",
    'Mean LOS (days)': f"{patients['los'].mean():.2f} ± {patients['los'].std():.2f}",
    'Median LOS (days)': f"{patients['los'].median():.2f}",
    'Mortality rate': f"{patients['death_outcome'].mean() * 100:.1f}%",
    'Ventilated %': f"{(vent_hours > 0).mean() * 100:.1f}%",
    'Mean Elixhauser score': f"{patients['elixhauser_vanwalraven'].mean():.1f}"
}

for key, val in summary.items():
    print(f"  {key}: {val}")

# %% NEXT STEPS
print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
1. LOAD FULL DATASET
   - Change sample_size=None in load_all_patients() 
   - This will take longer but gives complete picture

2. DEFINE COHORT INCLUSION CRITERIA
   - Minimum ICU stay (e.g., ≥4 hours)
   - First ICU admission only?
   - Exclude patients who died within 24h?

3. FEATURE ENGINEERING (for classical models)
   - Aggregate first 24h of temporal features (mean, min, max, std)
   - Handle missing data (imputation strategy)
   - Encode categorical variables

4. SPLIT DATA
   - Train/validation/test split (e.g., 70/15/15)
   - Stratify by LOS category or outcome

5. BUILD BASELINE MODELS
   - XGBoost for LOS classification
   - Cox model for survival analysis
""")

# %%
print("\n Exploration complete! Check the 'outputs/' folder for figures.")
