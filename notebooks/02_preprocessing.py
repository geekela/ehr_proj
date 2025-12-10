# %% [markdown]
# # Preprocessing Pipeline for Classical Models
# 
# This script prepares data for XGBoost and Cox survival models.
# - Loads full dataset
# - Applies inclusion/exclusion criteria
# - Extracts static features
# - Aggregates first 24h of temporal features
# - Handles missing data
# - Creates train/val/test splits

# %% Imports
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

warnings.filterwarnings('ignore')

# %% Configuration
DATA_DIR = Path("../data/temporal-respiratory-support")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Modeling parameters
FIRST_N_HOURS = 24  # Use first 24 hours for prediction
MIN_ICU_HOURS = 4   # Minimum ICU stay to include
LOS_THRESHOLD = 4   # Days - for binary classification (short vs long)
RANDOM_STATE = 42

print(f"Configuration:")
print(f"  - First N hours for features: {FIRST_N_HOURS}")
print(f"  - Minimum ICU stay: {MIN_ICU_HOURS} hours")
print(f"  - LOS threshold: {LOS_THRESHOLD} days")

# %% Load all patient files
def load_all_patients(data_dir: Path, sample_size: int = None) -> pd.DataFrame:
    """Load all patient CSV files from the data directory."""
    all_files = []
    
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            csv_files = list(folder.glob("*.csv"))
            all_files.extend(csv_files)
    
    print(f"Found {len(all_files)} patient files")
    
    if sample_size:
        all_files = all_files[:sample_size]
        print(f"Loading sample of {sample_size} patients...")
    else:
        print(f"Loading all {len(all_files)} patients...")
    
    dfs = []
    for f in tqdm(all_files, desc="Loading patients"):
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined):,}")
    
    return combined

# %% Load data (use sample_size for testing, None for full dataset)
# Start with a sample for development, then set to None for full run
df_raw = load_all_patients(DATA_DIR, sample_size=10000)  # Change to None for full dataset

# %% Define feature groups
STATIC_FEATURES = [
    'gender', 'anchor_age', 'race', 'insurance', 'language',
    'marital_status', 'first_careunit', 'elixhauser_vanwalraven',
    'height_inch', 'pbw_kg'
]

TEMPORAL_FEATURES = {
    'vitals': ['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni', 
               'mbp_ni', 'temperature', 'spo2', 'glucose'],
    
    'respiratory_status': ['invasive', 'noninvasive', 'highflow'],
    
    'ventilator_settings': ['set_peep', 'total_peep', 'rr', 'set_rr', 
                           'total_rr', 'set_tv', 'total_tv', 'set_fio2'],
    
    'blood_gas': ['calculated_bicarbonate', 'pCO2', 'pH', 'pO2', 'so2'],
    
    'gcs': ['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes'],
    
    'interventions': ['vasopressor', 'crrt'],
    
    'severity': ['sepsis3', 'sofa_24hours']
}

# Flatten temporal features
ALL_TEMPORAL = []
for group, cols in TEMPORAL_FEATURES.items():
    ALL_TEMPORAL.extend(cols)

print(f"\nStatic features: {len(STATIC_FEATURES)}")
print(f"Temporal features: {len(ALL_TEMPORAL)}")

# %% Apply inclusion criteria
def apply_inclusion_criteria(df: pd.DataFrame, min_hours: int = 4) -> pd.DataFrame:
    """
    Apply cohort inclusion/exclusion criteria.
    
    Inclusion:
    - ICU stay >= min_hours
    - Age >= 18
    - First ICU admission only (already in dataset)
    
    Exclusion:
    - Missing LOS
    - LOS > 90 days (data artifact)
    """
    # Get unique patients with their LOS
    patients = df.groupby('subject_id').first().reset_index()
    initial_count = len(patients)
    print(f"\nInitial patients: {initial_count}")
    
    # Convert LOS to hours for filtering
    patients['los_hours'] = patients['los'] * 24
    
    # Apply criteria
    criteria = (
        (patients['los_hours'] >= min_hours) &  # Minimum stay
        (patients['anchor_age'] >= 18) &         # Adults only
        (patients['los'].notna()) &              # Has LOS
        (patients['los'] <= 90)                  # Not an artifact
    )
    
    included_patients = patients[criteria]['subject_id'].tolist()
    
    # Filter main dataframe
    df_filtered = df[df['subject_id'].isin(included_patients)].copy()
    
    final_count = len(included_patients)
    print(f"After inclusion criteria: {final_count} ({final_count/initial_count*100:.1f}%)")
    print(f"  - Excluded {initial_count - final_count} patients")
    
    return df_filtered

df_included = apply_inclusion_criteria(df_raw, min_hours=MIN_ICU_HOURS)

# %% Extract static features for each patient
def extract_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per patient with static features."""
    
    # Get first row per patient (static vars are repeated)
    static_df = df.groupby('subject_id').first().reset_index()
    
    # Select static columns + identifiers + targets
    keep_cols = ['subject_id', 'stay_id', 'los', 
                 'discharge_outcome', 'death_outcome'] + STATIC_FEATURES
    
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in static_df.columns]
    
    return static_df[keep_cols].copy()

static_features = extract_static_features(df_included)
print(f"\nExtracted static features for {len(static_features)} patients")
print(f"Columns: {static_features.columns.tolist()}")

# %% Aggregate temporal features from first N hours
def aggregate_temporal_features(df: pd.DataFrame, 
                                first_n_hours: int = 24,
                                features: list = None) -> pd.DataFrame:
    """
    Aggregate temporal features from first N hours of ICU stay.
    
    For each feature, compute:
    - mean, std, min, max
    - first, last (for trend)
    - count of non-null values
    
    For binary features (invasive, vasopressor, etc):
    - max (ever on), sum (hours on), mean (proportion)
    """
    if features is None:
        features = ALL_TEMPORAL
    
    # Filter to first N hours only
    df_first = df[df['hr'] < first_n_hours].copy()
    
    print(f"\nAggregating {len(features)} features from first {first_n_hours} hours...")
    
    # Binary features get different aggregations
    binary_features = ['invasive', 'noninvasive', 'highflow', 'vasopressor', 'crrt', 'sepsis3']
    continuous_features = [f for f in features if f not in binary_features]
    
    aggregations = {}
    
    # Continuous features: statistical summaries
    for feat in continuous_features:
        if feat in df_first.columns:
            aggregations[feat] = ['mean', 'std', 'min', 'max', 'first', 'last', 'count']
    
    # Binary features: sum (hours on), max (ever on), mean (proportion)
    for feat in binary_features:
        if feat in df_first.columns:
            aggregations[feat] = ['sum', 'max', 'mean']
    
    # Perform aggregation
    agg_df = df_first.groupby('subject_id').agg(aggregations)
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    
    # Add trend features (last - first) for key vitals
    trend_features = ['heart_rate', 'sbp', 'mbp', 'spo2', 'temperature']
    for feat in trend_features:
        first_col = f'{feat}_first'
        last_col = f'{feat}_last'
        if first_col in agg_df.columns and last_col in agg_df.columns:
            agg_df[f'{feat}_trend'] = agg_df[last_col] - agg_df[first_col]
    
    print(f"Created {len(agg_df.columns) - 1} aggregated features")
    
    return agg_df

temporal_features = aggregate_temporal_features(df_included, 
                                                first_n_hours=FIRST_N_HOURS,
                                                features=ALL_TEMPORAL)
print(f"\nTemporal features shape: {temporal_features.shape}")

# %% Merge static and temporal features
def merge_features(static_df: pd.DataFrame, 
                   temporal_df: pd.DataFrame) -> pd.DataFrame:
    """Merge static and temporal features."""
    
    merged = static_df.merge(temporal_df, on='subject_id', how='inner')
    print(f"\nMerged dataset: {merged.shape}")
    
    return merged

df_features = merge_features(static_features, temporal_features)

# %% Create target variables
def create_targets(df: pd.DataFrame, los_threshold: float = 4.0) -> pd.DataFrame:
    """
    Create target variables for modeling.
    
    - los_days: continuous LOS (for regression)
    - los_binary: 0 = short (<threshold), 1 = long (>=threshold)
    - mortality: death outcome
    - los_hours: for survival analysis
    """
    df = df.copy()
    
    # Continuous LOS
    df['los_days'] = df['los']
    df['los_hours'] = df['los'] * 24
    
    # Binary LOS
    df['los_binary'] = (df['los'] >= los_threshold).astype(int)
    
    # Mortality
    df['mortality'] = df['death_outcome'].astype(int)
    
    print(f"\nTarget variables created:")
    print(f"  LOS (days): mean={df['los_days'].mean():.2f}, median={df['los_days'].median():.2f}")
    print(f"  LOS binary (threshold={los_threshold}d):")
    print(f"    Short (<{los_threshold}d): {(df['los_binary']==0).sum()} ({(df['los_binary']==0).mean()*100:.1f}%)")
    print(f"    Long (>={los_threshold}d): {(df['los_binary']==1).sum()} ({(df['los_binary']==1).mean()*100:.1f}%)")
    print(f"  Mortality: {df['mortality'].sum()} ({df['mortality'].mean()*100:.1f}%)")
    
    return df

df_features = create_targets(df_features, los_threshold=LOS_THRESHOLD)

# %% Handle categorical variables
def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables.
    
    Returns:
        - DataFrame with encoded variables
        - Dictionary of encoders (for later use)
    """
    df = df.copy()
    encoders = {}
    
    categorical_cols = ['gender', 'race', 'insurance', 'language', 
                        'marital_status', 'first_careunit']
    
    for col in categorical_cols:
        if col in df.columns:
            # Fill missing with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
            # Label encode
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
            print(f"  {col}: {len(le.classes_)} categories")
    
    return df, encoders

print("\nEncoding categorical variables:")
df_features, label_encoders = encode_categorical(df_features)

# %% Handle missing values
def handle_missing(df: pd.DataFrame, 
                   strategy: str = 'median') -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values in numerical features.
    
    Strategy options:
    - 'median': fill with median
    - 'mean': fill with mean
    - 'zero': fill with zero
    
    Returns:
        - DataFrame with imputed values
        - Dictionary of imputation values
    """
    df = df.copy()
    imputation_values = {}
    
    # Get numerical columns (exclude identifiers, targets, original categoricals)
    exclude_cols = ['subject_id', 'stay_id', 'los', 'los_days', 'los_hours', 
                    'los_binary', 'mortality', 'discharge_outcome', 'death_outcome',
                    'gender', 'race', 'insurance', 'language', 'marital_status', 
                    'first_careunit']
    
    numerical_cols = [c for c in df.columns 
                      if c not in exclude_cols 
                      and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"\nHandling missing values for {len(numerical_cols)} numerical columns...")
    
    missing_before = df[numerical_cols].isnull().sum().sum()
    
    for col in numerical_cols:
        if df[col].isnull().any():
            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = 0
            
            df[col] = df[col].fillna(fill_value)
            imputation_values[col] = fill_value
    
    missing_after = df[numerical_cols].isnull().sum().sum()
    print(f"  Missing values: {missing_before:,} → {missing_after:,}")
    
    return df, imputation_values

df_features, imputation_values = handle_missing(df_features, strategy='median')

# %% Define final feature set for modeling
def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns for modeling."""
    
    exclude_cols = [
        # Identifiers
        'subject_id', 'stay_id',
        # Original targets
        'los', 'discharge_outcome', 'death_outcome',
        # Created targets
        'los_days', 'los_hours', 'los_binary', 'mortality',
        # Original categorical (use encoded versions)
        'gender', 'race', 'insurance', 'language', 'marital_status', 'first_careunit'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return feature_cols

feature_columns = get_feature_columns(df_features)
print(f"\nFinal feature set: {len(feature_columns)} features")

# %% Split data into train/validation/test
def split_data(df: pd.DataFrame, 
               feature_cols: list,
               target_col: str = 'los_binary',
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> dict:
    """
    Split data into train/validation/test sets.
    
    Stratified by target to maintain class balance.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Additional columns to keep for analysis
    meta_cols = ['subject_id', 'los_days', 'los_hours', 'los_binary', 'mortality']
    meta = df[meta_cols].copy()
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test, meta_trainval, meta_test = train_test_split(
        X, y, meta,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_trainval, y_trainval, meta_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=random_state
    )
    
    print(f"\nData split (stratified by {target_col}):")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%) - Class 1: {y_train.mean()*100:.1f}%")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%) - Class 1: {y_val.mean()*100:.1f}%")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%) - Class 1: {y_test.mean()*100:.1f}%")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
        'X_val': X_val, 'y_val': y_val, 'meta_val': meta_val,
        'X_test': X_test, 'y_test': y_test, 'meta_test': meta_test,
        'feature_columns': feature_cols
    }

# Split for binary classification
splits = split_data(df_features, feature_columns, 
                    target_col='los_binary',
                    test_size=0.15, val_size=0.15,
                    random_state=RANDOM_STATE)

# %% Scale features
def scale_features(splits: dict) -> dict:
    """Apply StandardScaler to features."""
    
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(splits['X_train'])
    X_val_scaled = scaler.transform(splits['X_val'])
    X_test_scaled = scaler.transform(splits['X_test'])
    
    # Convert back to DataFrames
    splits['X_train_scaled'] = pd.DataFrame(X_train_scaled, 
                                            columns=splits['feature_columns'],
                                            index=splits['X_train'].index)
    splits['X_val_scaled'] = pd.DataFrame(X_val_scaled,
                                          columns=splits['feature_columns'],
                                          index=splits['X_val'].index)
    splits['X_test_scaled'] = pd.DataFrame(X_test_scaled,
                                           columns=splits['feature_columns'],
                                           index=splits['X_test'].index)
    splits['scaler'] = scaler
    
    print("\nFeatures scaled using StandardScaler")
    
    return splits

splits = scale_features(splits)

# %% Save processed data
def save_processed_data(splits: dict, 
                        encoders: dict,
                        imputation: dict,
                        output_dir: Path):
    """Save all processed data and preprocessing objects."""
    
    # Save splits
    with open(output_dir / 'data_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    print(f"\nSaved: {output_dir / 'data_splits.pkl'}")
    
    # Save preprocessing objects
    preprocessing_objects = {
        'label_encoders': encoders,
        'imputation_values': imputation,
        'scaler': splits['scaler'],
        'feature_columns': splits['feature_columns']
    }
    with open(output_dir / 'preprocessing_objects.pkl', 'wb') as f:
        pickle.dump(preprocessing_objects, f)
    print(f"Saved: {output_dir / 'preprocessing_objects.pkl'}")
    
    # Save CSV versions for inspection
    splits['X_train'].assign(y=splits['y_train']).to_csv(
        output_dir / 'train_data.csv', index=False)
    splits['X_test'].assign(y=splits['y_test']).to_csv(
        output_dir / 'test_data.csv', index=False)
    print(f"Saved: {output_dir / 'train_data.csv'}")
    print(f"Saved: {output_dir / 'test_data.csv'}")

save_processed_data(splits, label_encoders, imputation_values, OUTPUT_DIR)

# %% Summary statistics for report
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

summary = {
    'Total patients': len(df_features),
    'Features': len(feature_columns),
    'Training samples': len(splits['X_train']),
    'Validation samples': len(splits['X_val']),
    'Test samples': len(splits['X_test']),
    'LOS threshold (days)': LOS_THRESHOLD,
    'Class balance (% long stay)': f"{splits['y_train'].mean()*100:.1f}%",
    'First N hours used': FIRST_N_HOURS,
    'Mortality rate': f"{df_features['mortality'].mean()*100:.1f}%"
}

for key, val in summary.items():
    print(f"  {key}: {val}")

# %% Feature list for documentation
print("\n" + "=" * 60)
print("FEATURE LIST")
print("=" * 60)

# Group features by type
static_feat = [c for c in feature_columns if '_encoded' in c or c in 
               ['anchor_age', 'elixhauser_vanwalraven', 'height_inch', 'pbw_kg']]
temporal_feat = [c for c in feature_columns if c not in static_feat]

print(f"\nStatic features ({len(static_feat)}):")
for f in sorted(static_feat):
    print(f"  - {f}")

print(f"\nTemporal features ({len(temporal_feat)}):")
# Group by base feature
base_features = set()
for f in temporal_feat:
    base = f.rsplit('_', 1)[0] if any(suf in f for suf in ['_mean', '_std', '_min', '_max', '_first', '_last', '_count', '_sum', '_trend']) else f
    base_features.add(base)

for base in sorted(base_features):
    related = [f for f in temporal_feat if f.startswith(base)]
    print(f"  - {base}: {len(related)} aggregations")

# %% Next steps
print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
Data is ready! Next scripts to create:

1. 03_xgboost_classification.py
   - Train XGBoost for LOS binary classification
   - Hyperparameter tuning
   - Feature importance analysis
   
2. 04_survival_analysis.py
   - Cox proportional hazards model
   - Kaplan-Meier curves
   - Time-dependent ROC curves

3. 05_model_evaluation.py
   - Compare all models
   - Calibration plots
   - Fairness analysis

Load the data in next scripts with:
```python
import pickle
with open('../outputs/data_splits.pkl', 'rb') as f:
    splits = pickle.load(f)
X_train = splits['X_train_scaled']
y_train = splits['y_train']
```
""")

# %%
print("\n Preprocessing complete!")

# %%
