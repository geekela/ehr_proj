import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# %% 1. Configuration & Feature Groups
DATA_DIR = Path("../data/temporal-respiratory-support")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FIRST_N_HOURS = 24  
MIN_ICU_HOURS = 4   
LOS_THRESHOLD = 4   
BATCH_SIZE = 1000  # Adjust based on RAM; 1000 is a safe standard
RANDOM_STATE = 42

STATIC_FEATURES = [
    'gender', 'anchor_age', 'race', 'insurance', 'language',
    'marital_status', 'first_careunit', 'elixhauser_vanwalraven',
    'height_inch', 'pbw_kg'
]

TEMPORAL_FEATURES = {
    'vitals': ['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni', 'mbp_ni', 'temperature', 'spo2', 'glucose'],
    'respiratory_status': ['invasive', 'noninvasive', 'highflow'],
    'ventilator_settings': ['set_peep', 'total_peep', 'rr', 'set_rr', 'total_rr', 'set_tv', 'total_tv', 'set_fio2'],
    'blood_gas': ['calculated_bicarbonate', 'pCO2', 'pH', 'pO2', 'so2'],
    'gcs': ['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes'],
    'interventions': ['vasopressor', 'crrt'],
    'severity': ['sepsis3', 'sofa_24hours']
}

ALL_TEMPORAL = [col for group in TEMPORAL_FEATURES.values() for col in group]

# %% 2. Memory & Logic Helpers
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces memory footprint by optimizing numeric types."""
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    return df

def apply_inclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Your original logic: Age >= 18 and Stay >= 4 hours."""
    patients = df.groupby('subject_id').first().reset_index()
    criteria = (
        (patients['los'] * 24 >= MIN_ICU_HOURS) & 
        (patients['anchor_age'] >= 18) & 
        (patients['los'].notna()) & (patients['los'] <= 90)
    )
    included_ids = patients[criteria]['subject_id'].unique()
    return df[df['subject_id'].isin(included_ids)].copy()

def aggregate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes the first 24h into stats and trends."""
    df_first = df[df['hr'] < FIRST_N_HOURS].copy()
    binary_feats = ['invasive', 'noninvasive', 'highflow', 'vasopressor', 'crrt', 'sepsis3']
    
    aggregations = {}
    for feat in ALL_TEMPORAL:
        if feat in df_first.columns:
            if feat in binary_feats:
                aggregations[feat] = ['sum', 'max', 'mean']
            else:
                aggregations[feat] = ['mean', 'std', 'min', 'max', 'first', 'last']
    
    agg_df = df_first.groupby('subject_id').agg(aggregations)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Add Trend Features
    for feat in ['heart_rate', 'sbp', 'mbp', 'spo2', 'temperature']:
        if f'{feat}_first' in agg_df.columns:
            agg_df[f'{feat}_trend'] = agg_df[f'{feat}_last'] - agg_df[f'{feat}_first']
    return agg_df.reset_index()

# %% 3. The Batch Engine (REPLACED PART)
def run_batch_pipeline(data_dir: Path):
    all_files = []
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            all_files.extend(list(folder.glob("*.csv")))
    
    print(f"Total files to process: {len(all_files)}")
    processed_summaries = []
    
    for i in tqdm(range(0, len(all_files), BATCH_SIZE), desc="Processing Batches"):
        # 1. Load Batch
        batch_files = all_files[i : i + BATCH_SIZE]
        batch_df = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
        batch_df = downcast_df(batch_df)
        
        # 2. Filter & Extract (using your logic)
        batch_df = apply_inclusion_criteria(batch_df)
        if batch_df.empty: continue
            
        static = batch_df.groupby('subject_id')[['subject_id', 'stay_id', 'los', 'discharge_outcome', 'death_outcome'] + STATIC_FEATURES].first().reset_index()
        temporal = aggregate_temporal_features(batch_df)
        
        # 3. Combine Batch results and save to list
        merged = static.merge(temporal, on='subject_id', how='inner')
        processed_summaries.append(merged)
        
        # 4. Clear RAM (Crucial for 50k+ patients)
        del batch_df, static, temporal
        gc.collect() 

    return pd.concat(processed_summaries, ignore_index=True)

# %% 4. Run Pipeline and Final Formatting
df_final = run_batch_pipeline(DATA_DIR)

# --- Create Targets ---
df_final['los_days'] = df_final['los']
df_final['los_hours'] = df_final['los'] * 24
df_final['los_binary'] = (df_final['los'] >= LOS_THRESHOLD).astype(int)
df_final['mortality'] = df_final['death_outcome'].astype(int)

# --- Encode Categorical ---
label_encoders = {}
for col in ['gender', 'race', 'insurance', 'language', 'marital_status', 'first_careunit']:
    df_final[col] = df_final[col].fillna('Unknown')
    le = LabelEncoder()
    df_final[f'{col}_encoded'] = le.fit_transform(df_final[col].astype(str))
    label_encoders[col] = le

# --- Impute Missing ---
# Numerical only
exclude = ['subject_id', 'stay_id', 'los', 'los_days', 'los_hours', 'los_binary', 'mortality', 
           'discharge_outcome', 'death_outcome'] + STATIC_FEATURES
num_cols = [c for c in df_final.columns if c not in exclude and df_final[c].dtype != 'object']

imputation_values = {}
for col in num_cols:
    val = df_final[col].median()
    df_final[col] = df_final[col].fillna(val)
    imputation_values[col] = val

# %% 5. Splitting and Scaling
feature_cols = [c for c in df_final.columns if c not in exclude and '_encoded' not in c] + [f'{c}_encoded' for c in label_encoders.keys()]
# Filter feature_cols to ensure only columns present in df_final are used
feature_cols = [c for c in feature_cols if c in df_final.columns]

X = df_final[feature_cols]
y = df_final['los_binary']
meta = df_final[['subject_id', 'los_days', 'los_hours', 'mortality']]

# Split
X_trainval, X_test, y_trainval, y_test, m_trainval, m_test = train_test_split(X, y, meta, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(X_trainval, y_trainval, m_trainval, test_size=0.176, stratify=y_trainval, random_state=RANDOM_STATE)

# Scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

# %% 6. Save
splits = {
    'X_train_scaled': X_train_scaled, 'y_train': y_train, 'meta_train': m_train,
    'X_val_scaled': X_val_scaled, 'y_val': y_val, 'meta_val': m_val,
    'X_test_scaled': X_test_scaled, 'y_test': y_test, 'meta_test': m_test,
    'feature_columns': feature_cols, 'scaler': scaler
}

with open(OUTPUT_DIR / 'data_splits50k.pkl', 'wb') as f:
    pickle.dump(splits, f)

print(f"Processed {len(df_final)} patients. Memory optimization complete.")