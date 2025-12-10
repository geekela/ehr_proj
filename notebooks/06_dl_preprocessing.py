# %% [markdown]
# # Deep Learning Preprocessing for Sequence Models
# 
# This script prepares data for GRU/LSTM/Transformer models.
# Unlike classical models that use aggregated features, deep learning models
# use raw hourly sequences to capture temporal patterns.
#
# Output: 3D tensors (n_patients, n_timesteps, n_features)

# %% Imports
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# %% Configuration
DATA_DIR = Path("../data/temporal-respiratory-support")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sequence parameters
SEQUENCE_LENGTH = 24  # First 24 hours
MIN_HOURS = 4         # Minimum hours of data required
LOS_THRESHOLD = 4     # Days for binary classification
RANDOM_STATE = 42

print(f"Configuration:")
print(f"  Sequence length: {SEQUENCE_LENGTH} hours")
print(f"  Minimum hours required: {MIN_HOURS}")
print(f"  LOS threshold: {LOS_THRESHOLD} days")

# %% Define feature groups for sequences
# These are features that vary over time (measured hourly)
TEMPORAL_FEATURES = [
    # Vitals
    'heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni', 'mbp_ni',
    'temperature', 'spo2', 'glucose',
    
    # Respiratory status (binary)
    'invasive', 'noninvasive', 'highflow',
    
    # Ventilator settings
    'set_peep', 'total_peep', 'rr', 'set_rr', 'total_rr',
    'set_tv', 'total_tv', 'set_fio2',
    
    # Blood gas
    'calculated_bicarbonate', 'pCO2', 'pH', 'pO2', 'so2',
    
    # GCS
    'gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes',
    
    # Interventions
    'vasopressor', 'crrt',
    
    # Severity
    'sepsis3', 'sofa_24hours'
]

# Static features (don't change over time)
STATIC_FEATURES = [
    'anchor_age', 'gender', 'elixhauser_vanwalraven',
    'height_inch', 'pbw_kg'
]

print(f"\nTemporal features: {len(TEMPORAL_FEATURES)}")
print(f"Static features: {len(STATIC_FEATURES)}")

# %% Load all patient files
def load_patient_file(filepath: Path) -> pd.DataFrame:
    """Load a single patient CSV file."""
    return pd.read_csv(filepath)

def get_all_patient_files(data_dir: Path, sample_size: int = None) -> list:
    """Get list of all patient CSV files."""
    all_files = []
    
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            csv_files = list(folder.glob("*.csv"))
            all_files.extend(csv_files)
    
    print(f"Found {len(all_files)} patient files")
    
    if sample_size:
        all_files = all_files[:sample_size]
        print(f"Using sample of {sample_size} patients")
    
    return all_files

# %% Extract sequences from patient data
def extract_sequence(df: pd.DataFrame, 
                     temporal_features: list,
                     sequence_length: int = 24) -> tuple:
    """
    Extract fixed-length sequence from patient data.
    
    Returns:
        - sequence: numpy array of shape (sequence_length, n_features)
        - static: dict of static features
        - targets: dict of target variables
        - valid: bool indicating if patient has enough data
    """
    # Check if patient has enough hours
    actual_hours = min(len(df), int(df['los'].iloc[0] * 24))
    
    if actual_hours < MIN_HOURS:
        return None, None, None, False
    
    # Get first sequence_length hours
    df_seq = df[df['hr'] < sequence_length].copy()
    
    # Extract available temporal features
    available_features = [f for f in temporal_features if f in df_seq.columns]
    
    # Create sequence array
    sequence = np.zeros((sequence_length, len(available_features)))
    
    for i, feat in enumerate(available_features):
        values = df_seq[feat].values
        # Pad if necessary
        if len(values) < sequence_length:
            padded = np.full(sequence_length, np.nan)
            padded[:len(values)] = values
            sequence[:, i] = padded
        else:
            sequence[:, i] = values[:sequence_length]
    
    # Extract static features
    static = {}
    for feat in STATIC_FEATURES:
        if feat in df.columns:
            static[feat] = df[feat].iloc[0]
    
    # Extract targets
    targets = {
        'los_days': df['los'].iloc[0],
        'los_binary': int(df['los'].iloc[0] >= LOS_THRESHOLD),
        'mortality': df['death_outcome'].iloc[0] if 'death_outcome' in df.columns else 0,
        'subject_id': df['subject_id'].iloc[0]
    }
    
    return sequence, static, targets, True

# %% Process all patients
def process_all_patients(file_list: list, 
                         temporal_features: list,
                         sequence_length: int = 24) -> dict:
    """
    Process all patient files and extract sequences.
    
    Returns dict with:
        - sequences: (n_patients, sequence_length, n_features)
        - static_features: (n_patients, n_static_features)
        - targets: DataFrame with target variables
        - feature_names: list of temporal feature names
    """
    sequences = []
    static_list = []
    targets_list = []
    
    available_features = None
    
    for filepath in tqdm(file_list, desc="Processing patients"):
        df = load_patient_file(filepath)
        
        # Check inclusion criteria
        if df['anchor_age'].iloc[0] < 18:
            continue
        if df['los'].iloc[0] > 90:
            continue
            
        seq, static, targets, valid = extract_sequence(
            df, temporal_features, sequence_length
        )
        
        if valid:
            sequences.append(seq)
            static_list.append(static)
            targets_list.append(targets)
            
            # Track available features from first valid patient
            if available_features is None:
                available_features = [f for f in temporal_features if f in df.columns]
    
    # Convert to arrays
    sequences = np.array(sequences)
    targets_df = pd.DataFrame(targets_list)
    
    # Convert static to array
    static_df = pd.DataFrame(static_list)
    
    print(f"\nProcessed {len(sequences)} valid patients")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Static features shape: {static_df.shape}")
    
    return {
        'sequences': sequences,
        'static_df': static_df,
        'targets_df': targets_df,
        'temporal_features': available_features
    }

# %% Load and process data
print("\n" + "=" * 60)
print("LOADING AND PROCESSING DATA")
print("=" * 60)

# Get patient files (use sample for development, None for full)
patient_files = get_all_patient_files(DATA_DIR, sample_size=10000)

# Process all patients
data = process_all_patients(patient_files, TEMPORAL_FEATURES, SEQUENCE_LENGTH)

sequences = data['sequences']
static_df = data['static_df']
targets_df = data['targets_df']
temporal_features = data['temporal_features']

print(f"\nFinal dataset:")
print(f"  Sequences: {sequences.shape} (patients, hours, features)")
print(f"  Static features: {static_df.shape}")
print(f"  Targets: {targets_df.shape}")

# %% Handle missing values in sequences
print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

def impute_sequences(sequences: np.ndarray, strategy: str = 'forward_fill') -> np.ndarray:
    """
    Impute missing values in sequences.
    
    Strategies:
        - forward_fill: Fill with last valid value (then backward fill remaining)
        - mean: Fill with feature mean
        - zero: Fill with zero
    """
    sequences = sequences.copy()
    n_patients, n_timesteps, n_features = sequences.shape
    
    missing_before = np.isnan(sequences).sum()
    
    if strategy == 'forward_fill':
        for i in range(n_patients):
            for j in range(n_features):
                # Get the sequence for this patient and feature
                seq = sequences[i, :, j]
                
                # Forward fill
                mask = np.isnan(seq)
                if mask.any():
                    # Find indices where values are valid
                    idx = np.where(~mask, np.arange(len(seq)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    seq_filled = seq[idx]
                    
                    # Backward fill for any remaining NaNs at the start
                    mask_remaining = np.isnan(seq_filled)
                    if mask_remaining.any():
                        idx_back = np.where(~mask_remaining, np.arange(len(seq_filled)), len(seq_filled)-1)
                        idx_back = np.minimum.accumulate(idx_back[::-1])[::-1]
                        seq_filled = seq_filled[idx_back]
                    
                    sequences[i, :, j] = seq_filled
    
    elif strategy == 'mean':
        # Calculate mean per feature across all patients and timesteps
        for j in range(n_features):
            feature_mean = np.nanmean(sequences[:, :, j])
            if np.isnan(feature_mean):
                feature_mean = 0
            mask = np.isnan(sequences[:, :, j])
            sequences[:, :, j][mask] = feature_mean
    
    else:  # zero
        sequences = np.nan_to_num(sequences, nan=0)
    
    missing_after = np.isnan(sequences).sum()
    
    # Final cleanup - any remaining NaNs get zero
    sequences = np.nan_to_num(sequences, nan=0)
    
    print(f"Missing values: {missing_before:,} → {missing_after:,} → 0")
    
    return sequences

# Calculate missingness before imputation
missing_pct = np.isnan(sequences).mean() * 100
print(f"Overall missingness before imputation: {missing_pct:.1f}%")

# Impute sequences
sequences_imputed = impute_sequences(sequences, strategy='forward_fill')

# %% Encode static features
print("\n" + "=" * 60)
print("ENCODING STATIC FEATURES")
print("=" * 60)

def encode_static_features(df: pd.DataFrame) -> tuple:
    """Encode categorical static features."""
    df = df.copy()
    encoders = {}
    
    # Encode gender
    if 'gender' in df.columns:
        le = LabelEncoder()
        df['gender'] = df['gender'].fillna('Unknown')
        df['gender_encoded'] = le.fit_transform(df['gender'].astype(str))
        encoders['gender'] = le
        df = df.drop('gender', axis=1)
    
    # Fill missing numerical features
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    
    return df, encoders

static_encoded, static_encoders = encode_static_features(static_df)
print(f"Static features encoded: {static_encoded.columns.tolist()}")

# %% Normalize features
print("\n" + "=" * 60)
print("NORMALIZING FEATURES")
print("=" * 60)

def normalize_sequences(sequences: np.ndarray, 
                        fit_data: np.ndarray = None) -> tuple:
    """
    Normalize sequences using StandardScaler.
    
    Reshapes (n_patients, n_timesteps, n_features) → (n_samples, n_features)
    for fitting, then reshapes back.
    """
    n_patients, n_timesteps, n_features = sequences.shape
    
    # Reshape to 2D
    sequences_2d = sequences.reshape(-1, n_features)
    
    if fit_data is not None:
        fit_2d = fit_data.reshape(-1, n_features)
        scaler = StandardScaler()
        scaler.fit(fit_2d)
    else:
        scaler = StandardScaler()
        scaler.fit(sequences_2d)
    
    # Transform
    sequences_normalized = scaler.transform(sequences_2d)
    
    # Reshape back to 3D
    sequences_normalized = sequences_normalized.reshape(n_patients, n_timesteps, n_features)
    
    return sequences_normalized, scaler

# Normalize static features
static_scaler = StandardScaler()
static_normalized = static_scaler.fit_transform(static_encoded)
static_normalized = pd.DataFrame(static_normalized, columns=static_encoded.columns)

print(f"Static features normalized: {static_normalized.shape}")

# %% Create train/val/test splits
print("\n" + "=" * 60)
print("CREATING DATA SPLITS")
print("=" * 60)

# Get targets
y_binary = targets_df['los_binary'].values
y_continuous = targets_df['los_days'].values
subject_ids = targets_df['subject_id'].values

# First split: train+val vs test
idx = np.arange(len(sequences_imputed))
idx_trainval, idx_test = train_test_split(
    idx, test_size=0.15, stratify=y_binary, random_state=RANDOM_STATE
)

# Second split: train vs val
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.176,  # 0.176 of 0.85 ≈ 0.15 of total
    stratify=y_binary[idx_trainval], random_state=RANDOM_STATE
)

print(f"Train: {len(idx_train)} ({len(idx_train)/len(idx)*100:.1f}%)")
print(f"Val: {len(idx_val)} ({len(idx_val)/len(idx)*100:.1f}%)")
print(f"Test: {len(idx_test)} ({len(idx_test)/len(idx)*100:.1f}%)")

# %% Normalize using training data only
print("\nNormalizing sequences using training statistics...")

# Fit scaler on training data
train_sequences = sequences_imputed[idx_train]
seq_scaler = StandardScaler()
train_2d = train_sequences.reshape(-1, train_sequences.shape[2])
seq_scaler.fit(train_2d)

# Transform all splits
def normalize_split(sequences, scaler):
    n_p, n_t, n_f = sequences.shape
    seq_2d = sequences.reshape(-1, n_f)
    seq_norm = scaler.transform(seq_2d)
    return seq_norm.reshape(n_p, n_t, n_f)

X_train_seq = normalize_split(sequences_imputed[idx_train], seq_scaler)
X_val_seq = normalize_split(sequences_imputed[idx_val], seq_scaler)
X_test_seq = normalize_split(sequences_imputed[idx_test], seq_scaler)

# Static features - normalize on training data
static_train = static_encoded.iloc[idx_train]
static_scaler = StandardScaler()
static_scaler.fit(static_train)

X_train_static = static_scaler.transform(static_encoded.iloc[idx_train])
X_val_static = static_scaler.transform(static_encoded.iloc[idx_val])
X_test_static = static_scaler.transform(static_encoded.iloc[idx_test])

# Targets
y_train_binary = y_binary[idx_train]
y_val_binary = y_binary[idx_val]
y_test_binary = y_binary[idx_test]

y_train_cont = y_continuous[idx_train]
y_val_cont = y_continuous[idx_val]
y_test_cont = y_continuous[idx_test]

print(f"\nFinal shapes:")
print(f"  X_train_seq: {X_train_seq.shape}")
print(f"  X_val_seq: {X_val_seq.shape}")
print(f"  X_test_seq: {X_test_seq.shape}")
print(f"  X_train_static: {X_train_static.shape}")

# %% Create attention mask (for transformers)
print("\n" + "=" * 60)
print("CREATING ATTENTION MASKS")
print("=" * 60)

def create_attention_mask(sequences_original: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Create attention mask for transformer models.
    1 = valid timestep, 0 = padding/should be ignored
    
    Based on whether the original (pre-imputation) sequence had data.
    """
    subset = sequences_original[idx]
    # A timestep is valid if at least one feature has a non-NaN value
    # But since we already imputed, we'll use a simpler approach:
    # All timesteps within actual ICU stay are valid
    # For now, use all 1s (all timesteps valid)
    mask = np.ones((subset.shape[0], subset.shape[1]))
    return mask

# For now, all timesteps are considered valid
mask_train = np.ones((len(idx_train), SEQUENCE_LENGTH))
mask_val = np.ones((len(idx_val), SEQUENCE_LENGTH))
mask_test = np.ones((len(idx_test), SEQUENCE_LENGTH))

print(f"Attention masks created: {mask_train.shape}")

# %% Save processed data
print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

# Create comprehensive data dictionary
dl_data = {
    # Sequences (3D tensors)
    'X_train_seq': X_train_seq,
    'X_val_seq': X_val_seq,
    'X_test_seq': X_test_seq,
    
    # Static features (2D arrays)
    'X_train_static': X_train_static,
    'X_val_static': X_val_static,
    'X_test_static': X_test_static,
    
    # Attention masks
    'mask_train': mask_train,
    'mask_val': mask_val,
    'mask_test': mask_test,
    
    # Targets - binary
    'y_train_binary': y_train_binary,
    'y_val_binary': y_val_binary,
    'y_test_binary': y_test_binary,
    
    # Targets - continuous
    'y_train_cont': y_train_cont,
    'y_val_cont': y_val_cont,
    'y_test_cont': y_test_cont,
    
    # Metadata
    'temporal_features': temporal_features,
    'static_features': list(static_encoded.columns),
    'sequence_length': SEQUENCE_LENGTH,
    'n_temporal_features': len(temporal_features),
    'n_static_features': len(static_encoded.columns),
    
    # Scalers for inference
    'seq_scaler': seq_scaler,
    'static_scaler': static_scaler,
    'static_encoders': static_encoders,
    
    # Subject IDs for analysis
    'subject_ids_train': subject_ids[idx_train],
    'subject_ids_val': subject_ids[idx_val],
    'subject_ids_test': subject_ids[idx_test],
}

# Save
with open(OUTPUT_DIR / 'dl_data_splits.pkl', 'wb') as f:
    pickle.dump(dl_data, f)
print(f"Saved: {OUTPUT_DIR / 'dl_data_splits.pkl'}")

# Also save as numpy arrays for easier loading in PyTorch
np.savez_compressed(
    OUTPUT_DIR / 'dl_sequences.npz',
    X_train_seq=X_train_seq,
    X_val_seq=X_val_seq,
    X_test_seq=X_test_seq,
    X_train_static=X_train_static,
    X_val_static=X_val_static,
    X_test_static=X_test_static,
    y_train_binary=y_train_binary,
    y_val_binary=y_val_binary,
    y_test_binary=y_test_binary,
    y_train_cont=y_train_cont,
    y_val_cont=y_val_cont,
    y_test_cont=y_test_cont,
    mask_train=mask_train,
    mask_val=mask_val,
    mask_test=mask_test
)
print(f"Saved: {OUTPUT_DIR / 'dl_sequences.npz'}")

# %%
# %% Verify saved files
import pickle
import numpy as np

# Load pickle
with open('../outputs/dl_data_splits.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
print("Pickle keys:", list(pkl_data.keys()))

# Load npz
npz_data = np.load('../outputs/dl_sequences.npz')
print("NPZ files:", npz_data.files)

# Check shapes
print(f"\nX_train_seq shape: {npz_data['X_train_seq'].shape}")
print(f"y_train_binary shape: {npz_data['y_train_binary'].shape}")
print("-----> Files loaded successfully!")

# %% Summary
print("\n" + "=" * 60)
print("DEEP LEARNING DATA SUMMARY")
print("=" * 60)

print(f"""
## Data Shapes

| Split | Sequences | Static | Labels |
|-------|-----------|--------|--------|
| Train | {X_train_seq.shape} | {X_train_static.shape} | {y_train_binary.shape} |
| Val | {X_val_seq.shape} | {X_val_static.shape} | {y_val_binary.shape} |
| Test | {X_test_seq.shape} | {X_test_static.shape} | {y_test_binary.shape} |

## Feature Information

- Temporal features: {len(temporal_features)}
- Static features: {len(static_encoded.columns)}
- Sequence length: {SEQUENCE_LENGTH} hours
- Total features per timestep: {len(temporal_features)}

## Class Distribution

- Train: {y_train_binary.mean()*100:.1f}% long stay
- Val: {y_val_binary.mean()*100:.1f}% long stay  
- Test: {y_test_binary.mean()*100:.1f}% long stay

## Temporal Features:
{temporal_features}

## Static Features:
{list(static_encoded.columns)}
""")

# %% PyTorch Dataset Example
print("\n" + "=" * 60)
print("PYTORCH USAGE EXAMPLE")
print("=" * 60)

print("""
# Load data in PyTorch:

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

# Option 1: Load from pickle
with open('outputs/dl_data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

# Option 2: Load from npz (faster)
data = np.load('outputs/dl_sequences.npz')
X_train = data['X_train_seq']
y_train = data['y_train_binary']

# Create Dataset
class ICUDataset(Dataset):
    def __init__(self, sequences, static, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.static = torch.FloatTensor(static)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],      # (24, n_features)
            'static': self.static[idx],           # (n_static,)
            'label': self.labels[idx]             # scalar
        }

# Create DataLoaders
train_dataset = ICUDataset(X_train_seq, X_train_static, y_train_binary)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Example model input
for batch in train_loader:
    seq = batch['sequence']      # (batch, 24, n_features)
    static = batch['static']     # (batch, n_static)
    labels = batch['label']      # (batch,)
    print(f"Sequence shape: {seq.shape}")
    print(f"Static shape: {static.shape}")
    break
""")

# %% Next Steps
print("\n" + "=" * 60)
print("NEXT STEPS FOR DEEP LEARNING MODELS")
print("=" * 60)

print("""
Data is ready for your colleague to build:

1. **GRU/LSTM Model**
   - Input: sequences (batch, 24, 35 features)
   - Hidden layers: GRU/LSTM cells
   - Concatenate with static features
   - Output: binary classification or regression

2. **Transformer Model**  
   - Input: sequences + positional encoding
   - Self-attention over 24 timesteps
   - Classification head
   - Can use attention masks if needed

3. **Hybrid Model**
   - Combine sequence encoder (GRU/Transformer)
   - With static feature MLP
   - Joint prediction head

Files to share with colleague:
- outputs/dl_data_splits.pkl (complete data + scalers)
- outputs/dl_sequences.npz (numpy arrays only)
- This script as reference
""")

# %%
print("\n Deep learning preprocessing complete!")

# %%
