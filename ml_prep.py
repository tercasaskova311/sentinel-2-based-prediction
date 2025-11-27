# ============================================================================
# ML PREPARATION WORKFLOW - Deforestation Detection in Šumava
# ============================================================================
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ============================================================================
# STEP 1: IMPROVED LABELING STRATEGY
# ============================================================================

def create_training_labels(df, gfc_loss_patch, historical_dir):
    """
    Create more robust labels by combining multiple signals:
    1. GFC loss year (ground truth)
    2. NBR drops adjusted for seasonality
    3. Spatial statistics (not just means)
    """
    df = df.copy()
    
    # Initialize labels
    df['label'] = 0  # 0 = no disturbance, 1 = disturbance
    df['label_confidence'] = 0.0  # How confident we are (0-1)
    df['label_source'] = 'none'
    
    # ---------------------------------
    # A) GFC Ground Truth (High Confidence)
    # ---------------------------------
    print("Step 1: Labeling from GFC ground truth...")
    for i, row in df.iterrows():
        year = row['date'].year
        gfc_code = year - 2000
        
        if 1 <= gfc_code <= 24:  # GFC covers 2001-2024
            # Calculate fraction of AOI that was lost this year
            fraction_lost = np.nanmean(gfc_loss_patch == gfc_code)
            
            # If substantial loss occurred, label as disturbed
            if fraction_lost > 0.01:  # At least 1% of AOI lost
                df.loc[i, 'label'] = 1
                df.loc[i, 'label_confidence'] = min(fraction_lost * 10, 1.0)
                df.loc[i, 'label_source'] = 'gfc'
    
    gfc_labeled = df['label'].sum()
    print(f"  → {gfc_labeled} observations labeled from GFC")
    
    # ---------------------------------
    # B) Seasonal-Adjusted NBR Drops
    # ---------------------------------
    print("Step 2: Detecting NBR anomalies...")
    df = detect_nbr_anomalies(df, historical_dir)
    
    # ---------------------------------
    # C) Spatial Pattern Analysis
    # ---------------------------------
    print("Step 3: Computing spatial features...")
    df = add_spatial_features(df, historical_dir)
    
    return df


def detect_nbr_anomalies(df, historical_dir, window_size=5):
    """
    Detect NBR drops that are anomalous compared to seasonal baseline
    """
    df = df.copy()
    
    # Group by month to establish seasonal baseline
    df['month'] = df['date'].dt.month
    monthly_baseline = df.groupby('month')['NBR_mean'].agg(['mean', 'std'])
    
    for i in range(window_size, len(df)):
        current = df.iloc[i]
        
        # Compare to recent history (not just previous observation)
        recent_window = df.iloc[i-window_size:i]
        baseline_nbr = recent_window['NBR_mean'].median()
        baseline_std = recent_window['NBR_mean'].std()
        
        # Expected NBR for this month
        expected_nbr = monthly_baseline.loc[current['month'], 'mean']
        expected_std = monthly_baseline.loc[current['month'], 'std']
        
        # Detect anomalous drop
        nbr_drop = baseline_nbr - current['NBR_mean']
        
        # Only flag if:
        # 1. Drop is significant (> 0.15)
        # 2. Drop exceeds seasonal variation (> 2 std devs)
        # 3. Not already labeled by GFC
        if (nbr_drop > 0.15 and 
            nbr_drop > 2 * expected_std and
            df.loc[df.index[i], 'label'] == 0):
            
            df.loc[df.index[i], 'label'] = 1
            df.loc[df.index[i], 'label_confidence'] = min(nbr_drop / 0.3, 1.0)
            df.loc[df.index[i], 'label_source'] = 'nbr_anomaly'
    
    anomaly_labeled = (df['label_source'] == 'nbr_anomaly').sum()
    print(f"  → {anomaly_labeled} additional anomalies detected")
    
    return df


def add_spatial_features(df, historical_dir):
    """
    Add spatial statistics beyond just mean values
    This captures heterogeneity and spatial patterns
    """
    df = df.copy()
    
    # New columns for spatial features
    spatial_cols = ['NDVI_std', 'NBR_std', 'NDMI_std',
                    'NDVI_p10', 'NDVI_p90',
                    'NBR_p10', 'NBR_p90']
    
    for col in spatial_cols:
        df[col] = np.nan
    
    print(f"  Computing spatial statistics for {len(df)} images...")
    
    for i, row in df.iterrows():
        # Find the corresponding TIFF file
        date_str = row['date'].strftime('%Y-%m-%d')
        
        # Try different filename patterns
        possible_names = [
            f"sumava_{date_str}.tif",
            f"sumava_{row['date'].strftime('%Y_%m_%d')}.tif"
        ]
        
        tif_path = None
        for name in possible_names:
            path = os.path.join(historical_dir, name)
            if os.path.exists(path):
                tif_path = path
                break
        
        if tif_path is None:
            continue
        
        try:
            with rasterio.open(tif_path) as src:
                arr = src.read().astype('float32') / 10000.0
                
                ndvi = arr[0]
                nbr = arr[1]
                ndmi = arr[2]
                
                # Compute spatial statistics
                df.loc[i, 'NDVI_std'] = np.nanstd(ndvi)
                df.loc[i, 'NBR_std'] = np.nanstd(nbr)
                df.loc[i, 'NDMI_std'] = np.nanstd(ndmi)
                
                df.loc[i, 'NDVI_p10'] = np.nanpercentile(ndvi, 10)
                df.loc[i, 'NDVI_p90'] = np.nanpercentile(ndvi, 90)
                df.loc[i, 'NBR_p10'] = np.nanpercentile(nbr, 10)
                df.loc[i, 'NBR_p90'] = np.nanpercentile(nbr, 90)
                
        except Exception as e:
            print(f"    Warning: Could not process {tif_path}: {e}")
            continue
    
    return df


# ============================================================================
# STEP 2: CREATE TEMPORAL FEATURES
# ============================================================================

def create_temporal_features(df):
    """
    Add temporal context - changes over time windows
    """
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    # Rolling window features (3, 5, 10 observations back)
    for window in [3, 5, 10]:
        for col in ['NDVI_mean', 'NBR_mean', 'NDMI_mean']:
            # Rolling mean
            df[f'{col}_roll{window}'] = df[col].rolling(window, min_periods=1).mean()
            
            # Change from rolling baseline
            df[f'{col}_delta{window}'] = df[col] - df[f'{col}_roll{window}']
    
    # Time gaps between observations
    df['days_since_prev'] = df['date'].diff().dt.days
    df['days_since_prev'] = df['days_since_prev'].fillna(0)
    
    return df


# ============================================================================
# STEP 3: DATA QUALITY CHECKS
# ============================================================================

def quality_check(df):
    """
    Check data quality before ML training
    """
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    # 1. Class balance
    print("\n1. CLASS BALANCE:")
    label_counts = df['label'].value_counts()
    print(f"   No disturbance: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Disturbance:    {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    if label_counts.get(1, 0) < 10:
        print("   ⚠️  WARNING: Very few positive examples! Consider:")
        print("      - Lowering NBR drop threshold")
        print("      - Including more years of data")
        print("      - Using oversampling techniques")
    
    # 2. Missing values
    print("\n2. MISSING VALUES:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("   ✓ No missing values")
    
    # 3. Feature distributions
    print("\n3. FEATURE RANGES:")
    numeric_cols = ['NDVI_mean', 'NBR_mean', 'NDMI_mean']
    for col in numeric_cols:
        if col in df.columns:
            print(f"   {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    # 4. Temporal coverage
    print("\n4. TEMPORAL COVERAGE:")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total span: {(df['date'].max() - df['date'].min()).days} days")
    print(f"   Observations: {len(df)}")
    print(f"   Avg gap: {df['date'].diff().mean().days:.1f} days")
    
    return df


# ============================================================================
# STEP 4: TRAIN/TEST SPLIT (TEMPORAL)
# ============================================================================

def create_train_test_split(df, test_year=2024):
    """
    Split data temporally - never use future data for training!
    """
    # IMPORTANT: Use temporal split, not random split
    # Random split leaks information from future to past
    
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] >= test_year].copy()
    
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    print(f"Training:   {len(train_df)} obs ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Testing:    {len(test_df)} obs ({test_df['date'].min()} to {test_df['date'].max()})")
    print(f"\nTrain labels: 0={train_df['label'].value_counts().get(0,0)}, 1={train_df['label'].value_counts().get(1,0)}")
    print(f"Test labels:  0={test_df['label'].value_counts().get(0,0)}, 1={test_df['label'].value_counts().get(1,0)}")
    
    return train_df, test_df


# ============================================================================
# STEP 5: FEATURE ENGINEERING & SELECTION
# ============================================================================

def prepare_ml_features(df):
    """
    Select and engineer features for ML
    """
    # Feature groups
    base_features = ['NDVI_mean', 'NBR_mean', 'NDMI_mean']
    
    spatial_features = ['NDVI_std', 'NBR_std', 'NDMI_std',
                       'NDVI_p10', 'NDVI_p90', 'NBR_p10', 'NBR_p90']
    
    temporal_features = ['day_of_year', 'days_since_start', 'days_since_prev']
    
    # Rolling features
    rolling_features = [col for col in df.columns if 'roll' in col or 'delta' in col]
    
    # Combine all features
    all_features = base_features + spatial_features + temporal_features + rolling_features
    
    # Filter to only existing columns
    feature_cols = [col for col in all_features if col in df.columns]
    
    print(f"\nUsing {len(feature_cols)} features for ML:")
    print(f"  - Base indices: {len(base_features)}")
    print(f"  - Spatial stats: {len([f for f in spatial_features if f in df.columns])}")
    print(f"  - Temporal: {len([f for f in temporal_features if f in df.columns])}")
    print(f"  - Rolling/delta: {len([f for f in rolling_features if f in df.columns])}")
    
    return feature_cols


# ============================================================================
# STEP 6: VISUALIZATION & EDA
# ============================================================================

def visualize_labels(df):
    """
    Visualize the labeled dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Time series with labels
    ax = axes[0, 0]
    disturbed = df[df['label'] == 1]
    normal = df[df['label'] == 0]
    
    ax.scatter(normal['date'], normal['NBR_mean'], c='green', s=20, alpha=0.6, label='Normal')
    ax.scatter(disturbed['date'], disturbed['NBR_mean'], c='red', s=50, marker='x', label='Disturbance')
    ax.set_xlabel('Date')
    ax.set_ylabel('NBR Mean')
    ax.set_title('NBR Time Series with Labels')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Label distribution by year
    ax = axes[0, 1]
    label_by_year = df.groupby(['year', 'label']).size().unstack(fill_value=0)
    label_by_year.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title('Labels by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend(['Normal', 'Disturbance'])
    
    # 3. Label confidence distribution
    ax = axes[1, 0]
    df[df['label'] == 1]['label_confidence'].hist(bins=20, ax=ax, color='orange')
    ax.set_title('Confidence Distribution (Disturbances Only)')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    
    # 4. Label sources
    ax = axes[1, 1]
    source_counts = df[df['label'] == 1]['label_source'].value_counts()
    ax.bar(source_counts.index, source_counts.values, color=['red', 'orange', 'yellow'])
    ax.set_title('Disturbance Label Sources')
    ax.set_xlabel('Source')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('label_analysis.png', dpi=150)
    print("\n📊 Saved: label_analysis.png")
    plt.show()


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main_workflow(df, gfc_loss_patch, historical_dir):
    """
    Complete ML preparation workflow
    """
    print("\n" + "="*70)
    print("ML PREPARATION WORKFLOW")
    print("="*70)
    
    # Step 1: Create better labels
    df = create_training_labels(df, gfc_loss_patch, historical_dir)
    
    # Step 2: Add temporal features
    df = create_temporal_features(df)
    
    # Step 3: Quality checks
    df = quality_check(df)
    
    # Step 4: Visualize labels
    visualize_labels(df)
    
    # Step 5: Train/test split
    train_df, test_df = create_train_test_split(df, test_year=2024)
    
    # Step 6: Prepare features
    feature_cols = prepare_ml_features(df)
    
    # Save processed data
    df.to_csv('ml_ready_dataset.csv', index=False)
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    with open('feature_columns.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print("\n" + "="*70)
    print("✅ ML PREPARATION COMPLETE")
    print("="*70)
    print("Files saved:")
    print("  - ml_ready_dataset.csv (full dataset)")
    print("  - train_data.csv")
    print("  - test_data.csv")
    print("  - feature_columns.txt")
    print("  - label_analysis.png")
    
    return df, train_df, test_df, feature_cols


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('preprocessed_data.csv')  # From your preprocessing.py
    df['date'] = pd.to_datetime(df['date'])
    
    # Load GFC
    with rasterio.open('data/gfc_lossyear.tif') as src:
        gfc_loss_patch = src.read(1)
    
    # Run workflow
    df, train_df, test_df, features = main_workflow(
        df, 
        gfc_loss_patch, 
        historical_dir='data/historical'
    )
    
    print("\n🚀 Ready for ML training!")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Features: {len(features)}")