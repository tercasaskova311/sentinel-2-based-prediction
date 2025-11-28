import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime

from config import (
    DataPaths, LabelingConfig, FeatureConfig, MLConfig,
    VisualizationConfig, ReportingConfig
)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data():
    """Load preprocessed time series data"""
    print("\n" + "="*70)
    print("LOADING PREPROCESSED DATA")
    print("="*70)
    
    # Try pickle first (faster), fallback to CSV
    if DataPaths.PREPROCESSED_PKL.exists():
        print(f"📂 Loading from: {DataPaths.PREPROCESSED_PKL}")
        df = pd.read_pickle(DataPaths.PREPROCESSED_PKL)
    elif DataPaths.PREPROCESSED_CSV.exists():
        print(f"📂 Loading from: {DataPaths.PREPROCESSED_CSV}")
        df = pd.read_csv(DataPaths.PREPROCESSED_CSV)
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise FileNotFoundError(
            "Preprocessed data not found! Run preprocessing.py first."
        )
    
    print(f"✅ Loaded {len(df)} observations")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_gfc_data():
    """Load Global Forest Change loss year layer"""
    print("\n📂 Loading GFC loss year data...")
    
    if not DataPaths.GFC_LOSSYEAR.exists():
        raise FileNotFoundError(f"GFC data not found: {DataPaths.GFC_LOSSYEAR}")
    
    with rasterio.open(DataPaths.GFC_LOSSYEAR) as src:
        gfc_loss = src.read(1)
    
    # Count pixels by year
    unique_years = np.unique(gfc_loss[gfc_loss > 0])
    print(f"✅ GFC data loaded")
    print(f"   Loss years present: {len(unique_years)}")
    print(f"   Years: {sorted(unique_years + 2000)}")
    
    return gfc_loss

# ============================================================================
# LABELING STRATEGY
# ============================================================================

def create_multi_signal_labels(df, gfc_loss):
    """
    Create robust labels by combining three signals:
    1. GFC ground truth (high confidence)
    2. NBR anomaly detection (medium confidence)
    3. Spatial pattern analysis (supporting evidence)
    """
    print("\n" + "="*70)
    print("MULTI-SIGNAL LABELING STRATEGY")
    print("="*70)
    
    df = df.copy()
    
    # Initialize label columns
    df['label'] = 0
    df['label_confidence'] = 0.0
    df['label_source'] = 'none'
    df['gfc_loss_fraction'] = 0.0
    df['nbr_anomaly_score'] = 0.0
    df['spatial_anomaly_score'] = 0.0
    
    # Signal 1: GFC Ground Truth
    print("\n1️⃣  GFC GROUND TRUTH LABELING")
    df = label_from_gfc(df, gfc_loss)
    
    # Signal 2: NBR Anomaly Detection
    print("\n2️⃣  NBR ANOMALY DETECTION")
    df = detect_nbr_anomalies(df)
    
    # Signal 3: Spatial Pattern Analysis
    print("\n3️⃣  SPATIAL PATTERN ANALYSIS")
    df = analyze_spatial_patterns(df)
    
    # Combine signals for final labels
    print("\n4️⃣  COMBINING SIGNALS")
    df = combine_label_signals(df)
    
    return df


def label_from_gfc(df, gfc_loss):
    """Label observations using GFC ground truth"""
    labeled_count = 0
    
    for i, row in df.iterrows():
        year = row['year']
        gfc_code = year - 2000
        
        # GFC covers 2001-2024 (codes 1-24)
        if 1 <= gfc_code <= 24:
            # Calculate fraction of AOI that was lost this year
            fraction_lost = np.mean(gfc_loss == gfc_code)
            df.loc[i, 'gfc_loss_fraction'] = fraction_lost
            
            # Label if substantial loss occurred
            if fraction_lost > LabelingConfig.GFC_MIN_LOSS_FRACTION:
                df.loc[i, 'label'] = 1
                df.loc[i, 'label_confidence'] = min(
                    fraction_lost * LabelingConfig.GFC_CONFIDENCE_SCALE, 
                    1.0
                )
                df.loc[i, 'label_source'] = 'gfc'
                labeled_count += 1
    
    print(f"   ✓ Labeled {labeled_count} observations from GFC")
    print(f"   ✓ Mean loss fraction: {df['gfc_loss_fraction'].mean()*100:.3f}%")
    
    return df


def detect_nbr_anomalies(df):
    """
    Detect NBR drops that are anomalous compared to:
    - Recent temporal baseline
    - Seasonal expectations
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Establish seasonal baseline (monthly)
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month')['NBR_mean'].agg(['mean', 'std'])
    
    anomaly_count = 0
    window = LabelingConfig.NBR_ROLLING_WINDOW
    
    for i in range(window, len(df)):
        current = df.iloc[i]
        
        # Recent baseline (exclude current observation)
        recent_window = df.iloc[i-window:i]
        baseline_nbr = recent_window['NBR_mean'].median()
        
        # Expected NBR for this month
        month = current['month']
        expected_nbr = monthly_stats.loc[month, 'mean']
        seasonal_std = monthly_stats.loc[month, 'std']
        
        # Compute NBR drop
        nbr_drop = baseline_nbr - current['NBR_mean']
        
        # Anomaly score: how many std devs below expected?
        if seasonal_std > 0:
            anomaly_score = (expected_nbr - current['NBR_mean']) / seasonal_std
        else:
            anomaly_score = 0.0
        
        df.loc[df.index[i], 'nbr_anomaly_score'] = max(0, anomaly_score)
        
        # Label as anomaly if:
        # 1. Drop is significant
        # 2. Drop exceeds seasonal variation
        # 3. Not already labeled by GFC (to avoid overwriting high-confidence labels)
        if (nbr_drop > LabelingConfig.NBR_DROP_THRESHOLD and
            anomaly_score > LabelingConfig.NBR_ANOMALY_STD_THRESHOLD and
            df.loc[df.index[i], 'label'] == 0):
            
            df.loc[df.index[i], 'label'] = 1
            confidence = min(nbr_drop / 0.3, 1.0)  # Scale to 0-1
            df.loc[df.index[i], 'label_confidence'] = confidence * 0.7  # Lower than GFC
            df.loc[df.index[i], 'label_source'] = 'nbr_anomaly'
            anomaly_count += 1
    
    print(f"   ✓ Detected {anomaly_count} NBR anomalies")
    print(f"   ✓ Mean anomaly score: {df['nbr_anomaly_score'].mean():.2f}")
    
    return df


def analyze_spatial_patterns(df):
    """
    Analyze spatial heterogeneity patterns
    High spatial variation + low values → disturbance
    """
    spatial_anomalies = 0
    
    for i, row in df.iterrows():
        # High standard deviation indicates heterogeneous area
        # Low percentiles indicate damaged pixels present
        
        nbr_std = row['NBR_std']
        nbr_p10 = row['NBR_p10']
        nbr_mean = row['NBR_mean']
        
        # Spatial anomaly conditions:
        # 1. High spatial variation (heterogeneous)
        # 2. Low values in bottom 10% (damaged areas)
        # 3. Mean is also somewhat reduced
        
        spatial_score = 0.0
        
        if nbr_std > LabelingConfig.SPATIAL_STD_THRESHOLD:
            spatial_score += 0.5
        
        if nbr_p10 < 0.3:  # Very low values present
            spatial_score += 0.3
        
        if nbr_mean < 0.5:  # Overall reduction
            spatial_score += 0.2
        
        df.loc[i, 'spatial_anomaly_score'] = spatial_score
        
        # Only flag as disturbance if multiple signals agree
        # Don't override GFC or NBR labels
        if (spatial_score > 0.7 and
            row['nbr_anomaly_score'] > 1.0 and
            df.loc[i, 'label'] == 0):
            
            df.loc[i, 'label'] = 1
            df.loc[i, 'label_confidence'] = 0.5  # Lower confidence
            df.loc[i, 'label_source'] = 'spatial_pattern'
            spatial_anomalies += 1
    
    print(f"   ✓ Detected {spatial_anomalies} spatial anomalies")
    
    return df


def combine_label_signals(df):
    """
    Final label refinement: boost confidence when multiple signals agree
    """
    # Count how many signals detected each observation
    df['signal_count'] = (
        (df['gfc_loss_fraction'] > 0).astype(int) +
        (df['nbr_anomaly_score'] > 1.0).astype(int) +
        (df['spatial_anomaly_score'] > 0.5).astype(int)
    )
    
    # Boost confidence for multi-signal detections
    multi_signal = df['signal_count'] >= 2
    df.loc[multi_signal & (df['label'] == 1), 'label_confidence'] *= 1.2
    df['label_confidence'] = df['label_confidence'].clip(0, 1)
    
    # Summary
    label_summary = df.groupby(['label', 'label_source']).size()
    print(f"\n📊 LABELING SUMMARY:")
    print(label_summary.to_string())
    
    total_disturbed = (df['label'] == 1).sum()
    print(f"\n✓ Total disturbances: {total_disturbed} ({total_disturbed/len(df)*100:.1f}%)")
    print(f"✓ High confidence (>0.7): {(df['label_confidence'] > 0.7).sum()}")
    print(f"✓ Medium confidence (0.4-0.7): {((df['label_confidence'] >= 0.4) & (df['label_confidence'] <= 0.7)).sum()}")
    
    return df

# ============================================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================================

def create_temporal_features(df):
    """Add temporal context features"""
    print("\n" + "="*70)
    print("TEMPORAL FEATURE ENGINEERING")
    print("="*70)
    
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time-based features
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    df['days_since_prev'] = df['date'].diff().dt.days.fillna(0)
    
    # Rolling window features
    feature_count = 0
    for window in FeatureConfig.ROLLING_WINDOWS:
        for col in FeatureConfig.BASE_FEATURES:
            # Rolling mean
            roll_col = f'{col}_roll{window}'
            df[roll_col] = df[col].rolling(window, min_periods=1).mean()
            
            # Change from rolling baseline
            delta_col = f'{col}_delta{window}'
            df[delta_col] = df[col] - df[roll_col]
            
            feature_count += 2
    
    print(f"✓ Created {feature_count} temporal features")
    print(f"   - Rolling windows: {FeatureConfig.ROLLING_WINDOWS}")
    print(f"   - Base indices: {len(FeatureConfig.BASE_FEATURES)}")
    
    return df

# ============================================================================
# QUALITY CHECKS
# ============================================================================

def quality_check_labels(df):
    """Validate labeled dataset"""
    print("\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)
    
    # 1. Class balance
    print("\n1️⃣  CLASS BALANCE:")
    label_counts = df['label'].value_counts()
    for label in [0, 1]:
        count = label_counts.get(label, 0)
        pct = count / len(df) * 100
        label_name = "No disturbance" if label == 0 else "Disturbance"
        print(f"   {label_name}: {count} ({pct:.1f}%)")
    
    if label_counts.get(1, 0) < 10:
        print("   ⚠️  WARNING: Very few positive examples!")
        print("      Consider adjusting thresholds in config.py")
    
    # 2. Missing values
    print("\n2️⃣  MISSING VALUES:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("   ✓ No missing values")
    
    # 3. Temporal coverage by label
    print("\n3️⃣  TEMPORAL DISTRIBUTION:")
    label_by_year = df.groupby(['year', 'label']).size().unstack(fill_value=0)
    print(label_by_year.to_string())
    
    # 4. Feature ranges
    print("\n4️⃣  FEATURE RANGES:")
    for col in ['NDVI_mean', 'NBR_mean', 'NDMI_mean']:
        print(f"   {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    return True

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def create_temporal_split(df):
    """
    Temporal split to prevent data leakage
    Never use future data to predict the past!
    """
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT (TEMPORAL)")
    print("="*70)
    
    test_year = MLConfig.TEST_YEAR
    
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] >= test_year].copy()
    
    print(f"\n📅 Split year: {test_year}")
    print(f"\n🏋️  TRAINING SET:")
    print(f"   Observations: {len(train_df)}")
    print(f"   Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    train_labels = train_df['label'].value_counts()
    print(f"   Labels: Normal={train_labels.get(0,0)}, Disturbed={train_labels.get(1,0)}")
    
    print(f"\n🧪 TEST SET:")
    print(f"   Observations: {len(test_df)}")
    print(f"   Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    test_labels = test_df['label'].value_counts()
    print(f"   Labels: Normal={test_labels.get(0,0)}, Disturbed={test_labels.get(1,0)}")
    
    # Warning if test set has no positives
    if test_labels.get(1, 0) == 0:
        print("\n⚠️  WARNING: No disturbances in test set!")
        print("   Consider using earlier test year or expanding dataset")
    
    return train_df, test_df

# ============================================================================
# FEATURE SELECTION
# ============================================================================

def prepare_feature_columns(df):
    """Select final features for ML"""
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    
    # Collect all feature groups
    base_features = FeatureConfig.BASE_FEATURES
    spatial_features = [f for f in FeatureConfig.SPATIAL_FEATURES if f in df.columns]
    temporal_features = [f for f in FeatureConfig.TEMPORAL_FEATURES if f in df.columns]
    rolling_features = [c for c in df.columns if 'roll' in c or 'delta' in c]
    
    # Anomaly scores can also be features
    signal_features = ['nbr_anomaly_score', 'spatial_anomaly_score', 'signal_count']
    
    all_features = (
        base_features + 
        spatial_features + 
        temporal_features + 
        rolling_features +
        signal_features
    )
    
    # Filter to existing columns
    feature_cols = [c for c in all_features if c in df.columns]
    
    print(f"\n📊 Selected {len(feature_cols)} features:")
    print(f"   - Base indices: {len(base_features)}")
    print(f"   - Spatial stats: {len(spatial_features)}")
    print(f"   - Temporal: {len(temporal_features)}")
    print(f"   - Rolling/delta: {len(rolling_features)}")
    print(f"   - Signal scores: {len(signal_features)}")
    
    return feature_cols

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_labels(df):
    """Create comprehensive label visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. NBR time series with labels
    ax = axes[0, 0]
    normal = df[df['label'] == 0]
    disturbed = df[df['label'] == 1]
    
    ax.scatter(normal['date'], normal['NBR_mean'], 
              c=VisualizationConfig.COLORS['normal'], 
              s=20, alpha=0.5, label='Normal')
    ax.scatter(disturbed['date'], disturbed['NBR_mean'], 
              c=VisualizationConfig.COLORS['disturbed'], 
              s=80, marker='x', linewidths=2, label='Disturbance')
    ax.set_xlabel('Date')
    ax.set_ylabel('NBR Mean')
    ax.set_title('NBR Time Series with Disturbance Labels')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Labels by year and source
    ax = axes[0, 1]
    label_pivot = df[df['label'] == 1].groupby(['year', 'label_source']).size().unstack(fill_value=0)
    label_pivot.plot(kind='bar', ax=ax, stacked=True)
    ax.set_title('Disturbances by Year and Source')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend(title='Source')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # 3. Confidence distribution
    ax = axes[1, 0]
    disturbed = df[df['label'] == 1]
    ax.hist(disturbed['label_confidence'], bins=20, 
           color=VisualizationConfig.COLORS['warning'], edgecolor='black')
    ax.axvline(LabelingConfig.HIGH_CONFIDENCE_THRESHOLD, 
              color='red', linestyle='--', label='High confidence')
    ax.axvline(LabelingConfig.MEDIUM_CONFIDENCE_THRESHOLD,
              color='orange', linestyle='--', label='Medium confidence')
    ax.set_title('Label Confidence Distribution')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.legend()
    
    # 4. Signal agreement
    ax = axes[1, 1]
    signal_counts = df[df['label'] == 1]['signal_count'].value_counts().sort_index()
    ax.bar(signal_counts.index, signal_counts.values, 
          color=VisualizationConfig.COLORS['info'], edgecolor='black')
    ax.set_title('Number of Agreeing Signals (Disturbances)')
    ax.set_xlabel('Number of Signals')
    ax.set_ylabel('Count')
    ax.set_xticks([1, 2, 3])
    
    plt.tight_layout()
    plt.savefig(DataPaths.LABEL_ANALYSIS_PLOT, dpi=VisualizationConfig.FIGURE_DPI)
    print(f"\n📊 Saved: {DataPaths.LABEL_ANALYSIS_PLOT}")
    plt.show()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_ml_ready_data(df, train_df, test_df, feature_cols):
    """Save all ML-ready datasets"""
    print("\n💾 Saving ML-ready data...")
    
    # Full dataset
    df.to_csv(DataPaths.ML_DATASET, index=False)
    df.to_pickle(DataPaths.ML_DATASET_PKL)
    print(f"   ✓ Full dataset: {DataPaths.ML_DATASET}")
    
    # Train/test splits
    train_df.to_csv(DataPaths.TRAIN_DATA, index=False)
    test_df.to_csv(DataPaths.TEST_DATA, index=False)
    print(f"   ✓ Train data: {DataPaths.TRAIN_DATA}")
    print(f"   ✓ Test data: {DataPaths.TEST_DATA}")
    
    # Feature columns
    with open(DataPaths.FEATURE_COLS, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"   ✓ Feature list: {DataPaths.FEATURE_COLS}")
    
    # Metadata
    metadata = {
        'n_total': len(df),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'test_year': MLConfig.TEST_YEAR,
        'label_distribution': df['label'].value_counts().to_dict(),
        'created_date': datetime.now().isoformat()
    }
    
    metadata_path = DataPaths.OUTPUT_DIR / "02_ml_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✓ Metadata: {metadata_path}")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete ML preparation workflow"""
    print("\n" + "="*70)
    print("🤖 ML PREPARATION PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    df = load_preprocessed_data()
    gfc_loss = load_gfc_data()
    
    # Step 2: Multi-signal labeling
    df = create_multi_signal_labels(df, gfc_loss)
    
    # Step 3: Temporal feature engineering
    df = create_temporal_features(df)
    
    # Step 4: Quality checks
    quality_check_labels(df)
    
    # Step 5: Visualize labels
    visualize_labels(df)
    
    # Step 6: Train/test split
    train_df, test_df = create_temporal_split(df)
    
    # Step 7: Feature selection
    feature_cols = prepare_feature_columns(df)
    
    # Step 8: Save everything
    save_ml_ready_data(df, train_df, test_df, feature_cols)
    
    print("\n" + "="*70)
    print("✅ ML PREPARATION COMPLETE")
    print("="*70)
    print(f"📊 Total observations: {len(df)}")
    print(f"🏋️  Training samples: {len(train_df)}")
    print(f"🧪 Test samples: {len(test_df)}")
    print(f"🎯 Features: {len(feature_cols)}")
    print(f"🎨 Disturbances labeled: {(df['label']==1).sum()}")
    print(f"\n🚀 Next step: Train ML models!")
    
    return df, train_df, test_df, feature_cols


if __name__ == "__main__":
    df, train_df, test_df, features = main()