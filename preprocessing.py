"""
Sentinel-2 Preprocessing - Improved Version
Extracts time series, handles messy filenames, quality filtering
Saves outputs in multiple formats for ML pipeline
"""
import os
import re
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from config import (
    DataPaths, PreprocessingConfig, VisualizationConfig, 
    ReportingConfig, validate_configuration
)

# ============================================================================
# FILENAME PARSING
# ============================================================================

def extract_date_from_filename(fname):
    """
    Robustly extract date from various filename formats:
    - sumava_2020-10-25.tif
    - sumava_2021_04_17.tif
    - sumava_2021_04_17(1).tif
    - regionA_sumava_2022_07_02.tif
    """
    for pattern in PreprocessingConfig.DATE_PATTERNS:
        match = re.search(pattern, fname)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3:
                    year, month, day = map(int, groups)
                    return datetime(year, month, day)
            except ValueError:
                continue
    return None


def should_skip_file(filename):
    """Check if file should be skipped based on patterns"""
    for pattern in PreprocessingConfig.SKIP_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    return False

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_raster(path):
    """
    Load raster and perform quality checks
    Returns: (ndvi, nbr, ndmi, quality_metrics)
    """
    try:
        with rasterio.open(path) as src:
            # Read all bands
            arr = src.read().astype('float32')
            
            # Normalize to 0-1 range
            arr /= PreprocessingConfig.S2_SCALE_FACTOR
            
            # Extract indices
            ndvi = arr[PreprocessingConfig.NDVI_BAND]
            nbr = arr[PreprocessingConfig.NBR_BAND]
            ndmi = arr[PreprocessingConfig.NDMI_BAND]
            
            # Quality metrics
            metrics = {
                'valid_ndvi_fraction': np.isfinite(ndvi).sum() / ndvi.size,
                'valid_nbr_fraction': np.isfinite(nbr).sum() / nbr.size,
                'valid_ndmi_fraction': np.isfinite(ndmi).sum() / ndmi.size,
                'ndvi_range': (np.nanmin(ndvi), np.nanmax(ndvi)),
                'nbr_range': (np.nanmin(nbr), np.nanmax(nbr)),
                'ndmi_range': (np.nanmin(ndmi), np.nanmax(ndmi))
            }
            
            return ndvi, nbr, ndmi, metrics
            
    except Exception as e:
        print(f"  ⚠️  Error reading {path}: {e}")
        return None, None, None, None


def is_valid_observation(metrics):
    """Check if observation meets quality criteria"""
    if metrics is None:
        return False
    
    # Check sufficient valid pixels
    min_fraction = PreprocessingConfig.MIN_VALID_PIXELS_FRACTION
    if (metrics['valid_ndvi_fraction'] < min_fraction or
        metrics['valid_nbr_fraction'] < min_fraction or
        metrics['valid_ndmi_fraction'] < min_fraction):
        return False
    
    # Check value ranges
    valid_min, valid_max = PreprocessingConfig.VALID_RANGE
    for key in ['ndvi_range', 'nbr_range', 'ndmi_range']:
        min_val, max_val = metrics[key]
        if min_val < valid_min - 0.5 or max_val > valid_max + 0.5:
            return False
    
    return True

# ============================================================================
# PREPROCESSING WORKFLOW
# ============================================================================

def preprocess_sentinel_data():
    """
    Main preprocessing workflow:
    1. Scan directory for TIFFs
    2. Extract dates from filenames
    3. Load rasters and compute statistics
    4. Quality filtering
    5. Save results
    """
    print("\n" + "="*70)
    print("SENTINEL-2 PREPROCESSING")
    print("="*70)
    
    # Validate configuration
    if not validate_configuration():
        raise RuntimeError("Configuration validation failed!")
    
    historical_dir = DataPaths.HISTORICAL_DIR
    print(f"\n📂 Scanning directory: {historical_dir}")
    
    # Collect all TIFF files
    tif_files = []
    for root, dirs, files in os.walk(historical_dir):
        for file in files:
            if file.lower().endswith('.tif') and not should_skip_file(file):
                tif_files.append(Path(root) / file)
    
    print(f"   Found {len(tif_files)} TIFF files")
    
    # Process each file
    records = []
    skipped = []
    
    print(f"\n📊 Processing files...")
    for i, tif_path in enumerate(tif_files, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(tif_files)}")
        
        # Extract date
        date = extract_date_from_filename(tif_path.name)
        if date is None:
            skipped.append((tif_path.name, "No date found"))
            continue
        
        # Load and validate
        ndvi, nbr, ndmi, metrics = load_and_validate_raster(tif_path)
        
        if not is_valid_observation(metrics):
            skipped.append((tif_path.name, "Failed quality checks"))
            continue
        
        # Compute statistics
        record = {
            'date': date,
            'year': date.year,
            'month': date.month,
            'day_of_year': date.timetuple().tm_yday,
            'filename': tif_path.name,
            'filepath': str(tif_path),
            
            # Mean values
            'NDVI_mean': float(np.nanmean(ndvi)),
            'NBR_mean': float(np.nanmean(nbr)),
            'NDMI_mean': float(np.nanmean(ndmi)),
            
            # Standard deviations (spatial heterogeneity)
            'NDVI_std': float(np.nanstd(ndvi)),
            'NBR_std': float(np.nanstd(nbr)),
            'NDMI_std': float(np.nanstd(ndmi)),
            
            # Percentiles (distribution shape)
            'NDVI_p10': float(np.nanpercentile(ndvi, 10)),
            'NDVI_p25': float(np.nanpercentile(ndvi, 25)),
            'NDVI_p50': float(np.nanpercentile(ndvi, 50)),
            'NDVI_p75': float(np.nanpercentile(ndvi, 75)),
            'NDVI_p90': float(np.nanpercentile(ndvi, 90)),
            
            'NBR_p10': float(np.nanpercentile(nbr, 10)),
            'NBR_p25': float(np.nanpercentile(nbr, 25)),
            'NBR_p50': float(np.nanpercentile(nbr, 50)),
            'NBR_p75': float(np.nanpercentile(nbr, 75)),
            'NBR_p90': float(np.nanpercentile(nbr, 90)),
            
            'NDMI_p10': float(np.nanpercentile(ndmi, 10)),
            'NDMI_p25': float(np.nanpercentile(ndmi, 25)),
            'NDMI_p50': float(np.nanpercentile(ndmi, 50)),
            'NDMI_p75': float(np.nanpercentile(ndmi, 75)),
            'NDMI_p90': float(np.nanpercentile(ndmi, 90)),
            
            # Quality metrics
            'valid_fraction': float(np.mean([
                metrics['valid_ndvi_fraction'],
                metrics['valid_nbr_fraction'],
                metrics['valid_ndmi_fraction']
            ]))
        }
        
        records.append(record)
    
    # Create DataFrame
    if not records:
        raise RuntimeError("No valid observations processed!")
    
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n✅ Successfully processed {len(df)} observations")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} files:")
        for fname, reason in skipped[:5]:
            print(f"   - {fname}: {reason}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped) - 5} more")
    
    return df, skipped

# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def generate_quality_report(df):
    """Generate comprehensive quality report"""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("PREPROCESSING QUALITY REPORT")
    report_lines.append("="*70)
    
    # Temporal coverage
    report_lines.append("\n📅 TEMPORAL COVERAGE:")
    report_lines.append(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    report_lines.append(f"   Total span: {(df['date'].max() - df['date'].min()).days} days")
    report_lines.append(f"   Observations: {len(df)}")
    
    # Observations per year
    report_lines.append("\n📊 OBSERVATIONS PER YEAR:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        report_lines.append(f"   {year}: {count}")
    
    # Average time gaps
    df_sorted = df.sort_values('date')
    time_gaps = df_sorted['date'].diff().dt.days
    report_lines.append("\n⏱️  TIME GAPS:")
    report_lines.append(f"   Mean: {time_gaps.mean():.1f} days")
    report_lines.append(f"   Median: {time_gaps.median():.1f} days")
    report_lines.append(f"   Max: {time_gaps.max():.0f} days")
    
    # Value ranges
    report_lines.append("\n📈 VALUE RANGES:")
    for col in ['NDVI_mean', 'NBR_mean', 'NDMI_mean']:
        report_lines.append(f"   {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    # Data quality
    report_lines.append("\n✅ DATA QUALITY:")
    report_lines.append(f"   Avg valid fraction: {df['valid_fraction'].mean()*100:.1f}%")
    report_lines.append(f"   Missing values: {df.isnull().sum().sum()}")
    
    report = "\n".join(report_lines)
    print(report)
    
    # Save report
    if ReportingConfig.SAVE_INTERMEDIATE_RESULTS:
        with open(ReportingConfig.PREPROCESSING_REPORT, 'w') as f:
            f.write(report)
        print(f"\n💾 Report saved: {ReportingConfig.PREPROCESSING_REPORT}")
    
    return report

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_time_series(df):
    """Plot vegetation indices time series"""
    fig, axes = plt.subplots(3, 1, figsize=VisualizationConfig.MEDIUM_PLOT, sharex=True)
    
    # NDVI
    axes[0].plot(df['date'], df['NDVI_mean'], 'o-', 
                color=VisualizationConfig.COLORS['info'], alpha=0.6)
    axes[0].fill_between(df['date'], 
                         df['NDVI_mean'] - df['NDVI_std'],
                         df['NDVI_mean'] + df['NDVI_std'],
                         alpha=0.2, color=VisualizationConfig.COLORS['info'])
    axes[0].set_ylabel('NDVI')
    axes[0].set_title('Vegetation Indices Time Series - Šumava National Park')
    axes[0].grid(alpha=0.3)
    
    # NBR
    axes[1].plot(df['date'], df['NBR_mean'], 'o-',
                color=VisualizationConfig.COLORS['normal'], alpha=0.6)
    axes[1].fill_between(df['date'],
                         df['NBR_mean'] - df['NBR_std'],
                         df['NBR_mean'] + df['NBR_std'],
                         alpha=0.2, color=VisualizationConfig.COLORS['normal'])
    axes[1].set_ylabel('NBR')
    axes[1].grid(alpha=0.3)
    
    # NDMI
    axes[2].plot(df['date'], df['NDMI_mean'], 'o-',
                color=VisualizationConfig.COLORS['warning'], alpha=0.6)
    axes[2].fill_between(df['date'],
                         df['NDMI_mean'] - df['NDMI_std'],
                         df['NDMI_mean'] + df['NDMI_std'],
                         alpha=0.2, color=VisualizationConfig.COLORS['warning'])
    axes[2].set_ylabel('NDMI')
    axes[2].set_xlabel('Date')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(DataPaths.PREPROCESSING_PLOT, 
                dpi=VisualizationConfig.FIGURE_DPI)
    print(f"📊 Plot saved: {DataPaths.PREPROCESSING_PLOT}")
    plt.show()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_preprocessed_data(df):
    """Save in multiple formats for downstream use"""
    print("\n💾 Saving preprocessed data...")
    
    # CSV (human-readable, universally compatible)
    df.to_csv(DataPaths.PREPROCESSED_CSV, index=False)
    print(f"   ✓ CSV: {DataPaths.PREPROCESSED_CSV}")
    
    # Pickle (fast loading, preserves dtypes)
    df.to_pickle(DataPaths.PREPROCESSED_PKL)
    print(f"   ✓ Pickle: {DataPaths.PREPROCESSED_PKL}")
    
    # Metadata file
    metadata = {
        'n_observations': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'years': sorted(df['year'].unique().tolist()),
        'columns': df.columns.tolist(),
        'preprocessing_date': datetime.now().isoformat()
    }
    
    metadata_path = DataPaths.OUTPUT_DIR / "01_preprocessing_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✓ Metadata: {metadata_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete preprocessing workflow"""
    print("\n" + "="*70)
    print("🛰️  SENTINEL-2 PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Process data
    df, skipped = preprocess_sentinel_data()
    
    # Step 2: Quality report
    generate_quality_report(df)
    
    # Step 3: Visualize
    plot_time_series(df)
    
    # Step 4: Save outputs
    save_preprocessed_data(df)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"📊 Processed: {len(df)} observations")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n🎯 Next step: Run ml_prep.py for labeling and feature engineering")
    
    return df


if __name__ == "__main__":
    df = main()