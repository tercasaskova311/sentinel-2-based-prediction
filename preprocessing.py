"""
Sentinel-2 Preprocessing - Enhanced with Robust Null Handling
Handles masked/nodata values properly and provides detailed diagnostics
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
    """Robustly extract date from various filename formats"""
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
# ENHANCED DATA LOADING WITH NULL HANDLING
# ============================================================================

def clean_band_data(band, band_name="band"):
    """
    Clean a single band by:
    1. Masking nodata values
    2. Masking extreme outliers
    3. Masking zeros if they're suspicious
    """
    band = band.astype('float32')
    
    # Mask common nodata values BEFORE normalization
    for nodata in PreprocessingConfig.NODATA_VALUES:
        band[band == nodata] = np.nan
    
    # Normalize
    band = band / PreprocessingConfig.S2_SCALE_FACTOR
    
    # Mask extreme values (likely errors)
    band[np.abs(band) > PreprocessingConfig.EXTREME_VALUE_THRESHOLD] = np.nan
    
    # Check for suspicious zeros (if >80% are zeros, likely mask issue)
    if PreprocessingConfig.CHECK_FOR_ZEROS:
        zero_fraction = (band == 0).sum() / band.size
        if zero_fraction > PreprocessingConfig.MAX_ZERO_FRACTION:
            print(f"    ⚠️  {band_name}: {zero_fraction*100:.1f}% zeros - masking them")
            band[band == 0] = np.nan
    
    return band


def load_and_validate_raster(path):
    """
    Load raster with enhanced null handling
    Returns: (ndvi, nbr, ndmi, quality_metrics)
    """
    try:
        with rasterio.open(path) as src:
            # Read all bands
            arr = src.read()
            
            # Get mask (True where valid)
            mask = src.read_masks()
            
            # Clean each band
            ndvi_raw = arr[PreprocessingConfig.NDVI_BAND].copy()
            nbr_raw = arr[PreprocessingConfig.NBR_BAND].copy()
            ndmi_raw = arr[PreprocessingConfig.NDMI_BAND].copy()
            
            # Apply mask from raster
            if mask is not None and mask.size > 0:
                band_mask = mask[PreprocessingConfig.NDVI_BAND]
                ndvi_raw[band_mask == 0] = np.nan
                nbr_raw[band_mask == 0] = np.nan
                ndmi_raw[band_mask == 0] = np.nan
            
            # Clean each band
            ndvi = clean_band_data(ndvi_raw, "NDVI")
            nbr = clean_band_data(nbr_raw, "NBR")
            ndmi = clean_band_data(ndmi_raw, "NDMI")
            
            # Calculate valid pixel counts
            valid_ndvi = np.isfinite(ndvi)
            valid_nbr = np.isfinite(nbr)
            valid_ndmi = np.isfinite(ndmi)
            
            # Combined valid mask (all three indices must be valid)
            valid_all = valid_ndvi & valid_nbr & valid_ndmi
            
            # Quality metrics
            metrics = {
                'total_pixels': ndvi.size,
                'valid_ndvi_count': valid_ndvi.sum(),
                'valid_nbr_count': valid_nbr.sum(),
                'valid_ndmi_count': valid_ndmi.sum(),
                'valid_all_count': valid_all.sum(),
                'valid_ndvi_fraction': valid_ndvi.sum() / ndvi.size,
                'valid_nbr_fraction': valid_nbr.sum() / nbr.size,
                'valid_ndmi_fraction': valid_ndmi.sum() / ndmi.size,
                'valid_all_fraction': valid_all.sum() / ndvi.size,
                'ndvi_range': (np.nanmin(ndvi), np.nanmax(ndvi)),
                'nbr_range': (np.nanmin(nbr), np.nanmax(nbr)),
                'ndmi_range': (np.nanmin(ndmi), np.nanmax(ndmi)),
                'ndvi_has_data': np.isfinite(ndvi).any(),
                'nbr_has_data': np.isfinite(nbr).any(),
                'ndmi_has_data': np.isfinite(ndmi).any()
            }
            
            return ndvi, nbr, ndmi, metrics
            
    except Exception as e:
        print(f"  ⚠️  Error reading {path.name}: {e}")
        return None, None, None, None


def is_valid_observation(metrics):
    """Enhanced validation with better diagnostics"""
    if metrics is None:
        return False, "No metrics"
    
    # Check if any data exists at all
    if not (metrics['ndvi_has_data'] and metrics['nbr_has_data'] and metrics['ndmi_has_data']):
        return False, "No valid data in one or more bands"
    
    # Check sufficient valid pixels
    min_fraction = PreprocessingConfig.MIN_VALID_PIXELS_FRACTION
    if metrics['valid_all_fraction'] < min_fraction:
        return False, f"Valid pixels {metrics['valid_all_fraction']*100:.1f}% < {min_fraction*100:.0f}%"
    
    # Check value ranges
    valid_min, valid_max = PreprocessingConfig.VALID_RANGE
    for key, name in [('ndvi_range', 'NDVI'), ('nbr_range', 'NBR'), ('ndmi_range', 'NDMI')]:
        min_val, max_val = metrics[key]
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return False, f"{name} has no finite values"
        if min_val < valid_min or max_val > valid_max:
            return False, f"{name} out of range [{min_val:.2f}, {max_val:.2f}]"
    
    return True, "OK"

# ============================================================================
# PREPROCESSING WORKFLOW
# ============================================================================

def preprocess_sentinel_data():
    """
    Main preprocessing workflow with detailed diagnostics
    """
    print("\n" + "="*70)
    print("SENTINEL-2 PREPROCESSING - ENHANCED NULL HANDLING")
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
    
    if len(tif_files) == 0:
        raise RuntimeError(f"No TIFF files found in {historical_dir}")
    
    # Process each file with detailed tracking
    records = []
    skipped = []
    diagnostics = {
        'no_date': 0,
        'no_data': 0,
        'insufficient_pixels': 0,
        'out_of_range': 0,
        'other_error': 0,
        'success': 0
    }
    
    print(f"\n📊 Processing files...")
    for i, tif_path in enumerate(tif_files, 1):
        if i % 5 == 0 or i == 1:
            print(f"   Progress: {i}/{len(tif_files)} ({i/len(tif_files)*100:.0f}%)")
        
        # Extract date
        date = extract_date_from_filename(tif_path.name)
        if date is None:
            skipped.append((tif_path.name, "No date found"))
            diagnostics['no_date'] += 1
            continue
        
        # Load and validate
        ndvi, nbr, ndmi, metrics = load_and_validate_raster(tif_path)
        
        if metrics is None:
            skipped.append((tif_path.name, "Load error"))
            diagnostics['other_error'] += 1
            continue
        
        # Check validity
        is_valid, reason = is_valid_observation(metrics)
        
        if not is_valid:
            skipped.append((tif_path.name, reason))
            
            # Categorize failure
            if 'No valid data' in reason:
                diagnostics['no_data'] += 1
            elif 'Valid pixels' in reason:
                diagnostics['insufficient_pixels'] += 1
            elif 'out of range' in reason:
                diagnostics['out_of_range'] += 1
            else:
                diagnostics['other_error'] += 1
            continue
        
        # Compute statistics (using nanmean to ignore NaN values)
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
            
            # Standard deviations
            'NDVI_std': float(np.nanstd(ndvi)),
            'NBR_std': float(np.nanstd(nbr)),
            'NDMI_std': float(np.nanstd(ndmi)),
            
            # Percentiles
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
            'valid_fraction': float(metrics['valid_all_fraction']),
            'total_pixels': int(metrics['total_pixels']),
            'valid_pixels': int(metrics['valid_all_count'])
        }
        
        records.append(record)
        diagnostics['success'] += 1
    
    # Print diagnostics
    print("\n" + "="*70)
    print("PROCESSING DIAGNOSTICS")
    print("="*70)
    print(f"✅ Successful:            {diagnostics['success']}")
    print(f"❌ Skipped - No date:     {diagnostics['no_date']}")
    print(f"❌ Skipped - No data:     {diagnostics['no_data']}")
    print(f"❌ Skipped - Few pixels:  {diagnostics['insufficient_pixels']}")
    print(f"❌ Skipped - Out of range: {diagnostics['out_of_range']}")
    print(f"❌ Skipped - Other:       {diagnostics['other_error']}")
    print(f"📊 Total processed:       {len(tif_files)}")
    
    # Create DataFrame
    if not records:
        print("\n❌ ERROR: No valid observations processed!")
        print("\nShowing first 10 skipped files:")
        for fname, reason in skipped[:10]:
            print(f"   {fname}: {reason}")
        raise RuntimeError("No valid data - check your input files and config settings")
    
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n✅ Successfully processed {len(df)} observations")
    
    if skipped and len(skipped) <= 20:
        print(f"\n⚠️  Skipped files ({len(skipped)}):")
        for fname, reason in skipped:
            print(f"   - {fname}: {reason}")
    elif skipped:
        print(f"\n⚠️  Skipped {len(skipped)} files (showing first 10):")
        for fname, reason in skipped[:10]:
            print(f"   - {fname}: {reason}")
    
    return df, skipped, diagnostics

# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def generate_quality_report(df, diagnostics):
    """Generate comprehensive quality report"""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("PREPROCESSING QUALITY REPORT")
    report_lines.append("="*70)
    
    # Processing summary
    report_lines.append("\n📊 PROCESSING SUMMARY:")
    report_lines.append(f"   Successfully processed: {diagnostics['success']}")
    report_lines.append(f"   Skipped (no date):      {diagnostics['no_date']}")
    report_lines.append(f"   Skipped (no data):      {diagnostics['no_data']}")
    report_lines.append(f"   Skipped (few pixels):   {diagnostics['insufficient_pixels']}")
    report_lines.append(f"   Skipped (out of range): {diagnostics['out_of_range']}")
    
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
    
    # Value ranges and statistics
    report_lines.append("\n📈 VALUE RANGES:")
    for col in ['NDVI_mean', 'NBR_mean', 'NDMI_mean']:
        report_lines.append(f"   {col}: [{df[col].min():.3f}, {df[col].max():.3f}] (μ={df[col].mean():.3f})")
    
    # Data quality
    report_lines.append("\n✅ DATA QUALITY:")
    report_lines.append(f"   Avg valid fraction: {df['valid_fraction'].mean()*100:.1f}%")
    report_lines.append(f"   Min valid fraction: {df['valid_fraction'].min()*100:.1f}%")
    report_lines.append(f"   Avg valid pixels: {df['valid_pixels'].mean():.0f}/{df['total_pixels'].iloc[0]}")
    report_lines.append(f"   Missing values: {df.isnull().sum().sum()}")
    
    report = "\n".join(report_lines)
    print("\n" + report)
    
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
    """Plot vegetation indices time series with quality indicators"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # NDVI
    axes[0].plot(df['date'], df['NDVI_mean'], 'o-', 
                color=VisualizationConfig.COLORS['info'], alpha=0.6, markersize=4)
    axes[0].fill_between(df['date'], 
                         df['NDVI_mean'] - df['NDVI_std'],
                         df['NDVI_mean'] + df['NDVI_std'],
                         alpha=0.2, color=VisualizationConfig.COLORS['info'])
    axes[0].set_ylabel('NDVI')
    axes[0].set_title('Vegetation Indices Time Series - Šumava National Park')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # NBR
    axes[1].plot(df['date'], df['NBR_mean'], 'o-',
                color=VisualizationConfig.COLORS['normal'], alpha=0.6, markersize=4)
    axes[1].fill_between(df['date'],
                         df['NBR_mean'] - df['NBR_std'],
                         df['NBR_mean'] + df['NBR_std'],
                         alpha=0.2, color=VisualizationConfig.COLORS['normal'])
    axes[1].set_ylabel('NBR')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # NDMI
    axes[2].plot(df['date'], df['NDMI_mean'], 'o-',
                color=VisualizationConfig.COLORS['warning'], alpha=0.6, markersize=4)
    axes[2].fill_between(df['date'],
                         df['NDMI_mean'] - df['NDMI_std'],
                         df['NDMI_mean'] + df['NDMI_std'],
                         alpha=0.2, color=VisualizationConfig.COLORS['warning'])
    axes[2].set_ylabel('NDMI')
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Valid pixel fraction
    axes[3].plot(df['date'], df['valid_fraction']*100, 'o-',
                color='purple', alpha=0.6, markersize=4)
    axes[3].axhline(y=PreprocessingConfig.MIN_VALID_PIXELS_FRACTION*100, 
                   color='red', linestyle='--', label='Min threshold')
    axes[3].set_ylabel('Valid Pixels (%)')
    axes[3].set_xlabel('Date')
    axes[3].grid(alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig(DataPaths.PREPROCESSING_PLOT, 
                dpi=VisualizationConfig.FIGURE_DPI, bbox_inches='tight')
    print(f"📊 Plot saved: {DataPaths.PREPROCESSING_PLOT}")
    plt.show()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_preprocessed_data(df):
    """Save in multiple formats"""
    print("\n💾 Saving preprocessed data...")
    
    # CSV
    df.to_csv(DataPaths.PREPROCESSED_CSV, index=False)
    print(f"   ✓ CSV: {DataPaths.PREPROCESSED_CSV}")
    
    # Pickle
    df.to_pickle(DataPaths.PREPROCESSED_PKL)
    print(f"   ✓ Pickle: {DataPaths.PREPROCESSED_PKL}")
    
    # Metadata
    metadata = {
        'n_observations': len(df),
        'date_range': (str(df['date'].min()), str(df['date'].max())),
        'years': sorted(df['year'].unique().tolist()),
        'columns': df.columns.tolist(),
        'preprocessing_date': datetime.now().isoformat(),
        'config_settings': {
            'min_valid_fraction': PreprocessingConfig.MIN_VALID_PIXELS_FRACTION,
            'valid_range': PreprocessingConfig.VALID_RANGE,
            'scale_factor': PreprocessingConfig.S2_SCALE_FACTOR
        }
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
    print("🛰️  SENTINEL-2 PREPROCESSING PIPELINE - ENHANCED")
    print("="*70)
    
    # Step 1: Process data
    df, skipped, diagnostics = preprocess_sentinel_data()
    
    # Step 2: Quality report
    generate_quality_report(df, diagnostics)
    
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