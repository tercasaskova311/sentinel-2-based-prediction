import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


class Config:
    S2_SCALE_FACTOR = 10000
    # Valid range for normalized indices (-1 to 1)
    VALID_RANGE = (-1.0, 1.0)
    # Minimum fraction of valid pixels to accept an image
    MIN_VALID_PIXELS_FRACTION = 0.20  # 20% - reasonable for cloudy data

    # IQR multiplier for outlier detection
    IQR_MULTIPLIER = 3.0
    
    #GEE script uses mask_clouds() which already applies SCL
    # The exported data is already masked
    
    DATE_REGEX = r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})"  # Fixed: allows separators
    NDVI_BAND = 0  # First band in your export
    NBR_BAND = 1   # Second band
    NDMI_BAND = 2  # Third band

def extract_date_from_filename(fname):
    fname_str = str(fname) if isinstance(fname, Path) else fname
    match = re.search(Config.DATE_REGEX, fname_str)
    if match:
        try:
            y, m, d = map(int, match.groups())
            return datetime(y, m, d)
        except ValueError:
            # Invalid date (e.g., month=13)
            return None
    return None


# ============================================================
# ISSUE #2: Reprojection Function
# ============================================================

"""
Different satellite images may have different:
  1. Pixel sizes (10m vs 20m vs 30m)
  2. Coordinate systems (UTM Zone 33N vs WGS84)
  3. Grid alignments (pixels don't line up exactly)

- Takes source array with source transform (pixel→coordinate mapping)
- Creates destination array with target transform
- Interpolates values to fill destination grid
=> All images have same size, alignment, CRS

RESAMPLING METHODS:
- Nearest: Use nearest pixel value (fast, for discrete data)
- Bilinear: Average of 4 nearest pixels (smooth, for continuous data)
- Cubic: Average of 16 nearest pixels (smoother, slower)
"""

def reproject_to_match(src_arr, src_profile, dst_shape, dst_transform, dst_crs=None):
    """
    Reproject source array to match destination grid
    
    Args:
        src_arr: Source 2D array (H, W)
        src_profile: Source rasterio profile (has 'transform', 'crs')
        dst_shape: Target shape (H, W)
        dst_transform: Target affine transform (pixel→coordinate mapping)
        dst_crs: Target CRS (if None, uses source CRS)
    
    Returns:
        Reprojected array with dst_shape
    """
    dst_arr = np.empty(dst_shape, dtype=np.float32)
    
    reproject(
        source=src_arr,           # Input array
        destination=dst_arr,      # Output array (pre-allocated)
        src_transform=src_profile['transform'],  # How source pixels map to coordinates
        src_crs=src_profile['crs'],              # Source coordinate system
        dst_transform=dst_transform,             # How dest pixels map to coordinates  
        dst_crs=dst_crs or src_profile['crs'],   # Destination coord
        resampling=Resampling.bilinear
    )
    return dst_arr

#IOR => interquartile range
"""
- calcualte Q1 + Q3
- IOR = Q3-Q1
- def outliers as values beyond...
- residual contamination, shadow edges, water bodies...
"""
def remove_outliers(arr):
    """
    Remove statistical outliers using IQR method
    
    Rationale: Even after cloud masking, some artifacts remain
               (cloud shadows, haze, sensor noise)
    """
    # Calculate quartiles (ignoring NaN)
    q1, q3 = np.nanpercentile(arr, [25, 75])
    iqr = q3 - q1
    
    # Define bounds
    lower = q1 - Config.IQR_MULTIPLIER * iqr
    upper = q3 + Config.IQR_MULTIPLIER * iqr
    
    # Mask outliers as NaN
    arr[(arr < lower) | (arr > upper)] = np.nan
    return arr

#normalization dif index calculation..

#I want properties from -1 to 1
#so I can compare bands... so I do for example - NDVI = nir - red/ nir + red , same for nbr + ndmi...

def compute_index(numerator, denominator):
    """
    Compute normalized difference index with error handling
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Suppress warnings for 0/0 and inf
        idx = (numerator - denominator) / (numerator + denominator)
        
        # Replace infinities with NaN
        idx[np.isinf(idx)] = np.nan
        
        return idx

#===============================
#GEE - exports band => jsut read them...


def process_tiff(path, mtd_date=None):
  
    try:
        with rasterio.open(path) as src:
            # Read all bands
            arr = src.read().astype("float32")  # Shape: (bands, H, W)
            
            # DEBUG: Check what we actually have
            n_bands = arr.shape[0]
            if n_bands != 3:
                return None, f"Expected 3 bands (NDVI,NBR,NDMI), got {n_bands}"
            
            ndvi_raw = arr[Config.NDVI_BAND]
            nbr_raw = arr[Config.NBR_BAND]
            ndmi_raw = arr[Config.NDMI_BAND]
            
            # Check if data is already normalized or needs scaling
            # If max value > 10, assume it needs normalization
            max_val = np.nanmax([ndvi_raw.max(), nbr_raw.max(), ndmi_raw.max()])
            
            if max_val > 10:
                # Data is in 0-10000 range, needs normalization
                ndvi = ndvi_raw / Config.S2_SCALE_FACTOR
                nbr = nbr_raw / Config.S2_SCALE_FACTOR
                ndmi = ndmi_raw / Config.S2_SCALE_FACTOR
            else:
                # Already normalized
                ndvi = ndvi_raw
                nbr = nbr_raw
                ndmi = ndmi_raw
            
            # Mask invalid values
            # GEE exports use 0 or -9999 for nodata
            ndvi[ndvi == 0] = np.nan
            nbr[nbr == 0] = np.nan
            ndmi[ndmi == 0] = np.nan
            
            ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan
            nbr[(nbr < -1) | (nbr > 1)] = np.nan
            ndmi[(ndmi < -1) | (ndmi > 1)] = np.nan
            
            # Remove outliers 
            ndvi = remove_outliers(ndvi)
            nbr = remove_outliers(nbr)
            ndmi = remove_outliers(ndmi)
            
            # Check validity
            valid_mask = np.isfinite(ndvi)
            valid_fraction = valid_mask.sum() / ndvi.size
            
            if valid_fraction < Config.MIN_VALID_PIXELS_FRACTION:
                return None, f"Valid fraction too low: {valid_fraction:.2%}"
            
            # Determine date
            date = mtd_date or extract_date_from_filename(path.name)
            if date is None:
                return None, "Could not extract date from filename"
            
            # Compute statistics
            stats = {
                "date": date,
                "year": date.year,
                "month": date.month,
                "day_of_year": date.timetuple().tm_yday,
                "filename": path.name,
                "valid_fraction": float(valid_fraction),
                "valid_pixels": int(valid_mask.sum()),
                "total_pixels": int(ndvi.size),
                
                "NDVI_mean": float(np.nanmean(ndvi)),
                "NBR_mean": float(np.nanmean(nbr)),
                "NDMI_mean": float(np.nanmean(ndmi)),
                
                "NDVI_std": float(np.nanstd(ndvi)),
                "NBR_std": float(np.nanstd(nbr)),
                "NDMI_std": float(np.nanstd(ndmi)),
                
                "NDVI_p10": float(np.nanpercentile(ndvi, 10)),
                "NDVI_p50": float(np.nanpercentile(ndvi, 50)),
                "NDVI_p90": float(np.nanpercentile(ndvi, 90)),
                
                "NBR_p10": float(np.nanpercentile(nbr, 10)),
                "NBR_p50": float(np.nanpercentile(nbr, 50)),
                "NBR_p90": float(np.nanpercentile(nbr, 90)),
                
                "NDMI_p10": float(np.nanpercentile(ndmi, 10)),
                "NDMI_p50": float(np.nanpercentile(ndmi, 50)),
                "NDMI_p90": float(np.nanpercentile(ndmi, 90)),
                
                "NDVI_min": float(np.nanmin(ndvi)),
                "NDVI_max": float(np.nanmax(ndvi)),
                "NBR_min": float(np.nanmin(nbr)),
                "NBR_max": float(np.nanmax(nbr)),
            }
            
            return stats, None
    
    except Exception as e:
        return None, f"Error processing: {str(e)}"


def preprocess_directory(tiff_dir):    
    tiff_files = list(Path(tiff_dir).rglob("*.tif"))
    tiff_files = [
        p for p in tiff_files 
        if not any(skip in p.name.lower() for skip in ['(', 'copy', '._'])
    ]
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {len(tiff_files)} TIFF files found")
    print(f"{'='*70}")
    
    records = []
    skipped = []
    
    for i, path in enumerate(tiff_files, 1):
        if i % 10 == 0:
            print(f"Processing {i}/{len(tiff_files)}...")
        
        stats, reason = process_tiff(path)
        
        if stats is None:
            skipped.append((path.name, reason))
        else:
            records.append(stats)
    
    # Create DataFrame
    if not records:
        print("\n ERROR: No valid images processed!")
        print("\nSkipped files:")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")
        return None, skipped
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f" Successfully processed: {len(records)}")
    print(f" Skipped: {len(skipped)}")
    print(f" Date range: {df['date'].min()} to {df['date'].max()}")
    print(f" Observations per year: {df.groupby('year').size().to_dict()}")
    
    if skipped:
        print(f"\nSkipped files (first 10):")
        for name, reason in skipped[:10]:
            print(f"  - {name}: {reason}")
    
    return df, skipped

def debug_single_file(path):
    """
    Debug a single TIFF file to see what's inside
    """
    print(f"\n{'='*70}")
    print(f"DEBUGGING: {path}")
    print(f"{'='*70}")
    
    with rasterio.open(path) as src:
        arr = src.read()
        
        print(f"\nRaster Info:")
        print(f"  Bands: {src.count}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Dtype: {src.dtypes}")
        print(f"  Transform: {src.transform}")
        print(f"  Nodata: {src.nodata}")
        
        print(f"\nArray Info:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        
        for i in range(min(3, src.count)):
            band = arr[i]
            print(f"\nBand {i}:")
            print(f"  Min: {band.min()}")
            print(f"  Max: {band.max()}")
            print(f"  Mean: {band.mean():.6f}")
            print(f"  Non-zero: {(band != 0).sum()} / {band.size}")
            print(f"  Unique (first 10): {np.unique(band)[:10]}")
        
        # Try processing
        print(f"\n{'='*70}")
        print("PROCESSING TEST:")
        print(f"{'='*70}")
        
        stats, error = process_tiff(Path(path))
        
        if stats:
            print("\n SUCCESS! Statistics:")
            for key, val in stats.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.6f}")
                else:
                    print(f"  {key}: {val}")
        else:
            print(f"\n❌ FAILED: {error}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Debug mode: pass filename as argument
    if len(sys.argv) > 1:
        debug_single_file(sys.argv[1])
        sys.exit()
    
    # Normal mode: process directory
    df, skipped = preprocess_directory("data/historical")
    
    if df is not None:
        # Save results
        output_path = "output/01_preprocessed_corrected.csv"
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved: {output_path}")
        
        # Show preview
        print("\nPreview:")
        print(df.head())
        
        print("\nValue Ranges:")
        for col in ['NDVI_mean', 'NBR_mean', 'NDMI_mean']:
            print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")