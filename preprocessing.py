import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio

# ============================================================================
# CONFIG
# ============================================================================

INPUT_DIR = "data/historical"
OUTPUT_CSV = "output/01_preprocessed_timeseries.csv"
OUTPUT_PKL = "output/01_preprocessed_timeseries.pkl"

MIN_VALID_FRACTION = 0.30  # At least 30% valid pixels in given image...

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_date(filename):
    """Extract date from filename"""
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", str(filename))
    if match:
        try:
            return datetime(*map(int, match.groups()))
        except:
            return None
    return None


def process_tiff(filepath):
    """Process one TIFF file"""
    try:
        with rasterio.open(filepath) as src:
            arr = src.read().astype(np.float32)
            
            if arr.shape[0] != 8:
                return None, f"Expected 8 bands, got {arr.shape[0]}"
            
            # Extract bands
            # Band order: B4, B8, B11, B12, SCL, NDVI, NBR, NDMI
            b4 = arr[0]
            b8 = arr[1]
            b11 = arr[2]
            b12 = arr[3]
            scl = arr[4]
            ndvi = arr[5]  # Already computed by GEE
            nbr = arr[6]
            ndmi = arr[7]
            
            # Mask bad pixels using SCL
            good = (
                (scl != 0) & (scl != 1) & (scl != 3) &
                (scl != 8) & (scl != 9) & (scl != 10) & (scl != 11)
            )
            
            ndvi[~good] = np.nan
            nbr[~good] = np.nan
            ndmi[~good] = np.nan
            
            # Mask invalid ranges
            ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan
            nbr[(nbr < -1) | (nbr > 1)] = np.nan
            ndmi[(ndmi < -1) | (ndmi > 1)] = np.nan
            
            # Check validity
            valid_mask = np.isfinite(ndvi)
            valid_fraction = valid_mask.sum() / ndvi.size
            
            if valid_fraction < MIN_VALID_FRACTION:
                return None, f"Too few valid pixels: {valid_fraction:.1%}"
            
            # Get date
            date = extract_date(filepath.name)
            if date is None:
                return None, "No date in filename"
            
            # Compute stats
            stats = {
                "date": date,
                "year": date.year,
                "month": date.month,
                "day_of_year": date.timetuple().tm_yday,
                "filename": filepath.name,
                
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
                "NDMI_min": float(np.nanmin(ndmi)),
                "NDMI_max": float(np.nanmax(ndmi)),
            }
            
            return stats, None
            
    except Exception as e:
        return None, f"Error: {str(e)}"


def process_directory(input_dir):
    """Process all TIFFs in directory"""
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {len(tiff_files)} files found")
    print(f"{'='*70}")
    
    records = []
    skipped = []
    
    for i, filepath in enumerate(tiff_files, 1):
        if i % 5 == 0:
            print(f"Processing {i}/{len(tiff_files)}...")
        
        stats, error = process_tiff(filepath)
        
        if stats:
            records.append(stats)
        else:
            skipped.append((filepath.name, error))
    
    if not records:
        print("\n No valid images processed!")
        return None
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f" Processed: {len(records)}")
    print(f" Skipped: {len(skipped)}")
    
    print(f"\n Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\n Per year:")
    for year, count in df.groupby('year').size().items():
        print(f"   {year}: {count}")
    
    print(f"\n Value ranges:")
    print(f"   NDVI: [{df['NDVI_mean'].min():.3f}, {df['NDVI_mean'].max():.3f}]")
    print(f"   NBR:  [{df['NBR_mean'].min():.3f}, {df['NBR_mean'].max():.3f}]")
    print(f"   NDMI: [{df['NDMI_mean'].min():.3f}, {df['NDMI_mean'].max():.3f}]")
    
    print(f"\n Quality:")
    print(f"   Avg valid: {df['valid_fraction'].mean()*100:.1f}%")
    print(f"   Min valid: {df['valid_fraction'].min()*100:.1f}%")
    
    if skipped:
        print(f"\n  Skipped (first 5):")
        for name, reason in skipped[:5]:
            print(f"   {name}: {reason}")
    
    return df


def debug_file(filepath):
    """Debug single file"""
    print(f"\n{'='*70}")
    print(f"DEBUG: {filepath}")
    print(f"{'='*70}")
    
    with rasterio.open(filepath) as src:
        print(f"\nBands: {src.count}")
        print(f"Size: {src.width} x {src.height}")
        print(f"CRS: {src.crs}")
        
        arr = src.read()
        for i in range(min(8, src.count)):
            band = arr[i]
            print(f"\nBand {i}:")
            print(f"  Min: {band.min():.6f}")
            print(f"  Max: {band.max():.6f}")
            print(f"  Mean: {band.mean():.6f}")
    
    print(f"\n{'='*70}")
    print("PROCESSING TEST:")
    print(f"{'='*70}")
    
    stats, error = process_tiff(Path(filepath))
    
    if stats:
        print("\n SUCCESS!")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"\n FAILED: {error}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Debug mode
    if len(sys.argv) > 1:
        debug_file(sys.argv[1])
        sys.exit()
    
    # Process all
    df = process_directory(INPUT_DIR)
    
    if df is not None:
        Path("output").mkdir(exist_ok=True)
        
        df.to_csv(OUTPUT_CSV, index=False)
        df.to_pickle(OUTPUT_PKL)
        
        print(f"\n Saved:")
        print(f"   {OUTPUT_CSV}")
        print(f"   {OUTPUT_PKL}")
        
        print(f"\n Preview:")
        print(df.head())
        
