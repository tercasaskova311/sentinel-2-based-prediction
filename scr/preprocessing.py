#process it thought google drive app... GEE => drive export => local drive sync
#in preprocessing we will: extract date, mask clouds, calculate indices averages, create composites every 10 days
#plus parallel processing option...
#spatial detection = pixel-level composites - median of valid pixels over 10-day windows

import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed

GDRIVE_ROOT = Path.home() / "Library/CloudStorage"
gdrive_folders = list(GDRIVE_ROOT.glob("GoogleDrive-*"))

if gdrive_folders:
    MY_DRIVE = gdrive_folders[0] / "My Drive"
    DRIVE_INPUT_DIR = MY_DRIVE / "sumava_full"
else:
    raise FileNotFoundError("Did not find Google Drive folder")

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "01_preprocessed_timeseries.csv"
OUTPUT_PKL = OUTPUT_DIR / "01_preprocessed_timeseries.pkl"

MIN_VALID_FRACTION = 0.30  # Skip images with <30% valid pixels (too cloudy)
COMPOSITE_DAYS = 10         # Combine images every 10 days to reduce cloud noise
USE_PARALLEL = True         # Use parallel processing (faster but more memory)
MAX_WORKERS = 2             # Reduce if crashes

# ============================================================================
def extract_date(filename): 
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", str(filename))
    if match:
        try:
            return datetime(*map(int, match.groups()))
        except:
            return None
    return None

def process_tiff(filepath):
    """
    Process one TIFF file means:
    1. Load 4 bands (B8, NDVI, NBR, NDMI) from GEE export
    2. Mask invalid values (clouds, out-of-range) = check if enough valid pixels (>30%)
    4. Calculate mean indices
    """
    
    try:
        with rasterio.open(filepath) as src:
            arr = src.read().astype(np.float32)
            
            if arr.shape[0] != 4:
                return None, f"Expected 4 bands, got {arr.shape[0]}"
            
            b8, ndvi, nbr, ndmi = arr
            
            #APPLY MASKS - important => cloud masking on given indices = we mask by valid index values
            ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan
            nbr[(nbr < -1) | (nbr > 1)] = np.nan
            ndmi[(ndmi < -1) | (ndmi > 1)] = np.nan
            
            # Check validity - need at least 30% valid pixels
            valid_mask = np.isfinite(ndvi)
            valid_fraction = valid_mask.sum() / ndvi.size
            
            if valid_fraction < MIN_VALID_FRACTION:
                return None, f"Too few valid pixels: {valid_fraction:.1%}"
            
            # Get date from filename
            date = extract_date(filepath.name)
            if date is None:
                return None, "No date in filename"
            
            # Compute basic statistics for each image - just to get an overview
            stats = {
                "date": date,
                "year": date.year,
                "month": date.month,
                "filename": filepath.name,
                "valid_fraction": float(valid_fraction),
                
                # Core indices for forest health
                "NDVI_mean": float(np.nanmean(ndvi)),  # Vegetation greenness
                "NBR_mean": float(np.nanmean(nbr)),    # Bark beetle indicator
                "NDMI_mean": float(np.nanmean(ndmi)),  # Moisture stress
            }
            
            return stats, None
            
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# COMPOSITING - ok this is sth I found out recently => could be useful since we have many cloudy images
# the idea is to combine multiple images from a short period (rn is 10 days) into one "cleaner" image
# this reduces noise and fills gaps from clouds - gives us a better chance to see real changes on the ground
# ============================================================================

def create_pixel_composite(tiff_files, output_path):
#Create pixel-level composite from multiple TIFF files
#For each pixel: take median of valid (non-cloud) values

    arrays = []
    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            arr = src.read().astype(np.float32)
            # Mask clouds (invalid values)
            arr[(arr < -1) | (arr > 1)] = np.nan
            arrays.append(arr)
    
    # Stack along new axis
    stacked = np.stack(arrays, axis=0)  # Shape: (n_images, n_bands, height, width)
    
    # Take mean across images for each pixel
    composite = np.nanmean(stacked, axis=0)  # Shape: (n_bands, height, width)
    
    # Save composite image
    with rasterio.open(tiff_files[0]) as src:
        profile = src.profile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(composite)
    
    return output_path


def process_directory_parallel(input_dir, max_workers=2):
    
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    
    print(f"\nPREPROCESSING: {len(tiff_files)} files found")
    
    records = []
    skipped = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tiff, f): f for f in tiff_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            # Progress update every 10 files
            if i % 10 == 0 or i == len(tiff_files):
                print(f"Processed: {i}/{len(tiff_files)} ({i/len(tiff_files)*100:.1f}%)")
            
            filepath = futures[future]
            try:
                stats, error = future.result()
                if stats:
                    records.append(stats)
                else:
                    skipped.append((filepath.name, error))
            except Exception as e:
                skipped.append((filepath.name, str(e)))
    
    if not records:
        print("\n No valid images processed!")
        return None
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    # Print summary
    print(f"✓ Processed: {len(records)}")
    print(f"✗ Skipped: {len(skipped)} (too cloudy or errors)")
    
    print(f"\nPer year:")
    for year, count in df.groupby('year').size().items():
        print(f"   {year}: {count}")
    
    print(f"\nValue ranges:")
    print(f"   NDVI: [{df['NDVI_mean'].min():.3f}, {df['NDVI_mean'].max():.3f}]")
    print(f"   NBR:  [{df['NBR_mean'].min():.3f}, {df['NBR_mean'].max():.3f}]")
    print(f"   NDMI: [{df['NDMI_mean'].min():.3f}, {df['NDMI_mean'].max():.3f}]")
    
    print(f"\nQuality:")
    print(f"   Avg valid: {df['valid_fraction'].mean()*100:.1f}%")
    print(f"   Min valid: {df['valid_fraction'].min()*100:.1f}%")
    
    # Show some skipped files for debugging
    if skipped:
        print(f"\nSkipped files (first 10):")
        for filename, error in skipped[:10]:
            print(f"   • {filename}: {error}")
    
    return df


def process_directory_serial(input_dir):
    
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    
    print(f"\nPREPROCESSING: {len(tiff_files)} files found")
    
    records = []
    skipped = []
    
    for i, filepath in enumerate(tiff_files, 1):
        if i % 10 == 0 or i == len(tiff_files):
            print(f"Processing {i}/{len(tiff_files)} ({i/len(tiff_files)*100:.1f}%)")
        
        stats, error = process_tiff(filepath)
        
        if stats:
            records.append(stats)
        else:
            skipped.append((filepath.name, error))
    
    if not records:
        print("\n No valid images processed!")
        return None
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    print(f"✓ Processed: {len(records)}")
    print(f"✗ Skipped: {len(skipped)}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Verify directory exists
    if not DRIVE_INPUT_DIR.exists():
        print(f"\n ERROR: Directory not found!")
        print(f"   Looking for: {DRIVE_INPUT_DIR}")
        print(f"\n Contents of My Drive:")
        if MY_DRIVE.exists():
            for item in sorted(MY_DRIVE.iterdir()):
                if item.is_dir():
                    print(f"    {item.name}")
        exit(1)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process all TIFF files
    if USE_PARALLEL:
        df = process_directory_parallel(DRIVE_INPUT_DIR, max_workers=MAX_WORKERS)
    else:
        df = process_directory_serial(DRIVE_INPUT_DIR)
    
    if df is not None:
        # Save raw images
        raw_csv = OUTPUT_DIR / "01_all_images.csv"
        df.to_csv(raw_csv, index=False)
        print(f"\n✓ Saved all images: {raw_csv} ({len(df)} images)")
        
        # Create and save composites
        composites = create_pixel_composite(df, days=COMPOSITE_DAYS)
        
        composites.to_csv(OUTPUT_CSV, index=False)
        composites.to_pickle(OUTPUT_PKL)
        
        print("OUTPUT FILES")
        print(f"Raw images:  {raw_csv} ({len(df)} images)")
        print(f"Composites:  {OUTPUT_CSV} ({len(composites)} composites)")
        print(f"Pickle:      {OUTPUT_PKL}")
        
        print(f"\n Composite Preview:")
        print(composites[['date', 'year', 'month', 'n_images', 
                         'NDVI_mean', 'NBR_mean', 'NDMI_mean']].head(15))
        
        print(" PREPROCESSING COMPLETE")
