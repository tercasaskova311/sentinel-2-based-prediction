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
OUTPUT_COMPOSITES_DIR = OUTPUT_DIR / "composites"  # ← NEW: Directory for composite TIFFs
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
    3. Calculate mean indices for metadata
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
            
            # Compute basic statistics for metadata
            stats = {
                "date": date,
                "year": date.year,
                "month": date.month,
                "filepath": str(filepath),  #  Store filepath for later compositing
                "filename": filepath.name,
                "valid_fraction": float(valid_fraction),
                
                # Core indices for forest health (for metadata only)
                "NDVI_mean": float(np.nanmean(ndvi)),  # Vegetation greenness
                "NBR_mean": float(np.nanmean(nbr)),    # Bark beetle indicator
                "NDMI_mean": float(np.nanmean(ndmi)),  # Moisture stress
            }
            
            return stats, None
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# PIXEL-LEVEL COMPOSITING - I have adjusted thelogic here - I think this could be a smart way to do it because:
# - we take median of valid pixels over multiple images in the period of 10 days =>  which allow us to work with images eventought they are cloudy
# - we calculate valid fraction of the composite itself => so we know how much of the composite is valid data
# - we calculate mean indices from the composite itself => more robust statistics
# - this approach should give us better spatial detection capability
# - also I think later for labeling: we can use these composites directly to see spatial patterns of deforestation

def create_pixel_composite(tiff_files, output_path):
#pixel-level composite from multiple TIFF files => take MEDIAN of valid (non-cloud) values
    
    arrays = []
    valid_files = []
    
    for tiff in tiff_files:
        try:
            with rasterio.open(tiff) as src:
                arr = src.read().astype(np.float32)
                
                # Mask invalid values (clouds, out-of-range)
                for band_idx in range(arr.shape[0]):
                    band = arr[band_idx]
                    band[(band < -1) | (band > 1)] = np.nan
                
                arrays.append(arr)
                valid_files.append(tiff)
                
        except Exception as e:
            print(f" Skipping {tiff.name}: {e}")
    
    if not arrays:
        return None
    
    # Stack along new axis: (n_images, n_bands, height, width)
    stacked = np.stack(arrays, axis=0)
    
    # Take MEDIAN across images for each pixel (more robust than mean)
    composite = np.nanmedian(stacked, axis=0)  # Shape: (n_bands, height, width)
    
    # Calculate valid fraction of composite
    valid_fraction = np.isfinite(composite[1]).sum() / composite[1].size
    
    # Save composite image
    with rasterio.open(valid_files[0]) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, compress='lzw')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(composite.astype(np.float32))
    
    return output_path, valid_fraction


def create_composites_from_images(df, output_dir, days=10):
    """
    Create pixel-level composites from raw images
    
    Args:
        df: DataFrame with processed image metadata (must have 'filepath' and 'date' columns)
        output_dir: Directory to save composite TIFFs
        days: Number of days per composite period
    
    Returns:
        DataFrame with composite metadata
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by period
    df['period_key'] = (
        (df['date'] - pd.Timestamp('2020-01-01')).dt.days // days
    )
    
    print(f"CREATING PIXEL-LEVEL COMPOSITES")
    print(f"Images: {len(df)}")
    print(f"Periods: {df['period_key'].nunique()}")
    
    composite_metadata = []
    
    for period_key, period_df in df.groupby('period_key'):
        # Get files for this period
        tiff_files = [Path(fp) for fp in period_df['filepath']]
        first_date = period_df['date'].min()
        
        # Create composite filename
        composite_name = f"composite_{first_date.strftime('%Y-%m-%d')}.tif"
        output_path = output_dir / composite_name
        
        print(f"\n  Creating: {composite_name}")
        print(f"    From {len(tiff_files)} images: {first_date.strftime('%Y-%m-%d')} to {period_df['date'].max().strftime('%Y-%m-%d')}")
        
        # Create pixel-level composite
        result = create_pixel_composite(tiff_files, output_path)
        
        if result is None:
            print(f"Failed to create composite")
            continue
        
        composite_path, valid_fraction = result
        
        # Calculate mean indices from composite
        with rasterio.open(composite_path) as src:
            composite_arr = src.read()
            ndvi = composite_arr[1]  # NDVI is band 2
            nbr = composite_arr[2]   # NBR is band 3
            ndmi = composite_arr[3]  # NDMI is band 4
        
        # Store metadata
        composite_metadata.append({
            'date': first_date,
            'year': first_date.year,
            'month': first_date.month,
            'day_of_year': first_date.timetuple().tm_yday,
            'composite_path': str(composite_path),
            'n_images': len(tiff_files),
            'valid_fraction': valid_fraction,
            'NDVI_mean': float(np.nanmean(ndvi)),
            'NBR_mean': float(np.nanmean(nbr)),
            'NDMI_mean': float(np.nanmean(ndmi)),
        })
        
        print(f"    ✓ Saved: {output_path}")
        print(f"    ✓ Valid pixels: {valid_fraction*100:.1f}%")
    
    composites_df = pd.DataFrame(composite_metadata).sort_values('date').reset_index(drop=True)
    
    print(f"✓ Created {len(composites_df)} pixel-level composites")
    print(f"✓ Saved to: {output_dir}")    
    return composites_df


def process_directory_parallel(input_dir, max_workers=2):
    
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    print(f"Found: {len(tiff_files)} TIFF files")
    
    records = []
    skipped = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tiff, f): f for f in tiff_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            if i % 10 == 0 or i == len(tiff_files):
                print(f"  Processed: {i}/{len(tiff_files)} ({i/len(tiff_files)*100:.1f}%)")
            
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
    
    if skipped and len(skipped) <= 10:
        print(f"\nSkipped files:")
        for filename, error in skipped:
            print(f"   • {filename}: {error}")
    elif skipped:
        print(f"\nSkipped files (first 10):")
        for filename, error in skipped[:10]:
            print(f"   • {filename}: {error}")
        print(f"   ... and {len(skipped)-10} more")
    
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
                    print(f"{item.name}")
        exit(1)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # STEP 1: Process all raw TIFF files (extract metadata)
    if USE_PARALLEL:
        df = process_directory_parallel(DRIVE_INPUT_DIR, max_workers=MAX_WORKERS)
    else:
        print(" Serial processing not implemented - using parallel")
        df = process_directory_parallel(DRIVE_INPUT_DIR, max_workers=MAX_WORKERS)
    
    if df is None:
        print("\n Preprocessing failed!")
        exit(1)
    
    # Save raw image metadata
    raw_csv = OUTPUT_DIR / "01_all_images.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\n✓ Saved raw metadata: {raw_csv} ({len(df)} images)")
    
    # STEP 2: Create pixel-level composites
    composites_df = create_composites_from_images(
        df, 
        OUTPUT_COMPOSITES_DIR, 
        days=COMPOSITE_DAYS
    )
    
    # Save composite metadata
    composites_df.to_csv(OUTPUT_CSV, index=False)
    composites_df.to_pickle(OUTPUT_PKL)
    
    print(f"OUTPUT FILES")
    print(f"Raw metadata:     {raw_csv} ({len(df)} images)")
    print(f"Composites (CSV): {OUTPUT_CSV} ({len(composites_df)} composites)")
    print(f"Composites (PKL): {OUTPUT_PKL}")
    print(f"Composite TIFFs:  {OUTPUT_COMPOSITES_DIR}/ ({len(composites_df)} files)")
    

    
