import re
from pathlib import Path
from datetime import datetime
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
    raise FileNotFoundError("did not find Google Drive folder")

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "01_preprocessed_timeseries.csv"
OUTPUT_PKL = OUTPUT_DIR / "01_preprocessed_timeseries.pkl"

MIN_VALID_FRACTION = 0.30
COMPOSITE_DAYS = 10  # Create 10-day composites = basically means downloading every 10 days
USE_PARALLEL = True  
MAX_WORKERS = 2  # Reduce if crashes....

# ============================================================================

def extract_date(filename): #from the filename
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", str(filename))
    if match:
        try:
            return datetime(*map(int, match.groups()))
        except:
            return None
    return None


def process_tiff_efficient(filepath):
    """Process one TIFF file with downsampling to save memory"""
    try:
        with rasterio.open(filepath) as src:
            scale = 2
            
            arr = src.read(
                out_shape=(
                    src.count,
                    src.height // scale,
                    src.width // scale
                ),
                resampling=rasterio.enums.Resampling.average
            ).astype(np.float32)
            
            # Handle 4-band images (B8, NDVI, NBR, NDMI)
            if arr.shape[0] == 4:
                b8, ndvi, nbr, ndmi = arr
                # Skip cloud masking since we don't have SCL
                good = np.ones_like(ndvi, dtype=bool)  # Assume all pixels are good
                
            # Handle 8-band images (B4, B8, B11, B12, SCL, NDVI, NBR, NDMI)
            elif arr.shape[0] == 8:
                b4, b8, b11, b12, scl, ndvi, nbr, ndmi = arr
                # Mask bad pixels using SCL
                good = (
                    (scl != 0) & (scl != 1) & (scl != 3) &
                    (scl != 8) & (scl != 9) & (scl != 10) & (scl != 11)
                )
            else:
                return None, f"Expected 4 or 8 bands, got {arr.shape[0]}"
            
            # Apply mask
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
            
            # Compute stats (same as before)
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
            }
            
            return stats, None
            
    except Exception as e:
        return None, f"Error: {str(e)}"


def assign_composite_period(date, days=10):
    """Assign image to composite period"""
    from datetime import timedelta
    year = date.year
    day_of_year = date.timetuple().tm_yday
    period = (day_of_year - 1) // days
    
    # Create period start date
    period_start = datetime(year, 1, 1) + timedelta(days=period * days)
    return period_start


def create_composites(df, days=10):
    """Create temporal composites from all images"""
    print(f"\n{'='*70}")
    print(f"CREATING {days}-DAY COMPOSITES")
    print(f"{'='*70}")
    
    # Assign composite periods
    df['composite_period'] = df['date'].apply(
        lambda x: assign_composite_period(x, days)
    )
    
    print(f"\nImages: {len(df)}")
    print(f"Periods: {df['composite_period'].nunique()}")
    
    # Group by period and aggregate
    composites = df.groupby('composite_period').agg({
        'NDVI_mean': 'mean',
        'NBR_mean': 'mean',
        'NDMI_mean': 'mean',
        'NDVI_std': 'mean',
        'NBR_std': 'mean',
        'NDMI_std': 'mean',
        'NDVI_p50': 'median',
        'NBR_p50': 'median',
        'NDMI_p50': 'median',
        'valid_fraction': 'mean',
        'date': 'first',
        'filename': 'count'
    }).reset_index()
    
    # Rename count column
    composites.rename(columns={'filename': 'n_images'}, inplace=True)
    
    # Add temporal features
    composites['year'] = composites['date'].dt.year
    composites['month'] = composites['date'].dt.month
    composites['day_of_year'] = composites['date'].dt.dayofyear
    
    print(f"\nComposites created: {len(composites)}")
    print(f"Avg images per composite: {composites['n_images'].mean():.1f}")
    
    return composites.sort_values('date').reset_index(drop=True)


def process_directory_parallel(input_dir, max_workers=2):
    """Process all TIFFs with parallel processing"""
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {len(tiff_files)} files found")
    print(f"Using {max_workers} workers")
    print(f"{'='*70}")
    
    records = []
    skipped = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tiff_efficient, f): f 
                  for f in tiff_files}
        
        for i, future in enumerate(as_completed(futures), 1):
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
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"✓ Processed: {len(records)}")
    print(f"✗ Skipped: {len(skipped)}")
    
    print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")
    print(f"Total span: {(df['date'].max() - df['date'].min()).days} days")
    
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
    
    if skipped:
        print(f"\n Skipped files (first 10):")
        for name, reason in skipped[:10]:
            print(f"   • {name}: {reason}")
    
    return df


def process_directory_serial(input_dir):
    """Process all TIFFs serially (safer for low memory)"""
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {len(tiff_files)} files found")
    print(f"Serial processing (safer for memory)")
    print(f"{'='*70}")
    
    records = []
    skipped = []
    
    for i, filepath in enumerate(tiff_files, 1):
        if i % 10 == 0 or i == len(tiff_files):
            print(f"Processing {i}/{len(tiff_files)} ({i/len(tiff_files)*100:.1f}%)")
        
        stats, error = process_tiff_efficient(filepath)
        
        if stats:
            records.append(stats)
        else:
            skipped.append((filepath.name, error))
    
    if not records:
        print("\n No valid images processed!")
        return None
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    # Same summary as parallel version
    print(f"RESULTS")
    print(f" Processed: {len(records)}")
    print(f" Skipped: {len(skipped)}")
    
    return df

# ============================================================================

if __name__ == "__main__":
    # Verify directory exists
    if not DRIVE_INPUT_DIR.exists():
        print(f"\n ERROR: Directory not found!")
        print(f"   Looking for: {DRIVE_INPUT_DIR}")
        print(f"\n📁 Contents of My Drive:")
        if MY_DRIVE.exists():
            for item in sorted(MY_DRIVE.iterdir()):
                if item.is_dir():
                    print(f"   📁 {item.name}")
        print("\n Update the folder name in CONFIG section if needed")
        exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process files
    if USE_PARALLEL:
        df = process_directory_parallel(DRIVE_INPUT_DIR, max_workers=MAX_WORKERS)
    else:
        df = process_directory_serial(DRIVE_INPUT_DIR)
    
    if df is not None:
        # Save all images
        df.to_csv(OUTPUT_DIR / "01_all_images.csv", index=False)
        print(f"\n Saved all images: output/01_all_images.csv ({len(df)} images)")
        
        # Create and save composites
        composites = create_composites(df, days=COMPOSITE_DAYS)
        
        composites.to_csv(OUTPUT_CSV, index=False)
        composites.to_pickle(OUTPUT_PKL)
        
        print(f"Raw images:  output/01_all_images.csv ({len(df)} images)")
        print(f"Composites:  {OUTPUT_CSV} ({len(composites)} composites)")
        print(f"Pickle:      {OUTPUT_PKL}")
        
        print(f"\n Composite Preview:")
        print(composites[['date', 'year', 'month', 'n_images', 'NDVI_mean', 'NBR_mean', 'NDMI_mean']].head(15))
        