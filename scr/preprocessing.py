#process it thought google drive app... GEE => drive export => local drive sync
#preprocessing: extract date, mask clouds, calculate indices averages, create composites every 10 days

import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

GDRIVE_ROOT = Path.home() / "Library/CloudStorage"
gdrive_folders = list(GDRIVE_ROOT.glob("GoogleDrive-*"))

if gdrive_folders:
    MY_DRIVE = gdrive_folders[0] / "My Drive"
    DRIVE_INPUT_DIR = MY_DRIVE / "sumava_full"
else:
    raise FileNotFoundError("Did not find Google Drive folder")

OUTPUT_DIR = Path("output")
OUTPUT_COMPOSITES_DIR = OUTPUT_DIR / "composites"
OUTPUT_CSV = OUTPUT_DIR / "01_preprocessed_timeseries.csv"
OUTPUT_PKL = OUTPUT_DIR / "01_preprocessed_timeseries.pkl"

MIN_VALID_FRACTION = 0.30
COMPOSITE_DAYS = 10
USE_PARALLEL = True
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  

def extract_date(filename): 
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", str(filename))
    if match:
        try:
            return datetime(*map(int, match.groups()))
        except:
            return None
    return None


def process_tiff_fast(filepath):
    
    try:
        with rasterio.open(filepath) as src:
            # Quick metadata check first (don't read full array yet)
            if src.count != 4:
                return None, f"Expected 4 bands, got {src.count}"
            
            # Read all bands at once (faster than separate reads)
            arr = src.read(out_dtype=np.float32)
            
            b8, ndvi, nbr, ndmi = arr
            
            # Vectorized masking - all indices at once
            mask = ((ndvi < -1) | (ndvi > 1) | 
                   (nbr < -1) | (nbr > 1) | 
                   (ndmi < -1) | (ndmi > 1))
            
            ndvi[mask] = np.nan
            nbr[mask] = np.nan
            ndmi[mask] = np.nan
            
            # Quick validity check
            valid_fraction = np.isfinite(ndvi).sum() / ndvi.size
            
            if valid_fraction < MIN_VALID_FRACTION:
                return None, f"Too few valid pixels: {valid_fraction:.1%}"
            
            # Extract date
            date = extract_date(filepath.name)
            if date is None:
                return None, "No date in filename"
            
            # Minimal statistics (we'll recalculate from composites anyway)
            stats = {
                "date": date,
                "year": date.year,
                "month": date.month,
                "filepath": str(filepath),
                "filename": filepath.name,
                "valid_fraction": float(valid_fraction),
                "NDVI_mean": float(np.nanmean(ndvi)),
                "NBR_mean": float(np.nanmean(nbr)),
                "NDMI_mean": float(np.nanmean(ndmi)),
            }
            
            return stats, None
            
    except Exception as e:
        return None, f"Error: {str(e)}"

#this is important - soemthing I have changed - we can use composides instead of each imaginery
#composite = median of values of bands in the 10 days window, basically this avoid skipping too many images thanks to cloud masking becuase we take median form all images in given window and create a "new image - composite"
#i think we can ealy use this later for the labeling - simply use composites and performe spatial detection on them
#what is also importatn here taht we do pixel based compositing - meaning for each pixel we take median of all values accross images in given window - this is super important for later detection step


def create_pixel_composite_fast(tiff_files, output_path):
    
    if not tiff_files:
        return None
    
#Pre-allocate arrays
    first_file = tiff_files[0]
    with rasterio.open(first_file) as src:
        shape = src.shape
        n_bands = src.count
        profile = src.profile.copy()
    
    # Stack arrays efficiently
    arrays = np.zeros((len(tiff_files), n_bands, shape[0], shape[1]), dtype=np.float32)
    arrays[:] = np.nan  # Initialize with NaN
    
    valid_count = 0
    
    for i, tiff in enumerate(tiff_files):
        try:
            with rasterio.open(tiff) as src:
                arr = src.read(out_dtype=np.float32)
                
                # Vectorized masking
                mask = ((arr < -1) | (arr > 1))
                arr[mask] = np.nan
                
                arrays[i] = arr
                valid_count += 1
                
        except Exception as e:
            print(f"Skipping {tiff.name}: {e}")
            continue
    
    if valid_count == 0:
        return None
    
    #nanmedian with efficient axis
    with np.errstate(invalid='ignore'):  # Suppress warnings
        composite = np.nanmedian(arrays[:valid_count], axis=0)
    
    # Calculate valid fraction
    valid_fraction = np.isfinite(composite[1]).sum() / composite[1].size
    
    profile.update(dtype=rasterio.float32, compress='lzw')
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(composite)
    
    return output_path, valid_fraction


def create_composites_from_images_fast(df, output_dir, days=10):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by period
    df['period_key'] = (
        (df['date'] - pd.Timestamp('2020-01-01')).dt.days // days
    )
    
    print(f"CREATING PIXEL-LEVEL COMPOSITES")
    print(f"Images: {len(df)}")
    print(f"Periods: {df['period_key'].nunique()}")
    
    composite_jobs = []
    for period_key, period_df in df.groupby('period_key'):
        tiff_files = [Path(fp) for fp in period_df['filepath']]
        first_date = period_df['date'].min()
        composite_name = f"composite_{first_date.strftime('%Y-%m-%d')}.tif"
        output_path = output_dir / composite_name
        
        composite_jobs.append({
            'period_key': period_key,
            'tiff_files': tiff_files,
            'output_path': output_path,
            'first_date': first_date,
            'last_date': period_df['date'].max(),
        })
    
    print(f"\n Processing {len(composite_jobs)} composites in parallel...")
    
    # Process composites in parallel
    composite_metadata = []
    start_time = time.time()
    
    def process_composite_job(job):
        """Helper function for parallel processing"""
        result = create_pixel_composite_fast(job['tiff_files'], job['output_path'])
        
        if result is None:
            return None
        
        composite_path, valid_fraction = result
        
        # Read composite to calculate statistics
        with rasterio.open(composite_path) as src:
            composite_arr = src.read()
            ndvi = composite_arr[1]
            nbr = composite_arr[2]
            ndmi = composite_arr[3]
        
        return {
            'date': job['first_date'],
            'year': job['first_date'].year,
            'month': job['first_date'].month,
            'day_of_year': job['first_date'].timetuple().tm_yday,
            'composite_path': str(composite_path),
            'n_images': len(job['tiff_files']),
            'valid_fraction': valid_fraction,
            'NDVI_mean': float(np.nanmean(ndvi)),
            'NBR_mean': float(np.nanmean(nbr)),
            'NDMI_mean': float(np.nanmean(ndmi)),
        }
    
    #Parallel composite creation
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(composite_jobs))) as executor:
        futures = {executor.submit(process_composite_job, job): job for job in composite_jobs}
        
        for i, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            
            try:
                result = future.result()
                if result:
                    composite_metadata.append(result)
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(composite_jobs) - i) / rate if rate > 0 else 0
                    
                    print(f"  [{i}/{len(composite_jobs)}] {job['output_path'].name} "
                          f"({len(job['tiff_files'])} imgs) | "
                          f"ETA: {eta/60:.1f}min", end='\r')
                else:
                    print(f"\n Failed: {job['output_path'].name}")
                    
            except Exception as e:
                print(f"\n Error: {job['output_path'].name}: {e}")
    
    print()  
    
    composites_df = pd.DataFrame(composite_metadata).sort_values('date').reset_index(drop=True)
    
    elapsed = time.time() - start_time
    print(f"✓ Created {len(composites_df)} composites in {elapsed/60:.1f} minutes")
    print(f"✓ Average: {elapsed/len(composites_df):.1f}s per composite")
    
    return composites_df


def process_directory_parallel_fast(input_dir, max_workers):
    
    tiff_files = list(Path(input_dir).rglob("*.tif"))
    tiff_files = [f for f in tiff_files if not f.name.startswith('.')]
    
    print(f"STEP 1: PROCESSING RAW IMAGES")
    print(f"Files: {len(tiff_files)}")
    print(f"Workers: {max_workers}")
    
    records = []
    skipped = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tiff_fast, f): f for f in tiff_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            filepath = futures[future]
            
            try:
                stats, error = future.result()
                if stats:
                    records.append(stats)
                else:
                    skipped.append((filepath.name, error))
            except Exception as e:
                skipped.append((filepath.name, str(e)))
            
            # Better progress tracking
            if i % 10 == 0 or i == len(tiff_files):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(tiff_files) - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{len(tiff_files)} ({i/len(tiff_files)*100:.0f}%) | "
                      f"Rate: {rate:.1f} files/s | ETA: {eta/60:.1f}min", end='\r')
    
    print()
    
    if not records:
        print("\n No valid images processed!")
        return None
    
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    elapsed = time.time() - start_time
    
    print(f"PROCESSING SUMMARY")
    print(f"✓ Processed: {len(records)} in {elapsed/60:.1f} minutes")
    print(f"✓ Average: {elapsed/len(records):.2f}s per file")
    print(f"✗ Skipped: {len(skipped)}")
    
    print(f"\nPer year:")
    for year, count in df.groupby('year').size().items():
        print(f"   {year}: {count}")
    
    if skipped and len(skipped) <= 5:
        print(f"\nSkipped files:")
        for filename, error in skipped:
            print(f"   • {filename}: {error}")
    elif skipped:
        print(f"\nSkipped (showing first 5 of {len(skipped)}):")
        for filename, error in skipped[:5]:
            print(f"   • {filename}: {error}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    overall_start = time.time()
    
    if not DRIVE_INPUT_DIR.exists():
        print(f"\n ERROR: Directory not found!")
        print(f"   Looking for: {DRIVE_INPUT_DIR}")
        exit(1)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # STEP 1: Process raw images
    df = process_directory_parallel_fast(DRIVE_INPUT_DIR, max_workers=MAX_WORKERS)
    
    if df is None:
        print("\n Preprocessing failed!")
        exit(1)
    
    # Save raw metadata
    raw_csv = OUTPUT_DIR / "01_all_images.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\n✓ Saved raw metadata: {raw_csv}")
    
    # STEP 2: Create composites
    composites_df = create_composites_from_images_fast(
        df, 
        OUTPUT_COMPOSITES_DIR, 
        days=COMPOSITE_DAYS
    )
    
    # Save composite metadata
    composites_df.to_csv(OUTPUT_CSV, index=False)
    composites_df.to_pickle(OUTPUT_PKL)
    
    overall_elapsed = time.time() - overall_start
    
    print(f" PREPROCESSING COMPLETE!")
    print(f"\nOutput files:")
    print(f"  Raw metadata:     {raw_csv} ({len(df)} images)")
    print(f"  Composites (CSV): {OUTPUT_CSV} ({len(composites_df)} composites)")
    print(f"  Composite TIFFs:  {OUTPUT_COMPOSITES_DIR}/ ({len(composites_df)} files)")



