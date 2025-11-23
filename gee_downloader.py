"""
Bulk Sentinel-2 Download using Google Earth Engine (FREE!)
No credit card, no trial limits, perfect for academic projects

Prerequisites:
1. Sign up at https://earthengine.google.com (use your university email)
2. pip install earthengine-api geemap geopandas
3. Authenticate once: earthengine authenticate
"""

import ee
import geemap
import geopandas as gpd
import os
from datetime import datetime
import json

# ========================================
# CONFIGURATION
# ========================================
AOI_PATH = "sumava_aoi_clean.geojson"
OUTPUT_DIR = "data/historical"
RESOLUTION = 20  # meters

# Historical time range
START_YEAR = 2020
END_YEAR = 2024
MAX_CLOUD_COVERAGE = 30

# Download every N days
TEMPORAL_INTERVAL_DAYS = 16

# ========================================
# INITIALIZE EARTH ENGINE
# ========================================
ee.Initialize(project='youtubeapi-455317')
print("OK")

# ========================================
# LOAD AOI
# ========================================
aoi_gdf = gpd.read_file(AOI_PATH)
aoi_bounds = aoi_gdf.total_bounds  # [minx, miny, maxx, maxy]

# Convert to EE geometry
aoi_ee = ee.Geometry.Rectangle([
    aoi_bounds[0], aoi_bounds[1],  # minx, miny
    aoi_bounds[2], aoi_bounds[3]   # maxx, maxy
])

print(f"\n📍 AOI Bounds: {aoi_bounds}")
print(f"📏 AOI Area: {aoi_gdf.geometry.area.sum() / 1e6:.2f} km²")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# HELPER FUNCTIONS
# ========================================
def get_sentinel2_collection(year):
    """Get Sentinel-2 collection from april - october each year - because or cloud mask and snow mask and sun/shade influence in the winter..."""
    
    start_date = f"{year}-04-01"  # April (snow melting)
    end_date = f"{year}-10-31"    # October (before snow)
    
    print(f"\n🔍 Searching Sentinel-2 for {year}...")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Load Sentinel-2 Surface Reflectance collection
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi_ee)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVERAGE)))
    
    size = collection.size().getInfo()
    print(f"   Found {size} images with < {MAX_CLOUD_COVERAGE}% clouds")
    
    return collection

def add_indices(image):
    """
    Add spectral indices to image:
    NDVI: General vegetation health (classic but saturates)
    NBR: BEST for logging (sensitive to moisture + structure)
    NDMI: Moisture content (distinguishes rapid vs. slow change)
    NDWI: Water masking (prevents false positives)
    """
    
    # NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NBR (Normalized Burn Ratio)
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # NDMI (Moisture Index)
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    # NDWI (Water Index)
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    return image.addBands([ndvi, nbr, ndmi, ndwi])

def mask_clouds_seasonal(image):
    scl = image.select('SCL')
    
    # Get image date
    date = ee.Date(image.get('system:time_start'))
    month = date.get('month')
    
    # Winter months (Dec-Mar): Mask snow aggressively
    # Summer months (Apr-Nov): Keep snow (rare, might be cloud misclassification)
    
    mask_winter = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
                   .And(scl.neq(10)).And(scl.neq(11)))  # Mask snow
    
    mask_summer = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
                   .And(scl.neq(10)))  # Don't mask snow (likely misclassified clouds)
    
    # Apply seasonally
    mask = ee.Algorithms.If(month.gte(4).And(month.lte(10)),
                             mask_summer,  # Apr-Oct
                             mask_winter)  # Nov-Mar
    
    return image.updateMask(ee.Image(mask))
    
def download_image(image, date_str, year_dir):
    """Download single image as GeoTIFF"""
    
    print(f"   📥 Downloading {date_str}...", end=" ", flush=True)
    
    try:
        # Select and rename bands
        bands_to_download = image.select(['NDVI', 'NBR', 'NDMI'])
        
        # Export as GeoTIFF
        filename = f"sumava_{date_str}.tif"
        filepath = os.path.join(year_dir, filename)
        
        # Download using geemap
        geemap.ee_export_image(
            bands_to_download,
            filename=filepath,
            scale=RESOLUTION,
            region=aoi_ee,
            file_per_band=False
        )
        
        print("✅")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ========================================
# MAIN PIPELINE
# ========================================
def main():
    print("=" * 70)
    print("HISTORICAL SENTINEL-2 DOWNLOAD via GOOGLE EARTH ENGINE (FREE!)")
    print("=" * 70)
    
    all_downloads = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = os.path.join(OUTPUT_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        # Get collection
        collection = get_sentinel2_collection(year)
        
        # Apply cloud masking and add indices
        collection = collection.map(mask_clouds_seasonal).map(add_indices)
        
        # Get image list with dates
        image_list = collection.toList(collection.size())
        size = collection.size().getInfo()
        
        # Sample images (every N days)
        dates_info = []
        for i in range(0, size, max(1, size // (365 // TEMPORAL_INTERVAL_DAYS))):
            try:
                img = ee.Image(image_list.get(i))
                date = img.date().format('YYYY-MM-dd').getInfo()
                cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                
                dates_info.append({
                    'index': i,
                    'date': date,
                    'cloud_cover': cloud_cover,
                    'image': img
                })
            except:
                continue
        
        print(f"   📅 Selected {len(dates_info)} images (~every {TEMPORAL_INTERVAL_DAYS} days)")
        
        # Download each image
        year_downloads = []
        for info in dates_info:
            success = download_image(info['image'], info['date'], year_dir)
            
            if success:
                year_downloads.append({
                    'date': info['date'],
                    'cloud_cover': info['cloud_cover'],
                    'year': year
                })
        
        all_downloads.extend(year_downloads)
        print(f"\n✅ Year {year}: {len(year_downloads)} images downloaded\n")
    
    # Save metadata
    metadata = {
        'download_timestamp': datetime.now().isoformat(),
        'date_range': f"{START_YEAR}-{END_YEAR}",
        'total_images': len(all_downloads),
        'max_cloud_coverage': MAX_CLOUD_COVERAGE,
        'resolution': RESOLUTION,
        'temporal_interval_days': TEMPORAL_INTERVAL_DAYS,
        'aoi_bounds': aoi_bounds.tolist(),
        'bands': ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12', 'SCL'],
        'indices': ['NDVI', 'NBR', 'NDMI', 'NDWI'],
        'source': 'Google Earth Engine - COPERNICUS/S2_SR_HARMONIZED',
        'images': all_downloads
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'download_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 70)
    print(f"🎉 DOWNLOAD COMPLETE!")
    print(f"   Total images: {len(all_downloads)}")
    print(f"   Years: {START_YEAR}-{END_YEAR}")
    print(f"   Storage: {OUTPUT_DIR}")
    print(f"   Metadata: {metadata_path}")
    print("=" * 70)
    
    # Print summary
    print("\n📊 Summary by year:")
    for year in range(START_YEAR, END_YEAR + 1):
        year_imgs = [d for d in all_downloads if d['year'] == year]
        if year_imgs:
            avg_cloud = sum(d['cloud_cover'] for d in year_imgs) / len(year_imgs)
            print(f"   {year}: {len(year_imgs)} images, avg cloud: {avg_cloud:.1f}%")
    
    print("\n💡 Next steps:")
    print("   1. Check data/historical/ folder for downloaded images")
    print("   2. Each image contains 12 bands: B02-B12, SCL, NDVI, NBR, NDMI, NDWI")
    print("   3. Ready for preprocessing and labeling!")

if __name__ == "__main__":
    main()