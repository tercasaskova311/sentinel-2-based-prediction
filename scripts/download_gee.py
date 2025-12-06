"""
Simple GEE Download - Option A (Full Export) - Exports: B4, B8, B11, B12, SCL, NDVI, NBR, NDMI
"""

import ee
import geopandas as gpd
import os

# ============================================================================
# CONFIG
# ============================================================================
AOI_PATH = "data/boundaries/sumava_aoi_clean.geojson"
OUTPUT_DIR = "data/historical"

START_YEAR = 2020
END_YEAR = 2024

MAX_CLOUD = 30
RESOLUTION = 20

# ============================================================================
# SETUP
# ============================================================================
ee.Initialize(project='deforestration-detection')

aoi_gdf = gpd.read_file(AOI_PATH)
aoi_ee = ee.Geometry.Polygon(
    aoi_gdf.geometry.iloc[0].__geo_interface__["coordinates"]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

def process_image(image):
    """Compute indices and add to image"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    return image.addBands([ndvi, nbr, ndmi])


def export_image(image, date_str):
    """Export full image with all bands"""
    task = ee.batch.Export.image.toDrive(
        image=image.select(['B4', 'B8', 'B11', 'B12', 'SCL', 'NDVI', 'NBR', 'NDMI']),
        description=f"sumava_{date_str}",
        folder="sumava_full",
        fileNamePrefix=f"sumava_{date_str}",
        scale=RESOLUTION,
        region=aoi_ee,
        crs='EPSG:32633',
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    task.start()
    print(f"   ✓ {date_str}")


def download_year(year):
    """Download images for one year"""
    print(f"\n=== YEAR {year} ===")
    
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_ee)
        .filterDate(f"{year}-04-01", f"{year}-10-31")
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD))
        .map(process_image)
    )
    
    n = collection.size().getInfo()
    print(f"Found: {n} images")
    
    if n == 0:
        return
    
    # Sample every ~10 days
    image_list = collection.toList(n)
    step = max(1, n // 20)
    
    for i in range(0, n, step):
        img = ee.Image(image_list.get(i))
        date_str = img.date().format('YYYY-MM-dd').getInfo()
        export_image(img, date_str)

# ============================================================================
# RUN
# ============================================================================

print("\n" + "="*70)
print("DOWNLOADING SENTINEL-2 - FULL EXPORT")
print("="*70)

for year in range(START_YEAR, END_YEAR + 1):
    download_year(year)

print("\n" + "="*70)
print(" Export tasks queued!")
print("Monitor: https://code.earthengine.google.com/tasks")
print("Download from: Google Drive > sumava_full")
print(f"Save to: {OUTPUT_DIR}")
print("="*70)