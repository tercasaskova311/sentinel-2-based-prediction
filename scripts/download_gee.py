import ee
import geemap
import geopandas as gpd
import os
from datetime import datetime

# CONFIGURATION
AOI_PATH = "sumava_aoi_clean.geojson"
OUTPUT_DIR = "data/historical"
RESOLUTION = 20
START_YEAR = 2020
END_YEAR = 2024
MAX_CLOUD_COVERAGE = 30
TEMPORAL_INTERVAL_DAYS = 16

# INITIALIZE EE
ee.Initialize(project='youtubeapi-455317')

# LOAD AOI (correct polygon, no rectangle!)
aoi_gdf = gpd.read_file(AOI_PATH)

# EE polygon
aoi_ee = ee.Geometry.Polygon(
    aoi_gdf.geometry.iloc[0].__geo_interface__["coordinates"]
)

# Area calculation
aoi_proj = aoi_gdf.to_crs(32633)
print(f"📍 AOI Bounds: {aoi_gdf.total_bounds}")
print(f"📏 AOI Area: {aoi_proj.geometry.area.sum() / 1e6:.2f} km²")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
    nbr  = image.normalizedDifference(['B8','B12']).rename('NBR')
    ndmi = image.normalizedDifference(['B8','B11']).rename('NDMI')
    return image.addBands([ndvi, nbr, ndmi])

def mask_clouds(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)

def download_image(image, date_str, output_folder):
    task = ee.batch.Export.image.toDrive(
        image=image.select(['NDVI','NBR','NDMI']),
        description=f"sumava_{date_str}",
        folder="sumava_exports",     
        fileNamePrefix=f"sumava_{date_str}",
        scale=RESOLUTION,
        region=aoi_ee,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Export started: {date_str}")

def download_sentinel_year(year):
    start_date = f"{year}-04-01"
    end_date   = f"{year}-10-31"

    print(f"\n=== YEAR {year} ===")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_ee)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVERAGE))
        .map(mask_clouds)
        .map(add_indices)
    )

    n = collection.size().getInfo()
    print(f"Found: {n} images")

    # sample images ~every 16 days
    image_list = collection.toList(n)
    step = max(1, n // (365 // TEMPORAL_INTERVAL_DAYS))

    year_folder = os.path.join(OUTPUT_DIR, str(year))
    os.makedirs(year_folder, exist_ok=True)

    for i in range(0, n, step):
        img = ee.Image(image_list.get(i))
        date_str = img.date().format('YYYY-MM-dd').getInfo()
        download_image(img, date_str, year_folder)

for year in range(START_YEAR, END_YEAR + 1):
    download_sentinel_year(year)
