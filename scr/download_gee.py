#!/usr/bin/env python3
"""
Resume Sentinel-2 Download from Specific Date
Modified to start from 2022-05-19 onwards
"""

import ee
import geopandas as gpd
import os
from datetime import datetime, timedelta

# ========================================
# CONFIGURATION
# ========================================
AOI_PATH = "data/boundaries/sumava_aoi_clean.geojson"
OUTPUT_DIR = "data/historical"

# Resume configuration
RESUME_FROM_DATE = "2022-05-19"  
START_YEAR = 2022
END_YEAR = 2024

MAX_CLOUD = 50  
RESOLUTION = 20

# ========================================
# INITIALIZATION
# ========================================

print("RESUMING SENTINEL-2 DOWNLOAD FROM 2022-05-19")

ee.Initialize(project='deforestration-detection')

aoi_gdf = gpd.read_file(AOI_PATH)
aoi_ee = ee.Geometry.Polygon(
    aoi_gdf.geometry.iloc[0].__geo_interface__["coordinates"]
)
print("AOI loaded")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image(image):
    """Add spectral indices"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    return image.addBands([ndvi, nbr, ndmi])


def export_image(image, date_str):
    """Export only indices + one reference band"""
    export_bands = image.select(['B8', 'NDVI', 'NBR', 'NDMI']).toFloat()
    
    task = ee.batch.Export.image.toDrive(
        image=export_bands,
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
    return task


def get_existing_tasks():
    """Get list of already queued/completed tasks to avoid duplicates"""
    try:
        tasks = ee.batch.Task.list()
        existing_dates = set()
        
        for task in tasks:
            desc = task.status().get('description', '')
            if desc.startswith('sumava_'):
                # Extract date from description
                date_str = desc.replace('sumava_', '')
                existing_dates.add(date_str)
        
        return existing_dates
    except Exception as e:
        print(f" Could not check existing tasks: {e}")
        return set()


def download_year(year, start_date_filter=None):
    """
    Download images for a year, optionally filtering by start date
    
    Args:
        year: Year to download
        start_date_filter: Only download images after this date (YYYY-MM-DD)
    """
    
    print(f"\nYEAR {year}")
    print("=" * 70)
    
    # Determine date range
    if start_date_filter and year == int(start_date_filter.split('-')[0]):
        # For the resume year, start from resume date
        start_date = start_date_filter
        print(f"📍 Resuming from: {start_date}")
    else:
        start_date = f"{year}-04-01"
    
    end_date = f"{year}-10-31"
    
    # Get collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_ee)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD))
        .map(process_image)
    )
    
    n = collection.size().getInfo()
    print(f"Available images: {n}")
    
    if n == 0:
        print(" No images found")
        return []
    
    # Get existing tasks to avoid duplicates
    print("Checking for existing tasks...")
    existing_dates = get_existing_tasks()
    print(f"Found {len(existing_dates)} existing tasks")
    
    # Process images
    image_list = collection.toList(n)
    tasks = []
    skipped = 0
    
    for i in range(n):
        img = ee.Image(image_list.get(i))
        date_str = img.date().format('YYYY-MM-dd').getInfo()
        
        # Check if already queued
        if date_str in existing_dates:
            skipped += 1
            if skipped <= 3:  # Only show first few
                print(f"  ⏭️  {date_str} (already queued)")
            continue
        
        # Queue new export
        task = export_image(img, date_str)
        tasks.append((date_str, task))
        print(f" {date_str} (queued)")
        
        if (len(tasks) + skipped) % 10 == 0:
            print(f"     Progress: {len(tasks) + skipped}/{n}")
    
    if skipped > 3:
        print(f"  ... skipped {skipped - 3} more already-queued tasks")
    
    print(f"✓ New exports queued: {len(tasks)}")
    print(f"✓ Skipped (already queued): {skipped}")
    
    return tasks

print("CONFIGURATION")
print(f"Resuming from: {RESUME_FROM_DATE}")
print(f"Years to process: {START_YEAR}-{END_YEAR}")
print(f"Max cloud cover: {MAX_CLOUD}%")
print(f"Resolution: {RESOLUTION}m")

all_tasks = []
total_skipped = 0

for year in range(START_YEAR, END_YEAR + 1):
    if year == START_YEAR:
        tasks = download_year(year, start_date_filter=RESUME_FROM_DATE)
    else:
        tasks = download_year(year)
    
    all_tasks.extend(tasks)

print(f"Total new exports queued: {len(all_tasks)}")

if len(all_tasks) > 0:
    for i, (date_str, task) in enumerate(all_tasks[:10]):
        print(f"  {i+1}. {date_str}")
    
    if len(all_tasks) > 10:
        print(f"  ... and {len(all_tasks) - 10} more")

# you can check the downloading on: https://code.earthengine.google.com/tasks")