import ee
import time
import geopandas as gpd

#import sumava osm gps
sumava = gpd.read_file('sumava_data/sumava_filtered.geojson')

# Authenticate gee 
ee.Authenticate()   # opens browser to log in ... 
PROJECT_ID = 'youtubeapi-455317'   #project ID
ee.Initialize(project=PROJECT_ID)

#====================================================
# 2) Parameters (NPŠUMAVA )
# Merge multiple polygons into one
geom = sumava.unary_union

# Convert to GeoJSON-like dict
geom_json = geom.__geo_interface__

# Build Earth Engine geometry
BBOX = ee.Geometry(geom_json)
START = '2018-01-01'
END   = '2025-10-30'
CLOUD_THRESH = 20    # percent
FOLDER = 'GEE_Exports_20_25_sumava'   # google drive folder 

#====================================================
# 3) Build collection, filter and pick the least cloudy image
col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
       .filterBounds(ee.Geometry.Rectangle(BBOX))
       .filterDate(START, END)
       .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESH))
)

count = col.size().getInfo()
print(f'Found {count} images (cloud<{CLOUD_THRESH}%)')

if count == 0:
    raise SystemExit('No images found — try widening dates or CLOUD_THRESH.')

# choose lowest-cloud image
best = col.sort('CLOUDY_PIXEL_PERCENTAGE').first()

# add quick band selection for RGB (B4,B3,B2)
image_rgb = best.select(['B4','B3','B2'])

#===================================================
# 4) Export to Drive
task = ee.batch.Export.image.toDrive(
    image = image_rgb.clip(ee.Geometry.Rectangle(BBOX)),
    description = 'trento_20_25',
    folder = FOLDER,
    fileNamePrefix = 'trento_20_25',
    scale = 10,
    maxPixels = 1e13
)
task.start()
print('Export started. Task status:')
print(task.status())   # prints task metadata (id, state, etc.)
