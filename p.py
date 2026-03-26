import json
import subprocess
from osgeo import gdal
import numpy as np

# =============================================================================
# CONFIG — change filename to your latest version
# =============================================================================
TIF_PATH     = '/Users/terezasaskova/Downloads/sumava_alerts_2025-6.tif'
GEOJSON_RAW  = '/Users/terezasaskova/Downloads/alerts_raw.geojson'
GEOJSON_OUT  = '/Users/terezasaskova/Downloads/alerts_final.geojson'

# =============================================================================
# 1. Check raster
# =============================================================================
ds = gdal.Open(TIF_PATH)
band = ds.GetRasterBand(1)
data = band.ReadAsArray()
alert_px = (data == 1).sum()
area_ha = alert_px * 100 / 10000  # 10m pixels = 100m² each
print(f'Alert pixels : {alert_px}')
print(f'Alert area   : {area_ha:.1f} ha')
ds = None

# =============================================================================
# 2. Polygonize
# =============================================================================
print('Running polygonize...')
result = subprocess.run([
    'gdal_polygonize.py', '-8',
    TIF_PATH,
    '-b', '1',
    '-f', 'GeoJSON',
    GEOJSON_RAW,
    'OUTPUT', 'alerts'
], capture_output=True, text=True)

if result.returncode != 0:
    print('Polygonize error:', result.stderr)
else:
    print('Polygonize done')

# =============================================================================
# 3. Fix CRS + filter alert=1 + add area
# =============================================================================
with open(GEOJSON_RAW) as f:
    gj = json.load(f)

total_before = len(gj['features'])

# Add CRS — coordinates are EPSG:32633 (UTM Zone 33N)
gj['crs'] = {
    "type": "name",
    "properties": {"name": "urn:ogc:def:crs:EPSG::32633"}
}

# Keep only alert=1 with valid geometry
clean = [f for f in gj['features']
         if f['geometry'] is not None
         and f['properties'].get('alerts') == 1]

print(f'Features before filter : {total_before}')
print(f'Alert=1 features kept  : {len(clean)}')

# Add proper area using shoelace formula (exact, not bbox approximation)
def polygon_area_m2(coords):
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

for f in clean:
    coords = f['geometry']['coordinates'][0]
    area_m2 = polygon_area_m2(coords)
    f['properties']['area_ha'] = round(area_m2 / 10000, 4)

# Sort by area descending
clean.sort(key=lambda f: f['properties']['area_ha'], reverse=True)

# Print summary
areas = [f['properties']['area_ha'] for f in clean]
print(f'Total alert area       : {sum(areas):.1f} ha')
print(f'Largest patch          : {max(areas):.2f} ha')
print(f'Smallest patch         : {min(areas):.4f} ha')
print(f'Mean patch size        : {sum(areas)/len(areas):.2f} ha')

gj['features'] = clean

with open(GEOJSON_OUT, 'w') as f:
    json.dump(gj, f, indent=2)

print(f'Saved → {GEOJSON_OUT}')
print('Load in QGIS → assign CRS EPSG:32633 if prompted')