import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import os
# Load your data
cze_2 = gpd.read_file("sumava_zones_2.geojson")
filtered_json = cze_2[cze_2['KAT'] == "NP"]
aoi = filtered_json.dissolve()
aoi_proj = aoi.to_crs(32633)  # UTM zone 33N
area_km2 = aoi_proj.geometry.area.sum() / 1e6
aoi_proj.to_file("sumava_aoi_clean_proj.geojson", driver="GeoJSON")
print(f"📏 AOI Area: {area_km2:.2f} km²")

# Save the cleaned AOI
output_path = "sumava_aoi_clean.geojson"
aoi.to_file(output_path, driver='GeoJSON')


# Get bounding box for Sentinel Hub
minx, miny, maxx, maxy = aoi.total_bounds
print(f"\nBounding Box (for Sentinel Hub):")
print(f"  Min Lon: {minx:.6f}")
print(f"  Min Lat: {miny:.6f}")
print(f"  Max Lon: {maxx:.6f}")
print(f"  Max Lat: {maxy:.6f}")


"""
Bounding Box (for Sentinel Hub):
  Min Lon: 13.231189
  Min Lat: 48.713301
  Max Lon: 13.982592
  Max Lat: 49.191423
"""






