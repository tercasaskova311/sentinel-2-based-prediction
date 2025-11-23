import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import os
# Load your data
cze_2 = gpd.read_file("sumava_zones_2.geojson")

print(cze_2.info())
print(cze_2.columns.tolist())
print(cze_2.head())


print("\n=== Unique values in potential name columns ===")
for col in cze_2.columns:
    if cze_2[col].dtype == 'object':  # Text columns
        unique_vals = cze_2[col].unique()
        if len(unique_vals) < 20:  # Only show if not too many values
            print(f"\n{col}: {unique_vals}")

filtered_json = cze_2[cze_2['KAT'] == "NP"]
print(filtered_json)

aoi = filtered_json.dissolve()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original data
cze_2.plot(ax=axes[0], column=cze_2.columns[0], legend=True, cmap='Set3', 
         edgecolor='black', alpha=0.7)
axes[0].set_title('Original GeoJSON')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

# Filtered/dissolved AOI
aoi.plot(ax=axes[1], facecolor='lightgreen', edgecolor='darkgreen', 
         linewidth=2, alpha=0.5)
axes[1].set_title('Final AOI for Šumava')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('sumava_aoi_check.png', dpi=150)
plt.show()

aoi_proj = aoi.to_crs(32633)  # UTM zone 33N
area_km2 = aoi_proj.geometry.area.sum() / 1e6
aoi_proj.to_file("sumava_aoi_clean_proj.geojson", driver="GeoJSON")
print(f"📏 AOI Area: {area_km2:.2f} km²")

# Save the cleaned AOI
output_path = "sumava_aoi_clean.geojson"
aoi.to_file(output_path, driver='GeoJSON')
print(f"\n✅ Saved cleaned AOI to: {output_path}")

# Print AOI stats
print("\n=== AOI Statistics ===")
print(f"CRS: {aoi.crs}")
print(f"Bounds: {aoi.total_bounds}")

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






