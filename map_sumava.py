import geopandas as gpd
import matplotlib.pyplot as plt

# Load your data
gdf = gpd.read_file("sumava_data/SUMAVA_COMBINED.geojson")

# Quick plot
gdf.plot(figsize=(10, 8))
plt.title("Šumava Data")
plt.show()

# Check what you got
print(gdf.head())
print(f"Total features: {len(gdf)}")
print(f"Area names: {gdf['nazev'].unique()}")