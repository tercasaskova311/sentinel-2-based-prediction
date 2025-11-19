import geopandas as gpd
import matplotlib.pyplot as plt

# Load your data
cze_2 = gpd.read_file("sumava_zones_2.geojson")
cze_2.plot()
