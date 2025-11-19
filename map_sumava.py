import geopandas as gpd
import matplotlib.pyplot as plt

# Load your data
cze_zonace = gpd.read_file("sumava_data/Zonace_velkoplošných_zvláště_chráněných_území.geojson")

print(cze_zonace.groupby('NAZEV')['KOD'].unique())
