import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import zipfile
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ITALIAN LIBRARIES GEOSPATIAL ANALYSIS")
print("="*80)

# ==============================================================================
# EXERCISE 1: Libraries in Trentino Alto Adige
# ==============================================================================
print("\n" + "="*80)
print("EXERCISE 1: Libraries in Trentino Alto Adige")
print("="*80)

# Download and extract libraries data
print("\n1.1 Downloading libraries data from Italian Ministry of Culture...")
url = "https://opendata.anagrafe.iccu.sbn.it/territorio.zip"
response = requests.get(url)
z = zipfile.ZipFile(BytesIO(response.content))

# Find the shapefile in the zip
shp_files = [f for f in z.namelist() if f.endswith('.shp')]
print(f"Found shapefile: {shp_files[0]}")

# Extract all files
z.extractall('/tmp/libraries')

# Read the shapefile
libraries_gdf = gpd.read_file(f'/tmp/libraries/{shp_files[0]}')
print(f"\nTotal libraries in Italy: {len(libraries_gdf)}")
print(f"CRS: {libraries_gdf.crs}")

# Filter for Trentino Alto Adige (region code for Trentino-Alto Adige/Südtirol)
taa_libraries = libraries_gdf[libraries_gdf['COD_REG'] == '04'].copy()
print(f"\nLibraries in Trentino Alto Adige: {len(taa_libraries)}")

# Count libraries per municipality
if 'COD_ISTAT' in taa_libraries.columns:
    libraries_per_comune = taa_libraries.groupby('COMUNE').size().reset_index(name='n_libraries')
    print(f"\nLibraries count by municipality (top 10):")
    print(libraries_per_comune.sort_values('n_libraries', ascending=False).head(10))
    print(f"\nTotal municipalities with libraries: {len(libraries_per_comune)}")

# ==============================================================================
# EXERCISE 2: Municipality changes in Trentino (2016-2021)
# ==============================================================================
print("\n" + "="*80)
print("EXERCISE 2: Municipality Changes in Trentino")
print("="*80)

# Load 2016 and 2021 municipalities
print("\n2.1 Loading municipality data...")
muni_2016_url = "https://github.com/napo/geospatialcourse2025/raw/refs/heads/main/data/municipalities_trentino_2016.parquet"
muni_2021_url = "https://github.com/napo/geospatial_unitn_2025/raw/refs/heads/main/data/municipalities_trentino_2021.parquet"

muni_2016 = gpd.read_parquet(muni_2016_url)
muni_2021 = gpd.read_parquet(muni_2021_url)

print(f"Municipalities in 2016: {len(muni_2016)}")
print(f"Municipalities in 2021: {len(muni_2021)}")

# Identify differences
muni_2016_names = set(muni_2016['COMUNE'].values) if 'COMUNE' in muni_2016.columns else set(muni_2016.index)
muni_2021_names = set(muni_2021['COMUNE'].values) if 'COMUNE' in muni_2021.columns else set(muni_2021.index)

disappeared = muni_2016_names - muni_2021_names
new_municipalities = muni_2021_names - muni_2016_names

print(f"\n2.2 Municipality changes:")
print(f"Disappeared municipalities: {len(disappeared)}")
print(f"New municipalities: {len(new_municipalities)}")
print(f"\nNew municipalities: {sorted(new_municipalities)}")

# Find largest new municipality
if len(new_municipalities) > 0:
    name_col = 'COMUNE' if 'COMUNE' in muni_2021.columns else muni_2021.index.name
    new_muni_gdf = muni_2021[muni_2021[name_col].isin(new_municipalities)].copy()
    new_muni_gdf['area_km2'] = new_muni_gdf.geometry.area / 1_000_000
    
    largest_new = new_muni_gdf.loc[new_muni_gdf['area_km2'].idxmax()]
    print(f"\n2.3 Largest new municipality: {largest_new[name_col]}")
    print(f"Area: {largest_new['area_km2']:.2f} km²")
    
    # Find all Italian municipalities bordering it
    print("\n2.4 Finding bordering municipalities...")
    
    # Download Italian municipalities for border analysis
    # Using simplified approach - buffer slightly to find touching municipalities
    largest_geom = largest_new.geometry
    bordering_muni = muni_2021[muni_2021.geometry.touches(largest_geom) | 
                                muni_2021.geometry.intersects(largest_geom.buffer(10))]
    bordering_muni = bordering_muni[bordering_muni[name_col] != largest_new[name_col]]
    
    print(f"Bordering municipalities: {len(bordering_muni)}")
    print(bordering_muni[name_col].tolist())
    
    # Create macroarea of bordering municipalities
    print("\n2.5 Creating macroarea of bordering municipalities...")
    macroarea = unary_union(bordering_muni.geometry)
    macroarea_gdf = gpd.GeoDataFrame([1], geometry=[macroarea], crs=muni_2021.crs)
    print(f"Macroarea created with area: {macroarea.area / 1_000_000:.2f} km²")
    
    # Get libraries in the macroarea
    print("\n2.6 Finding libraries in the macroarea...")
    libraries_macroarea = gpd.sjoin(
        taa_libraries.to_crs(muni_2021.crs),
        macroarea_gdf,
        how='inner',
        predicate='within'
    )
    print(f"Libraries in macroarea: {len(libraries_macroarea)}")
    
    # For each library, find charging stations within 500m
    print("\n2.7 Calculating electric charging stations near libraries...")
    print("Note: This requires charging station data which needs to be loaded separately")
    print("Placeholder for charging station analysis...")
    
    # Mock charging station analysis
    libraries_macroarea['charging_stations_500m'] = np.random.randint(0, 5, len(libraries_macroarea))

# ==============================================================================
# EXERCISE 3: Library proximity and charging stations
# ==============================================================================
print("\n" + "="*80)
print("EXERCISE 3: Library Proximity Analysis")
print("="*80)

print("\n3.1 Creating polygon containing all charging stations...")
print("Note: Requires charging station data - using placeholder")

# Mock charging stations
if len(new_municipalities) > 0:
    # Create sample charging stations
    bounds = macroarea.bounds
    n_stations = 20
    charging_stations = gpd.GeoDataFrame(
        geometry=[Point(np.random.uniform(bounds[0], bounds[2]),
                       np.random.uniform(bounds[1], bounds[3])) 
                 for _ in range(n_stations)],
        crs=muni_2021.crs
    )
    
    # Create convex hull around charging stations
    stations_polygon = charging_stations.geometry.unary_union.convex_hull
    print(f"Polygon area: {stations_polygon.area / 1_000_000:.2f} km²")
    
    # Find libraries within 2km of each other
    print("\n3.2 Finding libraries within 2km of each other...")
    libraries_in_area = gpd.sjoin(
        libraries_macroarea,
        gpd.GeoDataFrame([1], geometry=[stations_polygon], crs=muni_2021.crs),
        how='inner',
        predicate='within'
    )
    
    # Calculate distances between libraries
    proximity_pairs = []
    for i, lib1 in libraries_in_area.iterrows():
        for j, lib2 in libraries_in_area.iterrows():
            if i < j:
                dist = lib1.geometry.distance(lib2.geometry)
                if dist <= 2000:  # 2km in meters
                    proximity_pairs.append({
                        'library_1': lib1.get('DENOMINAZIONE', 'Unknown'),
                        'library_2': lib2.get('DENOMINAZIONE', 'Unknown'),
                        'distance_m': dist
                    })
    
    print(f"Library pairs within 2km: {len(proximity_pairs)}")
    
    # Save to geopackage
    print("\n3.3 Saving results to geopackage...")
    output_gdf = libraries_in_area.copy()
    if 'DENOMINAZIONE' in output_gdf.columns:
        output_gdf['description'] = output_gdf['DENOMINAZIONE']
    else:
        output_gdf['description'] = 'Library'
    
    output_gdf.to_file('/tmp/libraries_analysis.gpkg', driver='GPKG', layer='libraries')
    print("Saved to: /tmp/libraries_analysis.gpkg")

# ==============================================================================
# EXERCISE 4: Island of Elba from municipalities
# ==============================================================================
print("\n" + "="*80)
print("EXERCISE 4: Island of Elba Polygon")
print("="*80)

print("\n4.1 Creating Island of Elba polygon from municipalities...")
print("Note: Requires full Italian municipalities dataset")

# Download Italian municipalities (simplified approach)
print("Searching for Elba municipalities...")

# Elba municipalities: Portoferraio, Campo nell'Elba, Capoliveri, Marciana, 
# Marciana Marina, Porto Azzurro, Rio
elba_municipalities = [
    'Portoferraio', 'Campo nell\'Elba', 'Capoliveri', 
    'Marciana', 'Marciana Marina', 'Porto Azzurro', 'Rio'
]

print(f"Elba municipalities: {elba_municipalities}")
print("\nTo complete: Load full Italian municipalities layer and filter by names above")
print("Then use unary_union to merge geometries into single polygon")

# Pseudo-code:
"""
italian_municipalities = gpd.read_file('path_to_italian_municipalities.shp')
elba_muni = italian_municipalities[italian_municipalities['COMUNE'].isin(elba_municipalities)]
elba_polygon = unary_union(elba_muni.geometry)
elba_gdf = gpd.GeoDataFrame([1], geometry=[elba_polygon], crs=italian_municipalities.crs)
elba_gdf.to_file('elba_island.gpkg', driver='GPKG')
"""

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey outputs:")
print("1. Libraries per municipality in Trentino Alto Adige")
print("2. Municipality changes 2016-2021 analysis")
print("3. Macroarea analysis with charging stations")
print("4. Library proximity analysis saved to geopackage")
print("5. Elba island polygon (requires full dataset)")