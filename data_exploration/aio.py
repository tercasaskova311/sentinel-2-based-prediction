import requests
import geopandas as gpd
from pathlib import Path
import json
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import os

class SumavaOSMDownloader: #Download Šumava NP data from OpenStreetMap using Overpass API
    
    def __init__(self, output_dir="sumava_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.overpass_url = "https://overpass-api.de/api/interpreter"
    
    def download_from_osm(self):
        """
        Download Šumava National Park from OpenStreetMap
        OSM has very accurate protected area boundaries
        """
        print("\n" + "="*70)
        print("DOWNLOADING FROM OPENSTREETMAP")
        print("="*70)
        print("Using Overpass API to query for Šumava National Park...")
        
        # Overpass QL query for Šumava National Park
        # This searches for boundaries with specific tags
        overpass_query = """
        [out:json][timeout:60];
        (
          // Search for Šumava National Park relation
          relation["boundary"="protected_area"]["protect_class"="2"]["name:cs"~"Šumava",i];
          relation["boundary"="national_park"]["name:cs"~"Šumava",i];
          relation["boundary"="protected_area"]["name"~"Šumava",i];
          relation["leisure"="nature_reserve"]["name"~"Šumava",i];
          // Also search by wikidata ID (most reliable)
          relation["wikidata"="Q213460"];
        );
        out geom;
        """
        
        print("\n📥 Querying Overpass API...")
        print("   This may take 10-30 seconds...")
        
        try:
            response = requests.post(
                self.overpass_url,
                data={'data': overpass_query},
                timeout=90
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'elements' not in data or len(data['elements']) == 0:
                print("\n❌ No results from OSM")
                return self.try_osm_nominatim()
            
            print(f"\n✅ Found {len(data['elements'])} OSM element(s)")
            
            # Convert OSM JSON to GeoDataFrame
            features = []
            for element in data['elements']:
                if element['type'] == 'relation' and 'members' in element:
                    # Extract geometry from relation members
                    geometry = self.osm_relation_to_geometry(element)
                    if geometry:
                        properties = element.get('tags', {})
                        properties['osm_id'] = element['id']
                        properties['osm_type'] = element['type']
                        features.append({
                            'type': 'Feature',
                            'geometry': geometry,
                            'properties': properties
                        })
            
            if not features:
                print("\n⚠️ Could not extract geometries from OSM")
                return self.try_osm_nominatim()
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')
            
            # Show what we got
            print(f"\n📋 Downloaded boundary:")
            if 'name' in gdf.columns:
                print(f"   Name: {gdf['name'].iloc[0]}")
            if 'name:cs' in gdf.columns:
                print(f"   Czech name: {gdf['name:cs'].iloc[0]}")
            
            # Calculate area
            gdf_utm = gdf.to_crs('EPSG:32633')
            area_km2 = gdf_utm.geometry.area.sum() / 1_000_000
            print(f"   Area: {area_km2:.2f} km²")
            print(f"   Expected: ~680 km²")
            
            if 600 < area_km2 < 750:
                print(f"   ✅ Area matches expected Šumava NP size!")
            else:
                print(f"   ⚠️ Area differs from expected - may need verification")
            
            # Save
            self.save_all_formats(gdf, "sumava_np_osm")
            
            return gdf
            
        except requests.exceptions.Timeout:
            print("\n⚠️ Overpass API timeout - trying alternative method...")
            return self.try_osm_nominatim()
        
        except Exception as e:
            print(f"\n❌ OSM download error: {e}")
            return self.try_osm_nominatim()
    
    def osm_relation_to_geometry(self, relation):
        """Convert OSM relation to geometry"""
        try:
            from shapely.geometry import Polygon, MultiPolygon, mapping
            from shapely.ops import unary_union
            
            if 'geometry' in relation:
                # Overpass returns geometry in a specific format
                members = relation['members']
                
                # Collect all ways that form the outer boundary
                outer_ways = []
                for member in members:
                    if member.get('role') == 'outer' and 'geometry' in member:
                        coords = [(node['lon'], node['lat']) for node in member['geometry']]
                        if len(coords) >= 3:
                            outer_ways.append(coords)
                
                if outer_ways:
                    # Try to create polygon
                    if len(outer_ways) == 1:
                        return mapping(Polygon(outer_ways[0]))
                    else:
                        # Multiple outer ways - try to merge
                        polygons = [Polygon(way) for way in outer_ways if len(way) >= 3]
                        if polygons:
                            merged = unary_union(polygons)
                            return mapping(merged)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Geometry conversion error: {e}")
            return None
    
    def try_osm_nominatim(self):
        """
        Alternative: Use Nominatim to get boundary
        """
        print("\n" + "="*70)
        print("ALTERNATIVE: NOMINATIM GEOCODING")
        print("="*70)
        
        # Nominatim search for Šumava NP
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        
        params = {
            'q': 'Národní park Šumava',
            'format': 'json',
            'polygon_geojson': 1,
            'limit': 5
        }
        
        headers = {
            'User-Agent': 'Sumava-Research-Project/1.0'
        }
        
        print("\n📥 Querying Nominatim...")
        
        try:
            time.sleep(1)  # Respect Nominatim rate limits
            response = requests.get(nominatim_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            results = response.json()
            
            if not results:
                print("\n❌ No results from Nominatim")
                return self.download_from_geojson_io()
            
            print(f"\n✅ Found {len(results)} result(s)")
            
            # Find the best match (national park)
            best_result = None
            for result in results:
                if 'geojson' in result:
                    name = result.get('display_name', '')
                    osm_type = result.get('osm_type', '')
                    print(f"   • {name}")
                    
                    if 'national park' in name.lower() or 'národní park' in name.lower():
                        best_result = result
                        break
            
            if not best_result:
                best_result = results[0]
            
            # Convert to GeoDataFrame
            feature = {
                'type': 'Feature',
                'geometry': best_result['geojson'],
                'properties': {
                    'name': best_result.get('display_name', 'Šumava NP'),
                    'osm_id': best_result.get('osm_id'),
                    'osm_type': best_result.get('osm_type'),
                    'source': 'Nominatim'
                }
            }
            
            gdf = gpd.GeoDataFrame.from_features([feature], crs='EPSG:4326')
            
            # Calculate area
            gdf_utm = gdf.to_crs('EPSG:32633')
            area_km2 = gdf_utm.geometry.area.sum() / 1_000_000
            print(f"\n📏 Area: {area_km2:.2f} km²")
            
            self.save_all_formats(gdf, "sumava_np_nominatim")
            
            return gdf
            
        except Exception as e:
            print(f"\n❌ Nominatim error: {e}")
            return self.download_from_geojson_io()
    
    def download_from_geojson_io(self):
        """
        Use geojson.io / GitHub datasets as fallback
        """
        print("\n" + "="*70)
        print("FALLBACK: GITHUB GEOJSON DATASETS")
        print("="*70)
        
        # Try various GitHub sources for Czech protected areas
        github_sources = [
            # Natural Earth Data - protected areas
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_parks_and_protected_lands.geojson",
        ]
        
        for url in github_sources:
            try:
                print(f"\n📥 Trying: {url.split('/')[-1]}")
                gdf = gpd.read_file(url)
                
                # Filter for Šumava
                mask = gdf.apply(
                    lambda row: any('umava' in str(val).lower() for val in row.values if isinstance(val, str)),
                    axis=1
                )
                
                gdf_sumava = gdf[mask]
                
                if len(gdf_sumava) > 0:
                    print(f"   ✅ Found Šumava in dataset!")
                    self.save_all_formats(gdf_sumava, "sumava_np_github")
                    return gdf_sumava
                    
            except Exception as e:
                print(f"   ⚠️ Failed: {e}")
                continue
        
        print("\n❌ All GitHub sources failed")
        return None
    
    def download_sumava_chko(self):
        """Download CHKO Šumava from OSM"""
        print("\n" + "="*70)
        print("DOWNLOADING CHKO ŠUMAVA FROM OSM")
        print("="*70)
        
        overpass_query = """
        [out:json][timeout:60];
        (
          relation["boundary"="protected_area"]["protect_class"="5"]["name:cs"~"Šumava",i];
          relation["boundary"="protected_area"]["name"~"Šumava",i]["designation"~"landscape",i];
        );
        out geom;
        """
        
        try:
            response = requests.post(
                self.overpass_url,
                data={'data': overpass_query},
                timeout=90
            )
            
            data = response.json()
            
            if data.get('elements'):
                print(f"\n✅ Found {len(data['elements'])} CHKO element(s)")
                # Process similar to NP
                features = []
                for element in data['elements']:
                    if element['type'] == 'relation':
                        geometry = self.osm_relation_to_geometry(element)
                        if geometry:
                            properties = element.get('tags', {})
                            properties['osm_id'] = element['id']
                            features.append({
                                'type': 'Feature',
                                'geometry': geometry,
                                'properties': properties
                            })
                
                if features:
                    gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')
                    self.save_all_formats(gdf, "sumava_chko_osm")
                    return gdf
            else:
                print("\n⚠️ No CHKO found in OSM")
                
        except Exception as e:
            print(f"\n⚠️ CHKO download failed: {e}")
        
        return None
    
    def save_all_formats(self, gdf, name):
        """Save in multiple formats"""
        try:
            # GeoJSON
            geojson_path = self.output_dir / f"{name}.geojson"
            gdf.to_file(geojson_path, driver='GeoJSON')
            print(f"\n💾 Saved formats:")
            print(f"   📄 GeoJSON: {geojson_path}")
            
            # Shapefile
            shp_path = self.output_dir / f"{name}.shp"
            gdf.to_file(shp_path, driver='ESRI Shapefile')
            print(f"   📄 Shapefile: {shp_path}")
            
            # GeoPackage
            gpkg_path = self.output_dir / f"{name}.gpkg"
            gdf.to_file(gpkg_path, driver='GPKG')
            print(f"   📦 GeoPackage: {gpkg_path}")
            
            # KML
            kml_path = self.output_dir / f"{name}.kml"
            gdf.to_file(kml_path, driver='KML')
            print(f"   🌍 KML: {kml_path}")
            
        except Exception as e:
            print(f"   ⚠️ Format conversion error: {e}")
    
    def create_visualization(self):
        """Create map preview"""
        import matplotlib.pyplot as plt
        
        geojson_files = [f for f in self.output_dir.glob("*.geojson") 
                        if 'approximate' not in f.name]
        
        if not geojson_files:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        for file in geojson_files:
            gdf = gpd.read_file(file)
            gdf.plot(ax=ax, alpha=0.5, edgecolor='red', linewidth=2, 
                    label=file.stem)
        
        ax.set_title('Šumava National Park - OpenStreetMap Data', fontsize=16)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        map_path = self.output_dir / "sumava_osm_preview.png"
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        print(f"\n🗺️  Map saved: {map_path}")
        
        plt.show()
    
    def run(self):
        """Main execution"""
        print("\n" + "="*70)
        print("ŠUMAVA DOWNLOADER - OPENSTREETMAP METHOD")
        print("="*70)
        print("Most reliable source for Czech protected areas")
        print("="*70)
        
        # Download NP
        np_gdf = self.download_from_osm()
        
        if np_gdf is not None:
            # Try CHKO too
            time.sleep(2)
            chko_gdf = self.download_sumava_chko()
            
            # Create visualization
            try:
                self.create_visualization()
            except:
                print("\n Could not create visualization")
            
            print("   → sumava_np_osm.geojson")

if __name__ == "__main__":
    downloader = SumavaOSMDownloader(output_dir="sumava_data")
    downloader.run()

# Load your data
cze_2 = gpd.read_file("sumava_zones_2.geojson")
filtered_json = cze_2[cze_2['KAT'] == "NP"]
aoi = filtered_json.dissolve()
aoi_proj = aoi.to_crs(32633)  # UTM zone 33N
area_km2 = aoi_proj.geometry.area.sum() / 1e6
aoi_proj.to_file("sumava_aoi_clean_proj.geojson", driver="GeoJSON")
print(f" AOI Area: {area_km2:.2f} km²")

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