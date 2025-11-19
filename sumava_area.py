"""
Šumava National Park Data Downloader - FIXED APPROACH
Uses alternative methods to get ONLY Šumava data
"""

import requests
import json
import geopandas as gpd
from pathlib import Path
import time

class SumavaDownloader:
    """Download Šumava NP data using multiple alternative approaches"""
    
    def __init__(self, output_dir="sumava_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # AOPK Open Data Portal
        self.aopk_opendata = "https://gis-aopkcr.opendata.arcgis.com"
        
        # AOPK REST Services
        self.aopk_base = "https://gis.nature.cz/arcgis/rest/services"
        
    def download_and_filter_arcgis(self, service_url, where_clause, name, filter_keyword='šumava'):
        """
        Download from ArcGIS and filter locally for Šumava only
        This solves the problem of ArcGIS queries returning all areas
        """
        params = {
            'where': where_clause,
            'outFields': '*',
            'f': 'geojson',
            'returnGeometry': 'true',
            'outSR': '4326'
        }
        
        query_url = f"{service_url}/query"
        
        print(f"\n📥 Downloading from: {name}")
        print(f"   URL: {query_url}")
        
        try:
            response = requests.get(query_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            if 'features' not in data or len(data['features']) == 0:
                print(f"   ⚠️ No features found")
                return None
            
            print(f"   📊 Retrieved {len(data['features'])} features")
            
            # Load as GeoDataFrame for filtering
            gdf = gpd.GeoDataFrame.from_features(data['features'])
            
            # Filter for Šumava
            print(f"   🔍 Filtering for '{filter_keyword}'...")
            
            # Search in all text columns
            mask = gdf.apply(
                lambda row: any(
                    filter_keyword.lower() in str(val).lower()
                    for val in row.values
                    if isinstance(val, str)
                ),
                axis=1
            )
            
            gdf_filtered = gdf[mask]
            
            if len(gdf_filtered) == 0:
                print(f"   ❌ No features matching '{filter_keyword}' found")
                print(f"   Available areas:")
                if 'nazev' in gdf.columns:
                    for area in gdf['nazev'].unique()[:10]:
                        print(f"      • {area}")
                return None
            
            print(f"   ✅ Filtered to {len(gdf_filtered)} Šumava feature(s)")
            
            if 'nazev' in gdf_filtered.columns:
                for area in gdf_filtered['nazev'].unique():
                    print(f"      • {area}")
            
            # Save
            output_path = self.output_dir / f"{name}.geojson"
            gdf_filtered.to_file(output_path, driver='GeoJSON')
            print(f"   💾 Saved: {output_path}")
            
            # Convert to other formats
            self.convert_formats(gdf_filtered, name)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def convert_formats(self, gdf, name):
        """Convert to multiple formats"""
        try:
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
            print(f"   ⚠️ Format conversion issue: {e}")
    
    def method_1_filtered_download(self):
        """
        METHOD 1: Download all protected areas, filter locally for Šumava
        This is the most reliable method
        """
        print("\n" + "="*70)
        print("METHOD 1: DOWNLOAD + LOCAL FILTERING")
        print("="*70)
        
        # National Parks
        print("\n1️⃣ National Parks...")
        np_url = f"{self.aopk_base}/UzemniOchrana/ChranUzemi/MapServer/0"
        self.download_and_filter_arcgis(np_url, "1=1", "sumava_np", 'šumava')
        
        time.sleep(1)
        
        # Protected Landscape Areas (CHKO)
        print("\n2️⃣ Protected Landscape Areas (CHKO)...")
        chko_url = f"{self.aopk_base}/UzemniOchrana/ChranUzemi/MapServer/1"
        self.download_and_filter_arcgis(chko_url, "1=1", "sumava_chko", 'šumava')
    
    def method_2_bbox_download(self):
        """
        METHOD 2: Use bounding box to limit download area
        Šumava bounding box (approximate): 48.6-49.2°N, 13.3-14.0°E
        """
        print("\n" + "="*70)
        print("METHOD 2: BOUNDING BOX FILTERING")
        print("="*70)
        
        # Šumava approximate bounding box
        bbox = "13.3,48.6,14.0,49.2"  # xmin,ymin,xmax,ymax
        
        print(f"\n📍 Using bbox: {bbox}")
        
        np_url = f"{self.aopk_base}/UzemniOchrana/ChranUzemi/MapServer/0"
        
        params = {
            'geometry': bbox,
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': '*',
            'f': 'geojson',
            'returnGeometry': 'true',
            'outSR': '4326'
        }
        
        try:
            response = requests.get(f"{np_url}/query", params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data and len(data['features']) > 0:
                gdf = gpd.GeoDataFrame.from_features(data['features'])
                
                # Still filter for Šumava in case bbox caught nearby areas
                mask = gdf.apply(
                    lambda row: 'šumava' in str(row).lower() or 'sumava' in str(row).lower(),
                    axis=1
                )
                gdf = gdf[mask]
                
                if len(gdf) > 0:
                    output_path = self.output_dir / "sumava_bbox_method.geojson"
                    gdf.to_file(output_path, driver='GeoJSON')
                    print(f"   ✅ Saved {len(gdf)} feature(s): {output_path}")
                    self.convert_formats(gdf, "sumava_bbox_method")
                else:
                    print("   ⚠️ No Šumava features in bbox")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def method_3_manual_coordinates(self):
        """
        METHOD 3: Create boundary from known coordinates
        Use this if download fails - creates approximate boundary
        """
        print("\n" + "="*70)
        print("METHOD 3: MANUAL BOUNDARY FROM COORDINATES")
        print("="*70)
        
        from shapely.geometry import Polygon
        
        # Approximate Šumava NP boundary (simplified)
        coords = [
            [13.32, 48.93],
            [13.85, 48.93],
            [13.85, 49.15],
            [13.32, 49.15],
            [13.32, 48.93]
        ]
        
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame(
            {'name': ['Šumava NP (Approximate)'], 'source': ['Manual']},
            geometry=[polygon],
            crs='EPSG:4326'
        )
        
        output_path = self.output_dir / "sumava_manual_boundary.geojson"
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"   ✅ Created approximate boundary: {output_path}")
        print(f"   ⚠️ This is a simplified rectangle - not official boundary!")
        self.convert_formats(gdf, "sumava_manual_boundary")
    
    def method_4_natura2000(self):
        """
        METHOD 4: Try Natura 2000 sites (EU protected areas)
        Šumava is also a Natura 2000 site
        """
        print("\n" + "="*70)
        print("METHOD 4: NATURA 2000 SITES")
        print("="*70)
        
        # Try AOPK Natura 2000 service
        natura_url = f"{self.aopk_base}/UzemniOchrana/Natura2000/MapServer"
        
        print("\n🔍 Exploring Natura 2000 layers...")
        
        try:
            response = requests.get(f"{natura_url}?f=json", timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'layers' in data:
                print(f"   Found {len(data['layers'])} layers")
                
                for layer in data['layers']:
                    layer_id = layer['id']
                    layer_name = layer['name']
                    
                    if 'šumava' in layer_name.lower() or layer_id in [0, 1, 2]:
                        print(f"\n   📥 Trying layer {layer_id}: {layer_name}")
                        self.download_and_filter_arcgis(
                            f"{natura_url}/{layer_id}",
                            "1=1",
                            f"sumava_natura_{layer_id}",
                            'šumava'
                        )
                        time.sleep(1)
        
        except Exception as e:
            print(f"   ⚠️ Natura 2000 service error: {e}")
    
    def create_combined(self):
        """Combine all successfully downloaded Šumava files"""
        print("\n" + "="*70)
        print("CREATING COMBINED ŠUMAVA DATASET")
        print("="*70)
        
        geojson_files = list(self.output_dir.glob("*.geojson"))
        
        if not geojson_files:
            print("   ⚠️ No GeoJSON files found to combine")
            return
        
        gdfs = []
        for file in geojson_files:
            if 'combined' not in file.name:
                try:
                    gdf = gpd.read_file(file)
                    if not gdf.empty:
                        gdf['source_file'] = file.stem
                        gdfs.append(gdf)
                except:
                    pass
        
        if gdfs:
            combined = gpd.pd.concat(gdfs, ignore_index=True)
            
            output_path = self.output_dir / "SUMAVA_COMBINED.geojson"
            combined.to_file(output_path, driver='GeoJSON')
            print(f"   ✅ Combined {len(gdfs)} datasets: {output_path}")
            print(f"   📊 Total features: {len(combined)}")
            
            self.convert_formats(combined, "SUMAVA_COMBINED")
    
    def run_all_methods(self):
        """Try all methods to get Šumava data"""
        print("\n" + "="*70)
        print("ŠUMAVA DATA DOWNLOADER - ALL METHODS")
        print("="*70)
        print("Trying multiple approaches to download ONLY Šumava data")
        print("="*70)
        
        # Try each method
        self.method_1_filtered_download()
        time.sleep(2)
        
        self.method_2_bbox_download()
        time.sleep(2)
        
        self.method_4_natura2000()
        time.sleep(2)
        
        # Combine results
        self.create_combined()
        
        # If nothing worked, create manual boundary
        if not list(self.output_dir.glob("sumava*.geojson")):
            print("\n⚠️ All download methods failed!")
            print("Creating manual approximate boundary as fallback...")
            self.method_3_manual_coordinates()
        
        self.print_summary()
    
    def print_summary(self):
        """Print summary of downloaded files"""
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        
        files = list(self.output_dir.glob("*"))
        
        if not files:
            print("\n❌ No files downloaded!")
            print("\nRECOMMENDATION:")
            print("Contact Šumava NP directly:")
            print("  Email: gis@npsumava.cz")
            print("  Website: https://geoportal.npsumava.cz")
            return
        
        print(f"\n📁 Output: {self.output_dir.absolute()}\n")
        
        for fmt in ['geojson', 'shp', 'gpkg', 'kml']:
            files_fmt = list(self.output_dir.glob(f"*.{fmt}"))
            if files_fmt:
                print(f"{fmt.upper()}: {len(files_fmt)} files")
                for f in sorted(files_fmt):
                    size = f.stat().st_size / 1024
                    print(f"  • {f.name} ({size:.1f} KB)")
        
        print("\n✅ READY FOR SENTINEL-2 ANALYSIS")


if __name__ == "__main__":
    downloader = SumavaDownloader(output_dir="sumava_data")
    downloader.run_all_methods()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
Use the downloaded files for:
✅ Sentinel-2 imagery clipping (AOI)
✅ Training/validation zones
✅ Study area boundaries

Coordinates in WGS84 (EPSG:4326) - compatible with all satellite platforms.

If download failed, contact Šumava NP directly:
📧 gis@npsumava.cz
🌐 https://geoportal.npsumava.cz
""")