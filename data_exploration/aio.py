import geopandas as gpd
import requests
from pathlib import Path
from shapely.geometry import shape
from shapely.validation import make_valid
import matplotlib.pyplot as plt
import time

class SumavaDownloader:
    """Download Šumava NP & CHKO via Nominatim (robust)"""

    def __init__(self, output_dir="sumava_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _download_nominatim(self, query):
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "polygon_geojson": 1,
            "limit": 1
        }
        headers = {"User-Agent": "SumavaDownloader/1.0"}
        time.sleep(1)  # Respect rate limit
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        geom = make_valid(shape(results[0]['geojson']))
        gdf = gpd.GeoDataFrame([{
            "name": results[0]["display_name"],
            "osm_id": results[0]["osm_id"],
            "osm_type": results[0]["osm_type"]
        }], geometry=[geom], crs="EPSG:4326")
        return gdf

    def download_all(self):
        print("Downloading Šumava NP...")
        np_gdf = self._download_nominatim("Národní park Šumava")
        if np_gdf is not None:
            np_gdf.to_file(self.output_dir / "sumava_np.geojson", driver="GeoJSON")
            print("Šumava NP downloaded")

        return np_gdf

    def preview_map(self):
        geojsons = list(self.output_dir.glob("*.geojson"))
        if not geojsons:
            print("No GeoJSON files to preview.")
            return
        fig, ax = plt.subplots(figsize=(12,10))
        for f in geojsons:
            gpd.read_file(f).plot(ax=ax, alpha=0.5, edgecolor='red', linewidth=2, label=f.stem)
        ax.set_title("Šumava NP area")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    downloader = SumavaDownloader()
    np_gdf = downloader.download_all()
    downloader.preview_map()
