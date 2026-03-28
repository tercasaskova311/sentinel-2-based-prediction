import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import requests


@dataclass
class ForestDisturbanceLayer:
    """
    A class representing an ArcGIS layer.
    Args:
        id: the ID of the layer
        title: the title of the layer
        url: the URL of the layer
        type: the type of the layer
    """
    id: str
    title: str
    url: str
    type: str


class ForestDisturbancePolygonsExtractor:
    """
    A class for extracting layers from an ArcGIS map.
    Fetches the webmap ID from the app config and then fetches the layers from the webmap.
    Then downloads the layers features and saves them as GPKG files.
    """
    CONTENT_API = "https://{portal}/sharing/rest/content/items/{id}/data"

    def __init__(self, portal: str):
        """
        Initialize the ArcGISExtractor.
        Args:
            portal: the portal of the ArcGIS map
        """
        # e.g. "czechglobe.maps.arcgis.com" (no protocol)
        self.portal = portal

    def _fetch(self, url: str, params: Optional[dict] = None, timeout: int = 30) -> dict[str, Any]:
        """
        Fetch data from the ArcGIS map.
        Args:
            url: the URL to fetch data from
            params: the parameters to pass to the URL
            timeout: the timeout for the request
        Returns:
            dict: a dictionary of the data
        """
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error parsing JSON from {url}: {e}")
            return {}

    def _content_url(self, item_id: str) -> str:
        """
        Formats the content URL for the item.
        Args:
            item_id: the ID of the item
        Returns:
            str: the content URL
        """
        return self.CONTENT_API.format(portal=self.portal, id=item_id)

    def _get_webmap_id(self, app_id: str) -> Optional[str]:
        """
        Fetches the webmap ID from the app config.
        Args:
            app_id: the ID of the app
        Returns:
            str: the webmap ID
        """
        print(f"Fetching app config for: {app_id}")
        data = self._fetch(self._content_url(app_id), params={"f": "json"}, timeout=15)

        try:
            webmap_id = data.get("values", {}).get("webmap")
            source_id = data.get("source", None)

            if not webmap_id:
                print(f"Webmap ID not found in app config for {app_id}")
                return None
            print(f"Found webmap ID: {webmap_id}")
            return webmap_id or source_id
        except Exception as e:
            print(f"Error extracting webmap ID from app config for {app_id}: {e}")
            return None

    def _get_layers(self, webmap_id: str) -> list[ForestDisturbanceLayer]:
        """
        Fetches the layers from the webmap.
        Args:
            webmap_id: the ID of the webmap
        Returns:
            list[ForestDisturbanceLayer]: a list of the layers
        """
        print(f"Fetching webmap config for: {webmap_id}")
        data = self._fetch(self._content_url(webmap_id), params={"f": "json"}, timeout=15)

        try:
            operational_layers = data.get("operationalLayers", [])
            layers_info: list[ForestDisturbanceLayer] = []
            for layer in operational_layers:
                layer_info = ForestDisturbanceLayer(
                    id=layer.get("id"),
                    title=layer.get("title"),
                    url=layer.get("url"),
                    type=layer.get("layerType"),
                )
                layers_info.append(layer_info)

                # sub layers (for group layers)
                if "layers" in layer:
                    for sub in layer.get("layers", []):
                        sub_info = ForestDisturbanceLayer(
                            id=sub.get("id"),
                            title=sub.get("title"),
                            url=sub.get("url"),
                            type=sub.get("layerType"),
                        )
                        layers_info.append(sub_info)

            print(f"Extracted {len(layers_info)} layers from webmap config")
            return layers_info
        except Exception as e:
            print(f"Error extracting layers from webmap config for {webmap_id}: {e}")
            return []

    def get_layers(self, app_id: str) -> list[ForestDisturbanceLayer]:
        """
        Fetches the layers from the app.
        Args:
            app_id: the ID of the app
        Returns:
            list[ForestDisturbanceLayer]: a list of the layers
        """
        webmap_id = self._get_webmap_id(app_id)
        if not webmap_id:
            print(f"Cannot get layers without webmap ID for app {app_id}")
            return []
        return self._get_layers(webmap_id)

    def download_layer(self, url: str, crs: int = 32633) -> dict[str, Any]:
        """
        Downloads the layer from the ArcGIS map.
        Args:
            url: the URL of the layer
            crs: the CRS of the layer
        Returns:
            dict: a dictionary of the features
        """
        count_r = self._fetch(f"{url}/query?where=1=1&returnCountOnly=true", params={"f": "json"}, timeout=15)
        total = count_r.get("count", 0)
        print(f"Total features to download: {total}")

        all_features: list[dict[str, Any]] = []
        offset = 0
        batch_size = 1000

        while offset < total:
            params = {
                "where": "1=1",
                "outFields": "*",
                "f": "geojson",
                "returnGeometry": True,
                "outSR": crs,
                "resultOffset": offset,
                "resultRecordCount": batch_size,
            }

            try:
                data = self._fetch(f"{url}/query", params=params, timeout=15)
                features = data.get("features", [])

                if not features:
                    print(f"No more features returned at offset {offset}. Ending download.")
                    break

                all_features.extend(features)
                offset += len(features)
                # print(f"Downloaded {offset}/{total} features...")
            except requests.exceptions.RequestException as e:
                print(f"Request error at offset {offset}: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"JSON decode error at offset {offset}: {e}")
                break

        print(f"Finished downloading. Total features downloaded: {len(all_features)}")
        return {"type": "FeatureCollection", "features": all_features}

    def download_and_save(
        self,
        url: str,
        save_path: str | Path,
        driver: str = "GPKG",
        crs: int = 32633,
        verbose: bool = True,
    ):
        """
        Downloads the layer from the ArcGIS map and saves it as a GPKG file.
        Args:
            url: the URL of the layer
            save_path: the path to save the layer
            driver: the driver to save the layer
            crs: the CRS of the layer
            verbose: whether to print verbose output
        Returns:
            gpd.GeoDataFrame: the GeoDataFrame of the features
        """
        try:
            geojson_data = self.download_layer(url, crs)

            if not geojson_data.get("features"):
                if verbose:
                    print(f"No features downloaded from {url}. Skipping save.")
                return None

            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs=f"EPSG:{crs}")

            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)

            gdf.to_file(save_path, driver=driver)
            if verbose:
                print(f"Saved downloaded data to {save_path}")

            size = save_path.stat().st_size / 1e6
            print(f"File size: {size:.2f} MB")
            return gdf
        except Exception as e:
            print(f"Error downloading or saving layer from {url}: {e}")
            return None

