from __future__ import annotations

import json
import os
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

from . import config
from .features import build_features, build_s2_composite, compute_slope, rasterize_disturbance_polygons
from .processing import get_stable_mask

MONTHS = config.MONTHS
S2_BANDS = config.S2_BANDS


def get_gpkg_paths(data_dir: str = "data/sumava-czechglobe/"):
    """
    Get the paths of the CzechGlobe deforestation files.
    Args:
        data_dir: the directory containing the deforestation files
    Returns:
        list: the paths of the deforestation files
    """
    files = os.listdir(data_dir)
    print("Downloaded files:")
    for f in sorted(files):
        print(f" {f}")
    return files


def load_deforestation_files(files: list, data_dir: str = Path("data/sumava-czechglobe/")) -> gpd.GeoDataFrame | None:
    """
    Load the CzechGlobe deforestation files.
    Args:
        files: the paths of the deforestation files
        data_dir: the directory containing the deforestation files
    Returns:
        gpd.GeoDataFrame: the deforestation files
    """
    dfs = []

    for period in files:
        if "deforestation" in period:
            file_path = data_dir / period

            if Path(file_path).exists():
                gdf = gpd.read_file(file_path)
                dfs.append(gdf)
                print(f"Loaded {file_path} with {len(gdf)} features\n")
            else:
                print(f"File not found: {file_path}")

    if dfs:
        deforestation_all = gpd.GeoDataFrame(
            pd.concat(dfs, ignore_index=True),
            geometry="geometry",
            crs=dfs[0].crs,
        )
        print(f"Total: {len(deforestation_all)} features\n\n")
        print(deforestation_all.columns, "\n")
    else:
        print("No deforestation files loaded\n")
        return None

    return deforestation_all


def get_aoi_stats(aoi_ee: ee.Geometry) -> dict:
    """
    Get the statistics of the AOI.
    Args:
        aoi_ee: the area of interest
    Returns:
        dict: the statistics of the AOI
    """
    pixel_area_image = ee.Image(1).reproject(crs="EPSG:4326", scale=20).multiply(ee.Image.pixelArea())

    stats = (
        pixel_area_image.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=aoi_ee,
            scale=20,
            maxPixels=1e9,
        )
        .getInfo()
    )

    print(f'Mean: {stats["constant_mean"]:.2f} m^2')
    print(f'Min:  {stats["constant_min"]:.2f} m^2')
    print(f'Max:  {stats["constant_max"]:.2f} m^2')
    print(f'StdDev: {stats["constant_stdDev"]:.2f} m^2')
    print(f'Variation: {(stats["constant_max"]-stats["constant_min"]):.2f} m^2')
    return stats


def count_disturbance_pixels(image, name, aoi_ee, scale: int = 20):
    """
    Count the disturbance pixels in the AOI.
    Args:
        image: the image to count the disturbance pixels in
        name: the name of the image
        aoi_ee: the area of interest
        scale: the scale at which to perform the calculation
    Returns:
        int: the number of disturbance pixels
    """
    stats = image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=scale,
        maxPixels=1e9,
    )

    count = stats.get("disturbance").getInfo()
    print(f"{name}: {count:,} disturbance pixels")
    return count


def sample_yearly_data(
    feature_collection: ee.FeatureCollection,
    years: list,
    aoi: ee.Geometry,
    was_forest_confirmed: ee.Image,
    ever_disturbed: ee.Image,
    num_points: int = 400,
    months: list = MONTHS,
    bands: list = S2_BANDS,
    seed_offset: int = 0,
) -> ee.FeatureCollection:
    """
    Build yearly-matched Sentinel-2 features and sample
    disturbance/stable pixels for each year.

    Args:
        feature_collection: disturbance polygons with 'year' property
        years: years to process
        aoi: area of interest
        was_forest_confirmed: confirmed forest mask
        ever_disturbed: ever disturbed mask
        num_points: samples per class per year
        months: months for S2 composite
        bands: S2 bands to use
        seed_offset: offset added to year for random seed

    Returns:
        merged FeatureCollection of all samples
    """
    all_samples = []

    for yr in years:
        print(f"  Processing year {yr}...")

        # Build yearly matched features
        s2_prev = build_s2_composite(year=yr - 1, months=months, aoi=aoi, bands=bands)
        s2_curr = build_s2_composite(year=yr, months=months, aoi=aoi, bands=bands)

        if s2_prev is None or s2_curr is None:
            print(f"  Skipping {yr} — insufficient imagery")
            continue

        ndvi_slope = compute_slope(s2_prev, s2_curr, "NDVI")
        nbr_slope = compute_slope(s2_prev, s2_curr, "NBR")
        features_yr = build_features(s2_prev, s2_curr, ndvi_slope, nbr_slope)

        # Labels for this year only
        labels_yr = rasterize_disturbance_polygons(
            feature_collection.filter(ee.Filter.eq("year", yr)),
            aoi,
        )

        # Stable mask
        stable_mask = get_stable_mask(
            year_labels=labels_yr,
            was_forest_confirmed=was_forest_confirmed,
            ever_disturbed=ever_disturbed,
        )

        if yr < 2017:
            # take 80% of samples since there are few images for early years
            num_points_yr = int(num_points * 0.8)
        else:
            num_points_yr = num_points

        # Sample disturbance pixels
        dist = (
            features_yr.updateMask(labels_yr)
            .addBands(ee.Image.constant(1).rename("label"))  # disturbance = 1
            .addBands(ee.Image.constant(yr).rename("year"))  # year as band for sampling
            .stratifiedSample(
                numPoints=num_points_yr,
                classBand="label",
                region=aoi,
                scale=20,
                seed=yr + seed_offset,
                dropNulls=True,
                geometries=True,
            )
        )

        # Sample stable pixels
        stable = (
            features_yr.updateMask(stable_mask)
            .addBands(ee.Image.constant(0).rename("label"))  # stable = 0
            .addBands(ee.Image.constant(yr).rename("year"))  # year as band for sampling
            .stratifiedSample(
                numPoints=num_points_yr,
                classBand="label",
                region=aoi,
                scale=20,
                seed=yr + seed_offset,
                dropNulls=True,
                geometries=True,  # pixel center geometry included in output
            )
        )

        all_samples.extend([dist, stable])

    if not all_samples:
        raise ValueError("No samples collected — check imagery availability")

    return ee.FeatureCollection(all_samples).flatten()


def parse_geometry(geom_str: str):
    """
    Parse the geometry from the string.
    Args:
        geom_str: the string containing the geometry
    Returns:
        dict: the geometry
    """
    if geom_str is None or pd.isna(geom_str):
        return None
    try:
        geom_dict = json.loads(geom_str)  # primary parser for .geo JSON
    except Exception as e:
        print(f"Error parsing geometry: {e}")
        return None

    if isinstance(geom_dict, dict):
        return geom_dict

    print("Parsed geometry is not a dict")
    return None

