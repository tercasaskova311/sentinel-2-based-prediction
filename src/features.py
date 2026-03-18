import ee
from src import config

def build_s2_composite(year, months, aoi, bands,
                       s2_collection_name='COPERNICUS/S2_SR_HARMONIZED',
                       cloud_threshold=20):
    start_date = f'{year}-{months[0]:02d}-01'
    end_date   = f'{year}-{months[-1]:02d}-31'
    
    s2_collection = (ee.ImageCollection(s2_collection_name)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    )

    count = s2_collection.size().getInfo()
    print(f'Number of Sentinel-2 images for {year} in AOI: {count}')

    if count < 5:
        print(f'Warning: Only {count} images found for {year}. Consider increasing cloud threshold or expanding date range.')

    composite = (s2_collection
                 .median()
                 .select(bands)
                 .reproject(crs='EPSG:4326', scale=20)
                 )

    # compute indices
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    nbr = composite.normalizedDifference(['B8', 'B12']).rename('NBR')
    composite = composite.addBands(ndvi).addBands(nbr)
    
    return composite


def _compute_slope(image_early, image_late, years_apart, band):
    return (image_late.select(band)
            .subtract(image_early.select(band))
            .divide(years_apart)
            .rename(f'{band}_slope'))

def _build_yearly_features(s2_early, s2_y1, s2_y2, years_apart):
    """Build features for a given year based on the early year and the year of interest.
    Args:
        s2_early: ee.Image composite for the early year (e.g., 2018 for training, 2022 for validation)
        s2_y1: ee.Image composite for the year of interest (e.g., 2020 for training, 2023 for validation)
        s2_y2: ee.Image composite for the year after the year of interest (e.g., 2021 for training, 2024 for validation)
        years_apart: number of years between the early year and the year of interest (e.g., 2 for training, 1 for validation)
    Returns:
        ee.Image: An image containing the features for the year of interest, including the original bands for the year of interest and the slopes of NDVI and NBR between the early year and the
    """
    all_bands = config.S2_BANDS + ['NDVI', 'NBR']

    ndvi_slope  = _compute_slope(s2_early, s2_y2, years_apart, 'NDVI')
    nbr_slope   = _compute_slope(s2_early, s2_y2, years_apart, 'NBR')

    return (s2_y1.select(all_bands)
            .rename([f'{b}_y1' for b in all_bands])
            .addBands(s2_y2.select(all_bands)
            .rename([f'{b}_y2' for b in all_bands]))
            .addBands(ndvi_slope)
            .addBands(nbr_slope))


def get_all_features(aoi, bands, months=None):
    if months is None:
        months = config.MONTHS
    # Pre-training years: 2018 and 2019
    s2_2018 = build_s2_composite(year=2018, months=months, aoi=aoi, bands=bands)
    s2_2019 = build_s2_composite(year=2019, months=months, aoi=aoi, bands=bands)

    # Training years
    s2_2020 = build_s2_composite(year=2020, months=months, aoi=aoi, bands=bands)
    s2_2021 = build_s2_composite(year=2021, months=months, aoi=aoi, bands=bands)

    # Validation year
    s2_2022 = build_s2_composite(year=2022, months=months, aoi=aoi, bands=bands)
    s2_2023 = build_s2_composite(year=2023, months=months, aoi=aoi, bands=bands)

    # Test year
    s2_2024 = build_s2_composite(year=2024, months=months, aoi=aoi, bands=bands)
    s2_2025 = build_s2_composite(year=2025, months=months, aoi=aoi, bands=bands)


    train_features      = _build_yearly_features(s2_2018, s2_2020, s2_2021, 3)
    val_features        = _build_yearly_features(s2_2022, s2_2022, s2_2023, 1)
    test_features_fair  = _build_yearly_features(s2_2023, s2_2023, s2_2024, 1)
    test_features       = _build_yearly_features(s2_2024, s2_2024, s2_2025, 1)

    data_sets = {
        "s2_2020": s2_2020,
        "s2_2022": s2_2022,
        "s2_2024": s2_2024,
    }

    return train_features, val_features, test_features_fair, test_features, data_sets