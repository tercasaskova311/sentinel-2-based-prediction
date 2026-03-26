import ee
from src import config
def _mask_s2_clouds(image):
    """
    Mask clouds and cirrus in Sentinel-2 surface reflectance images using the QA60 band.
    The QA60 band contains bit flags for clouds and cirrus:
    - Bit 10: Cloud mask (1 = cloud, 0 = clear)
    - Bit 11: Cirrus mask (1 = cirrus, 0 = clear)
    This function creates a mask that keeps only pixels where both cloud and cirrus bits are 0 (i.e., clear pixels).
    Args:
        image (ee.Image): A Sentinel-2 surface reflectance image with a QA60 band.
    Returns:
        ee.Image: The input image with clouds and cirrus masked out.
    """
    qa = image.select('QA60')
    cloud_bit_mask  = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
              .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return image.updateMask(mask)


def build_s2_composite(year, months, aoi, bands,
                       s2_collection_name='COPERNICUS/S2_SR_HARMONIZED',
                       cloud_threshold=10, with_count=False):
    start_date = f'{year}-{months[0]:02d}-01'
    end_date   = f'{year}-{months[-1]:02d}-31'

    if year <= 2016:
        cloud_threshold = 20  # allow more cloud for sparse years
    
    s2_collection = (ee.ImageCollection(s2_collection_name)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
        .select(['B4','B8','B11','B12','QA60'])
        .map(_mask_s2_clouds)
    )

    count = s2_collection.size().getInfo()
    print(f'Number of Sentinel-2 images for {year} in AOI: {count}')

    if count < 2:
        print(f'Warning: Only {count} images found for {year}. Consider increasing cloud threshold or expanding date range.')
        return None

    composite = (s2_collection
                 .median()
                 .select(bands)
                 .reproject(crs='EPSG:4326', scale=20)
                 )
    if with_count:
        composite = composite.set('image_count', count)

    # compute indices
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    nbr = composite.normalizedDifference(['B8', 'B12']).rename('NBR')
    composite = composite.addBands(ndvi).addBands(nbr)
    
    return composite


def compute_slope(image_early, image_late, band, years_apart=1):
    return (image_late.select(band)
            .subtract(image_early.select(band))
            .divide(years_apart)
            .rename(f'{band}_slope'))


def rasterize_disturbance_polygons(fc, aoi_ee, scale=20):
    """
    Rasterizes the disturbance polygons into a binary image where pixels within any disturbance polygon are set to 1, and others are 0.
    The resulting image is clipped to the AOI and reprojected to EPSG:4326 with the specified scale.
    1: disturbance pixel, 0: non-disturbance pixel (background)
    """
    # Create a raster where each pixel value is the area of the disturbance polygon it falls into
    return (ee.Image(0)
                        .byte().paint(fc, color=1)
                        .clip(aoi_ee)
                        .reproject(crs='EPSG:4326', scale=scale)
                        .rename('disturbance'))



def build_features(s2_y1, s2_y2, ndvi_slope, nbr_slope):
    """Build features for a given year based on the early year and the year of interest.
    Args:
        s2_y1: ee.Image composite for the early year
        s2_y2: ee.Image composite for the year of interest
    Returns:
        ee.Image: An image containing the features for the year of interest, including the original bands for the year of interest and the slopes of NDVI and NBR between the early year and the year of interest
    """
    all_bands = config.S2_BANDS + ['NDVI', 'NBR']

    return (s2_y1.select(all_bands)
            .rename([f'{b}_y1' for b in all_bands])
            .addBands(s2_y2.select(all_bands)
            .rename([f'{b}_y2' for b in all_bands]))
            .addBands(ndvi_slope)
            .addBands(nbr_slope))