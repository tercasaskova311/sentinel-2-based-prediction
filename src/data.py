import ee
def get_sumave_aoi()->ee.Geometry:
    """
    Get the geometry of the Šumava national park from the WDPA dataset.
    Args:
        None
    Returns:
        ee.Geometry: Geometry of the Šumava national park.
    """
    sumava_all = ee.FeatureCollection('WCMC/WDPA/current/polygons').filter(ee.Filter.eq('NAME', 'Šumava'))

    # filter for national park and dissolve
    sumava_np = sumava_all.filter(ee.Filter.eq('DESIG', 'Národní park (NP)'))
    return sumava_np.geometry().dissolve()


def get_base_datasets()->tuple[ee.Image, ee.Image]:
    """
    Load the ESA WorldCover 2021 and Hansen Global Forest Change 2024 datasets.
    worldcover: ESA WorldCover 2021 land cover map (Map band) -> single band with land cover classes
    hansen: Hansen Global Forest Change 2024 dataset -> multiple bands including 'lossYear' and 'treecover2000'

    Args:
        None
    Returns:
        tuple: (worldcover_image, hansen_image)
    """
    worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map')

    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')

    return worldcover, hansen