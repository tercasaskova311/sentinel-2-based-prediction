import ee
from . import config
def get_confirmed_forest_mask(hansen_treecover:ee.Image, worldcover_map:ee.Image,
                              scale:int, tree_class:int, hansen_tree_cover_threshold:int)->tuple:
    """
    Create a confirmed forest mask by combining Hansen tree cover and WorldCover land cover classes.
    - Hansen tree cover > 70% resampled to 20m
    - WorldCover tree cover class (class 10) with focal mode ( pixels must be surrounded by other tree cover pixels)
      to get core forest areas, then resampled to 20m
    - Final mask is the intersection of both masks to get high-confidence forest areas.

    Args:
        hansen_treecover: ee.Image with the 'treecover2000' band from the Hansen dataset
        worldcover_map: ee.Image with the 'Map' band from the ESA WorldCover dataset in 2020
        scale: the scale at which to perform the resampling and calculations (e.g., 20 for 20m)
        tree_class: the WorldCover class value that corresponds to tree cover (e.g., 10)
        hansen_tree_cover_threshold: the canopy cover percentage threshold for Hansen tree cover (e.g., 70)

    Returns:
        (ee.Image, ee.Image, ee.Image): A tuple containing the confirmed forest mask, Hansen forest mask, and WorldCover core forest mask.

    """
    
    # hansen treecover > 70% resampled to 20m
    # # Upsampling Hansen tree cover from 30m to 20m just reprojecting it and copy the values 
    was_forest_20m = (hansen_treecover
                     .reproject(crs='EPSG:4326', scale=scale)
                     .gt(hansen_tree_cover_threshold))
    

    # worldcover tree cover class
    # get tree cover class and apply focal mode to get core forest areas, then resample to 20m
    wc_forest = worldcover_map.eq(tree_class)
    wc_forest_core = wc_forest.focal_min(radius=10, kernelType='circle', units='meters', iterations=1)
    wc_forest_core_20m = (
        wc_forest_core
        .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=9)
        .reproject(crs='EPSG:4326', scale=scale)
    )

    # Combine Hansen and WorldCover masks
    return was_forest_20m.And(wc_forest_core_20m).rename('confirmed_forest_mask'), was_forest_20m.rename('hansen_forest2000_mask'), wc_forest_core_20m.rename('worldcover_core_forest_mask')

def get_stable_mask(year_labels, was_forest_confirmed, ever_disturbed):
    """
    Stable pixels = confirmed forest
                    AND not disturbed this year
                    AND never disturbed in any recorded year
    """
    return (was_forest_confirmed
        .And(year_labels.Not())      # not disturbed this year
        .And(ever_disturbed.Not()))  # never disturbed anywhere


def calculate_loss_area_by_year(loss_year_img: ee.Image, mask: ee.Image, aoi: ee.Geometry, scale: int)->dict:
    """
    Calculate the area of forest loss by year within the AOI using the Hansen loss year image and a mask of confirmed forest.
    Args:
        loss_year_img: ee.Image with the 'area_loss_year' band from the Hansen dataset, resampled to 20m
        mask: binary mask of confirmed forest areas
        aoi: ee.Geometry representing the area of interest
        scale: the scale at which to perform the calculation
    Returns:
        dict: A dictionary containing the loss area by year.
    """
    loss_area_by_year = (ee.Image.pixelArea()
                         .addBands(loss_year_img)
                         .updateMask(mask.selfMask())
                         .reduceRegion(
                            reducer=ee.Reducer.sum().group(groupField=1, groupName='area_loss_year'),
                            geometry=aoi,
                            scale=scale,
                            maxPixels=1e10
                        )).getInfo()

    return loss_area_by_year