import ee
import geemap
import geopandas as gpd

AOI_PATH = "data/boundaries/sumava_aoi_clean.geojson"
SCALE = 30

ee.Initialize(project='youtubeapi-455317')

gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
lossyear = gfc.select("lossyear") 
lossyear = lossyear.updateMask(lossyear.gt(0))

aoi_gdf = gpd.read_file(AOI_PATH)
aoi_ee = ee.Geometry.Polygon(
    aoi_gdf.geometry.iloc[0].__geo_interface__["coordinates"]
)

task = ee.batch.Export.image.toDrive(
    image=lossyear,
    description="GFC_lossyear_export",
    folder="GEE_exports",            
    fileNamePrefix="gfc_lossyear",
    scale=SCALE,
    region=aoi_ee,
    maxPixels=1e13
)

task.start()
print("Export started. Go to https://code.earthengine.google.com/tasks to monitor.")
