# GEE Config
EE_PROJECT_ID = 'disturbance-detection'
# 'deforestation-detection-481313'

# Classification codes
WORLDCOVER_CLASS_TREES = 10
WORLDCOVER_CLASS_BUILT_UP = 50

# Threshold
HANSEN_CANOPY_COVER_THRESHOLD = 70
TARGET_SCALE_M = 20

# Months to consider for sentinel2 data
MONTHS   = [5, 6, 7, 8, 9, 10]

# Sentinel-2 bands to use
S2_BANDS = ['B4', 'B8', 'B11', 'B12']
