# sentinel-2-based-prediction
Sentinel-2 Based Deforestation/Bark Bettle regrowth + Suspicious Logging Detection in Šumava


Sentinel-2 Based Deforestation/Bark Bettle regrowth + Suspicious Logging Detection in Šumava


STRUCTURE
detect forest disturbance and classify some of it as “potentially suspicious” using:
✔ Sentinel-2 10 m yearly composites
✔ Hansen Global Forest Change (for training)
✔ Šumava NP zone maps (Zone 1 = no logging allowed)
✔ Forest road maps (OpenStreetMap)
✔ Patch-based ML model (ChangeFormer or UNet)

and produce:
● A forest-disturbance probability map
● A “suspicious disturbance” map with categories:
disturbance in Zone 1
disturbance > 30m from forest road
patch too small (<0.25 ha)
disturbance hugging zone boundaries (<20m)
disturbance not in Hansen (extra detections)


PART 1: AREA OF INTEREST 
1.1 Collect administrative boundaries
You need:
NP Šumava boundary
CHKO Šumava boundary
Zone maps (Zone 1, 2, 3)
→ from Národní park Šumava website or AOPK ČR – Mapové služby
→ They offer GeoJSON/WMS shape downloads.
1.2 Merge them into a single AOI
Entrance point: QGIS
Load all boundaries.
Merge NP + CHKO into one polygon → call it “Šumava_AOI.geojson”.
Keep zone boundaries separate → “zones.geojson”.
1.3 Keep everything in EPSG:4326
This is required for Google Earth Engine (GEE).

PART 2 (DOWNLOADING SENTINEL-2)
Even without Sentinel-1, Sentinel-2 is fully enough for the project.
You will produce one yearly composite per year, e.g.:
2016
2018
2020
2022
2024
Why skip years?
→ To focus on larger change intervals with less noise.
2.1 Why yearly composites?
Remove clouds
Remove shadows
Stabilize seasonality
Make images comparable
Essential for ML
2.2 Image filters to apply in GEE:
AOI filter
CLOUDY_PIXEL_PERCENTAGE <= 20
Use only June–September each year (growing season)
Apply a cloud mask (QA60 band or S2 Cloud Probability)
2.3 Composite type
The best for deforestation is median or percentile (25th) composite.
2.4 Export settings
Export each composite to Google Drive with:
10 m resolution
Float32
Region = your AOI
Scale = 10
CRS = EPSG:4326
2.5 Recommended years
Pick years depending on the bark-beetle timeline:
Minimum set:
2016 – 2019 – 2022 – 2024
More complete:
2016–2024 yearly

PART 3 (DOWNLOAD LABELS (GROUND TRUTH))
You need labels so the ML model can learn “what deforestation looks like.”
3.1 Hansen Forest Loss (Global Forest Change)
Select “lossyear” band
Clip to AOI
Reproject to 10 m
Convert to binary mask per year:
lossyear == target year → 1
else → 0
3.2 Combine Hansen maps into two labels:
Binary forest loss (0 = no loss, 1 = loss)
Loss year (optional for time-aware ML)
3.3 Optional: land cover map (Dynamic World)
Use Dynamic World V1 for:
forest class
built-up
grassland
“masks” to confirm confusion areas
Not required but helpful.

PART 4 (ALIGN & PREPARE YOUR DATA LOCALLY)
Everything must have the same:
resolution (10 m)
CRS (EPSG:4326 or local UTM)
bounds
pixel alignment
4.1 Use QGIS or Rasterio for:
resampling
clipping
stacking bands
4.2 Create your dataset folders:
data/
    sent2_composites/
        2016.tif
        2018.tif
        2020.tif
        2022.tif
        2024.tif

    labels/
        hansen_binary.tif
        hansen_year.tif

    zones/
        zones.geojson

    roads/
        forest_roads.geojson

4.3 Patch extraction
You need training patches, example:
patch size = 160×160 pixels (1.6×1.6 km)
~5,000–10,000 patches
Label each patch using the Hansen mask.

PART 5 (TRAINING)
This part is easiest in Google Colab.
5.1 Model options
✔ ChangeFormer-S (best)
✔ UNet (works fine)
✔ Siamese UNet (for pairwise change detection)
5.2 Input to the model:
For each patch:
Image_t0 (e.g., 2018 composite, RGB + extra bands)
Image_t1 (e.g., 2022 composite)
Label (Hansen loss in 2018–2022 interval)

Recommended spectral bands:
B2 (Blue)
B3 (Green)
B4 (Red)
B8 (NIR)
NDVI (computed locally)
NDWI (optional)
5.3 Output:
A probability map of change (forest loss)
5.4 Training tips:
Train 20–40 epochs
Use data augmentation
Use smaller model (ChangeFormer-S, not B)
Sentinel-2 deforestation models converge fast
Track accuracy vs. Hansen labels for validation

PART 6 (INFERENCE (RUNNING THE MODEL OVER THE WHOLE REGION))
For each year pair, run inference:
2016→2018
2018→2020
2020→2022
2022→2024
Generate:
a probability raster
threshold it to get a binary disturbance map
Then merge these maps into a single forest disturbance catalogue.

PART 7 (DETECTING “SUSPICIOUS LOGGING”)
This is the scientifically cool part.
You apply rules after you have the disturbance map.
7.1 Step 1 — Convert disturbance raster → polygons
Using QGIS or GeoPandas:
raster to polygons
dissolve small holes
compute patch area
Keep only patches ≥ 0.03 ha (3 pixels) to remove noise.

7.2 Step 2 — Compute attributes for each patch
For each patch, compute:
A. Patch area (ha)
small (<0.25 ha) → selective / suspicious
medium (0.25–2 ha) → local management
large (>2 ha) → clearcuts or beetle outbreak
B. Distance to forest road
Using your roads.geojson:
if distance > 30–50 m → suspicious

Because legal logging almost always uses roads.
C. Intersection with zone boundaries
Using zones.geojson:
if patch intersects Zone 1 → very suspicious
if patch within 20–40m of Zone 1 boundary → suspicious
D. Overlap with Hansen loss
if predicted change not in Hansen → possibly hidden/under-reported logging
(very interesting scientifically)
E. Patch shape metrics
You can compute:
compactness
elongation
perimeter/area (shape index)
Selective illegal cuts often have:
irregular shapes
narrow “strips”
access-point patterns

7.3 Step 3 — Classification into categories
Create a column “suspicion_reason”:
Category 1 — Inside Zone 1
Logging is prohibited → high-interest patch.
Category 2 — No road access
Patch far from roads → suspicious.
Category 3 — Small selective patches (<0.25 ha)
Often not in logging records → interesting.
Category 4 — Boundary manipulation
Patch hugging NP zones or CHKO boundaries.
Category 5 — Model-detected but not in Hansen
Possible unreported disturbance.
You can assign:
SUSPICIOUS_LEVEL = sum(all triggered conditions)


PART 8 (FINAL VISUALIZATIONS)
Using QGIS:
You produce:
✔ Map 1
Forest disturbance Šumava 2016–2024
✔ Map 2
Suspicious disturbance categories
✔ Map 3
Heatmap of suspicious logging events
✔ Map 4
Disturbance inside Zone 1 only
✔ Graphs
yearly forest loss
patch size distribution
distance to roads histogram
✔ Final PDF report
Perfect for school.

PART 9 (OPTIONAL HISTORICAL ANALYSIS)
If you want to extend the project:
Historical comparisons:
2010 orthophoto vs 2024
Landsat 5/7/8 time series (1990–2024)
Historical clearcut clusters vs bark beetle outbreak spread
Not needed for the machine learning part, but a cool appendix.







