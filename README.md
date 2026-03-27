# Šumava NP — Forest Disturbance Detection

Sentinel-2 harmonic regression model  and random forest models for detecting forest disturbance (logging, bark beetle) in Šumava National Park, Czech Republic. 

## 1 harmonic regression model 
- built in Google Earth Engine, loosely based on the CCDC framework (Zhu & Woodcock, 2014).

## 2 machine learning models

---
## harmonic regression model  Configuration

All tunable parameters live in the `CONFIG` object at the top of the script:

```javascript
var CONFIG = {
  archiveStart:    '2020-01-01',   // Start of harmonic model training period
  archiveEnd:      '2024-01-01',   // End of training period (exclusive)
  detectStart:     '2024-01-01',   // Start of detection window
  detectEnd:       '2024-12-31',   // End of detection window
  seasonStart:     4,              // First month of growing season (April)
  seasonEnd:       11,             // Last month of growing season (November)
  numHarmonics:    2,              // 1 = annual only, 2 = annual + semi-annual
  zThreshResidual: -2.5,           // z-score threshold for flagging a scene
  minScenes:       2,              // Minimum alert scenes for persistence filter
  minPatchPx:      50,             // Minimum patch size in pixels (50 px = 0.5 ha)
  maxCloudPct:     50,             // Maximum cloud cover % for scene inclusion
  exportScale:     10,             // Export resolution in metres
  exportCRS:       'EPSG:32633',   // UTM Zone 33N
  exportFolder:    'GEE_Sumava',   // Google Drive output folder
};
```
---

## Outputs

### GEE exports (Google Drive → `GEE_Sumava/`)

| File | Content |
|------|---------|
| `sumava_alerts_YYYY.tif` | Final disturbance alerts (all types), patch-filtered |
| `sumava_residual_z_YYYY.tif` | Minimum residual z-score across detection season |
| `sumava_scene_count_YYYY.tif` | Number of alert scenes per pixel |
| `sumava_first_day_YYYY.tif` | Day-of-year of first alert (days since Jan 1) |

All outputs are `Float32` GeoTIFFs in `EPSG:32633` at 10 m resolution.

> **Important:** exports use `.selfMask()` so masked pixels are written as `NoData`, not `0`. Ensure your GIS software interprets `NoData` correctly before area calculations.

### Console stats (GEE)

After running, the Console prints:

```
Core forest area (km²)
Persistent alerts pre-forest-mask (ha)
Clean alerts post-forest-mask (ha)
Final alerts post-patch-filter (ha)
Abrupt alerts — logging (ha)
Mean scene count per final alert pixel
```

---
## 2024 Results

| Metric | Value |
|--------|-------|
| Metric                           | 2024                  
| Archive images                   | 644                   
| Detection images (season)        | 104                   
| Candidate alerts (pre-filters)   | 2285 ha              
| Final alerts post-patch-filter   | 423 ha
| Mean scene count per final alert pixel | 2.966   

---

## Known Issues & Limitations

**Slow bark beetle decline** — The abruptness filter (`firstAlertFraction < 0.6`) is designed to separate logging from gradual dieback. Bark beetle outbreaks that begin early in the season may be misclassified as logging. 

---

## Dependencies

- Google Earth Engine account with access to `COPERNICUS/S2_SR_HARMONIZED` and `ESA/WorldCover/v200`
- Python ≥ 3.9 with `gdal`, `geopandas`, `shapely`
- QGIS (optional) for visual inspection

---

## References

Zhu, Z., & Woodcock, C. E. (2014). Continuous change detection and classification of land cover using all available Landsat data. *Remote Sensing of Environment*, 144, 152–171.

ESA WorldCover 2021 v2.0 — `ESA/WorldCover/v200` via Google Earth Engine.

Copernicus Sentinel-2 Level-2A Surface Reflectance — `COPERNICUS/S2_SR_HARMONIZED` via Google Earth Engine.