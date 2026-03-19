// =============================================================================
// ŠUMAVA NP — FOREST DISTURBANCE DETECTION
// Same-season rolling z-score · Sentinel-2 only · Zero external labels
//
// No Hansen. No training data. No ground truth dependency.
//
// Forest mask: ESA WorldCover 2021 (10m, tree cover class = 10)
//   — purely a spatial mask to restrict alerts to forested pixels
//   — no loss labels, no reference dataset
//
// Core logic:
//   For each new S2 image, compare it to the same calendar months in the
//   archive (2020–2024). Z-score < threshold = anomaly. Pixels flagged in
//   >= N scenes AND inside forest AND part of a patch >= 0.5 ha = alert.
//
// Baselines are precomputed ONCE per season month (not inside map()) to
// avoid GEE memory limit errors from deep lazy evaluation graphs.
// =============================================================================


// =============================================================================
// 0. CONFIGURATION
// =============================================================================

var CONFIG = {

  // Archive used to build seasonal baselines
  archiveStart: '2020-01-01',
  archiveEnd:   '2025-01-01',   // exclusive

  // New imagery to score
  detectStart: '2025-01-01',
  detectEnd:   '2025-12-31',

  // Growing season only — avoids winter phenology noise
  seasonStart: 5,   // May
  seasonEnd:   9,   // September

  // For detection image in month M, baseline = archive months [M-buf, M+buf]
  seasonBuffer: 1,

  // Z-score thresholds (negative = drop below baseline = disturbance)
  zThreshNBR:  -2.5,
  zThreshNDVI: -2.0,

  // Min number of scenes that must flag a pixel (persistence check)
  minScenes: 2,

  // Min contiguous patch size in pixels (10m: 50 px = 0.5 ha)
  minPatchPx: 50,

  // S2 scene-level cloud pre-filter
  maxCloudPct: 50,

  // Export
  exportScale:  20,            // 20m — half filesize vs 10m, still sufficient
  exportCRS:    'EPSG:32633',  // UTM 33N
  exportFolder: 'GEE_Sumava',
};


// =============================================================================
// 1. AOI
// =============================================================================



print('AOI area (km²):', aoi.area(1).divide(1e6).round());


// =============================================================================
// 2. FOREST MASK — ESA WorldCover 2021
// =============================================================================
//
// WorldCover classes (10m):
//   10 = Tree cover       ← what we want
//   20 = Shrubland
//   30 = Grassland
//   40 = Cropland
//   50 = Built-up
//   60 = Bare / sparse vegetation
//   70 = Snow and ice
//   80 = Permanent water
//   90 = Herbaceous wetland
//   95 = Mangroves
//   100 = Moss and lichen
//
// Using WorldCover instead of Hansen means:
//   - No deforestation labels anywhere in the pipeline
//   - 10m resolution matches S2 native resolution
//   - Reflects current (2021) land cover, not year-2000 baseline

var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()                          // single global mosaic
  .select('Map')
  .clip(aoi);

// Tree cover pixels only
var treeMask = worldcover.eq(10);

// Edge removal: focalMin over 3×3 — pixel stays 1 only if all neighbours
// are also forest. Avoids band-renaming side effects of reduceNeighborhood.
var coreForest = treeMask
  .focalMin({radius: 1, kernelType: 'square', units: 'pixels'})
  .eq(1)
  .rename('forest')
  .selfMask()
  .clip(aoi);

// Verify forest mask — pixelArea() is scale-independent (no resampling artefacts)
// Result is in m²; divide by 1e6 to get km²
print('WorldCover core forest area (km2):',
  ee.Number(coreForest.multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer:   ee.Reducer.sum(),
      geometry:  aoi,
      scale:     10,
      maxPixels: 1e10
    })
    .get('forest')
  ).divide(1e6).round()
);


// =============================================================================
// 3. SENTINEL-2 PREPROCESSING
// =============================================================================

function scaleS2(img) {
  return img.select('B.*').multiply(0.0001)
    .addBands(img.select('SCL'))
    .copyProperties(img, img.propertyNames());
}

function maskClouds(img) {
  var scl = img.select('SCL');
  var bad = scl.eq(3)    // cloud shadow
    .or(scl.eq(8))       // cloud medium probability
    .or(scl.eq(9))       // cloud high probability
    .or(scl.eq(10))      // thin cirrus
    .or(scl.eq(11));     // snow / ice
  return img.updateMask(bad.not());
}

function addIndices(img) {
  var nir   = img.select('B8');
  var red   = img.select('B4');
  var swir2 = img.select('B12');
  var nbr   = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR');
  var ndvi  = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  return nbr.addBands(ndvi)
    .copyProperties(img, ['system:time_start', 'system:index']);
}

function loadS2(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.calendarRange(CONFIG.seasonStart, CONFIG.seasonEnd, 'month'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
    .map(scaleS2)
    .map(maskClouds)
    .map(addIndices);
}


// =============================================================================
// 4. LOAD COLLECTIONS
// =============================================================================

var s2Archive   = loadS2(CONFIG.archiveStart, CONFIG.archiveEnd);
var s2Detection = loadS2(CONFIG.detectStart,  CONFIG.detectEnd);

print('S2 archive images:',   s2Archive.size());
print('S2 detection images:', s2Detection.size());


// =============================================================================
// 5. PRECOMPUTE MONTHLY BASELINES
// =============================================================================
//
// Built ONCE per season month outside of map() — avoids rebuilding the entire
// archive computation graph for each of the 152 detection images.
//
// For month M: baseline = all archive images from months [M-buf, M+buf].
// Stored as an ee.ImageCollection with a 'month' property for fast lookup.

var baselineImages = [];

for (var m = CONFIG.seasonStart; m <= CONFIG.seasonEnd; m++) {
  var lo    = m - CONFIG.seasonBuffer;
  var hi    = m + CONFIG.seasonBuffer;
  var slice = s2Archive.filter(ee.Filter.calendarRange(lo, hi, 'month'));

  var baseline = slice.select('NBR').mean().rename('NBR_mean')
    .addBands(slice.select('NBR').reduce(ee.Reducer.stdDev()).max(0.01).rename('NBR_std'))
    .addBands(slice.select('NDVI').mean().rename('NDVI_mean'))
    .addBands(slice.select('NDVI').reduce(ee.Reducer.stdDev()).max(0.01).rename('NDVI_std'))
    .set('month', m);

  baselineImages.push(baseline);
}

var baselineCol = ee.ImageCollection(baselineImages);

print('Baselines precomputed for months ' + CONFIG.seasonStart + '–' + CONFIG.seasonEnd);


// =============================================================================
// 6. SCORE DETECTION IMAGES
// =============================================================================

function scoreImage(img) {
  var month    = img.date().get('month');
  var baseline = ee.Image(baselineCol.filter(ee.Filter.eq('month', month)).first());

  var nbrZ  = img.select('NBR')
    .subtract(baseline.select('NBR_mean'))
    .divide(baseline.select('NBR_std'))
    .rename('NBR_z');

  var ndviZ = img.select('NDVI')
    .subtract(baseline.select('NDVI_mean'))
    .divide(baseline.select('NDVI_std'))
    .rename('NDVI_z');

  // AND: both indices must drop — cuts cloud-remnant false positives
  var alert = nbrZ.lt(CONFIG.zThreshNBR)
    .and(ndviZ.lt(CONFIG.zThreshNDVI))
    .rename('alert')
    .toFloat();

  return img.addBands([nbrZ, ndviZ, alert])
    .set('system:time_start', img.get('system:time_start'));
}

var s2Scored = s2Detection.map(scoreImage);

print('Scoring complete');


// =============================================================================
// 7. PERSISTENCE FILTER
// =============================================================================

var alertCount = s2Scored.select('alert').sum().rename('alert_count');

var persistentAlert = alertCount
  .gte(CONFIG.minScenes)
  .rename('persistent_alert');


// =============================================================================
// 8. FOREST MASK + PATCH FILTER
// =============================================================================

var maskedAlert = persistentAlert.updateMask(coreForest);

var patchSize  = maskedAlert.connectedPixelCount(CONFIG.minPatchPx + 1);
var cleanAlert = maskedAlert
  .updateMask(patchSize.gte(CONFIG.minPatchPx))
  .rename('clean_alert');


// =============================================================================
// 9. DIAGNOSTICS
// =============================================================================

// Most extreme NBR z-score per pixel — use to tune zThreshNBR
var nbrZMin = s2Scored.select('NBR_z').min().rename('NBR_z_min');

// First day offset (from detectStart) that each pixel was flagged
var detectStartMs = ee.Date(CONFIG.detectStart).millis();

var firstAlertDay = s2Scored
  .map(function(img) {
    var day = ee.Image.constant(
      img.date().millis().subtract(detectStartMs).divide(86400000)
    ).toFloat().rename('first_alert_day');
    return day.updateMask(img.select('alert'));
  })
  .min()
  .updateMask(cleanAlert)
  .rename('first_alert_day');


// =============================================================================
// 10. VISUALISATION
// =============================================================================

Map.centerObject(aoi, 11);

// True colour
var s2TC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(CONFIG.detectStart, CONFIG.detectEnd)
  .filter(ee.Filter.calendarRange(CONFIG.seasonStart, CONFIG.seasonEnd, 'month'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
  .map(scaleS2).map(maskClouds)
  .select(['B4', 'B3', 'B2']).median().clip(aoi);

Map.addLayer(s2TC,
  {min: 0, max: 0.3, gamma: 1.4},
  'S2 true colour', true);

// WorldCover tree mask
Map.addLayer(coreForest.clip(aoi),
  {palette: ['1a9641'], opacity: 0.3},
  'Forest mask (WorldCover tree cover)', false);

// NBR z-score min — threshold tuning layer
Map.addLayer(nbrZMin.clip(aoi),
  {min: -5, max: 0, palette: ['d73027', 'f46d43', 'fdae61', 'ffffbf', 'ffffff']},
  'NBR z-score min (red = most anomalous)', false);

// Scene count — persistence
Map.addLayer(alertCount.updateMask(alertCount.gt(0)).clip(aoi),
  {min: 1, max: 10, palette: ['ffffcc', 'feb24c', 'f03b20']},
  'Alert scene count', false);

// Persistent alerts before patch filter
Map.addLayer(persistentAlert.selfMask().clip(aoi),
  {palette: ['FFA500'], opacity: 0.7},
  'Persistent alerts (pre patch filter)', false);

// Final alerts — primary output
Map.addLayer(cleanAlert.selfMask().clip(aoi),
  {palette: ['FF0000'], opacity: 0.9},
  'FINAL ALERTS', true);

// First alert day
Map.addLayer(firstAlertDay.clip(aoi),
  {min: 0, max: 365, palette: ['d94701', 'fd8d3c', 'ffffcc']},
  'First alert day (orange = earlier)', false);

// NP boundary
Map.addLayer(
  ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), 0, 2),
  {palette: ['00ffff']}, 'Šumava NP boundary');


// =============================================================================
// 11. CONSOLE STATS
// =============================================================================

// Forest area in km² using pixelArea — avoids resampling fractions at coarse scale
var forestKm2 = ee.Number(
  coreForest.multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e10})
    .get('forest')
).divide(1e6);

// Use pixelArea() for all area stats — avoids resampling fractions and
// is more memory-efficient than summing unmasked binary images at fine scale.
// selfMask() before reduceRegion means GEE only touches non-zero pixels.

// bestEffort: true — GEE auto-coarsens scale if memory limit is hit
// selfMask() before reduce so GEE skips masked (non-alert) pixels entirely
var persistAreaHa = ee.Number(
  persistentAlert.selfMask().clip(aoi)
    .multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer:     ee.Reducer.sum(),
      geometry:    aoi,
      scale:       100,
      maxPixels:   1e10,
      bestEffort:  true
    })
    .get('persistent_alert')
).divide(1e4).round();

var cleanAreaHa = ee.Number(
  cleanAlert.selfMask().clip(aoi)
    .multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer:     ee.Reducer.sum(),
      geometry:    aoi,
      scale:       100,
      maxPixels:   1e10,
      bestEffort:  true
    })
    .get('clean_alert')
).divide(1e4).round();

var meanScenes = alertCount.updateMask(cleanAlert).clip(aoi)
  .reduceRegion({
    reducer:    ee.Reducer.mean(),
    geometry:   aoi,
    scale:      100,
    maxPixels:  1e10,
    bestEffort: true
  })
  .get('alert_count');

print('--- Stats ---');
print('Core forest area (km2):', forestKm2);
print('Persistent alert area, pre-patch (ha):', persistAreaHa);
print('Final alert area, post-patch (ha):', cleanAreaHa);
print('Mean scene count per alert pixel:', meanScenes);


// =============================================================================
// 12. EXPORT
// =============================================================================

var yr = CONFIG.detectStart.slice(0, 4);

function exportImg(img, name) {
  Export.image.toDrive({
    image:          img.toFloat(),
    description:    name,
    folder:         CONFIG.exportFolder,
    fileNamePrefix: name,
    region:         aoi,
    scale:          CONFIG.exportScale,
    crs:            CONFIG.exportCRS,
    maxPixels:      1e10,
    fileFormat:     'GeoTIFF'
  });
}

exportImg(cleanAlert,                        'sumava_alerts_'      + yr);
exportImg(nbrZMin.clip(aoi),                 'sumava_nbr_zscore_'  + yr);
exportImg(alertCount.clip(aoi),              'sumava_scene_count_' + yr);
exportImg(firstAlertDay.unmask(0).clip(aoi), 'sumava_first_day_'   + yr);

// Alert polygons as GeoJSON for field use
Export.table.toDrive({
  collection: cleanAlert.selfMask().reduceToVectors({
    geometry: aoi, scale: CONFIG.exportScale,
    geometryType: 'polygon', eightConnected: true,
    maxPixels: 1e10, reducer: ee.Reducer.countEvery()
  }).map(function(f) {
    return f.set({area_ha: f.geometry().area(1).divide(1e4).round(), year: yr});
  }),
  description:    'sumava_alert_polygons_' + yr,
  folder:         CONFIG.exportFolder,
  fileNamePrefix: 'sumava_alert_polygons_' + yr,
  fileFormat:     'GeoJSON'
});
