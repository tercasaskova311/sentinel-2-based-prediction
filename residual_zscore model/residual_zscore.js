// =============================================================================
// ŠUMAVA NP — FOREST DISTURBANCE DETECTION
// Harmonic model baseline · Sentinel-2 NBR · Zero external labels
//
// Key improvement over simple z-score:
//   Instead of mean/std per calendar month, fit a harmonic (sine/cosine)
//   regression to the full NBR time series. The model captures the smooth
//   seasonal curve. Residuals = actual - predicted. Z-score of residuals
//   is much tighter for healthy/stressed forest (small consistent residuals)
//   and large for acute structural removal (single huge residual).
//
//   Dry/stressed trees: gradual decline → small residuals → z near 0
//   Logging:            sudden removal → massive residual → z << -3
//
// Based on: CCDC (Continuous Change Detection and Classification) approach
// Zhu & Woodcock (2014), adapted for GEE interactive mode memory constraints.
//
// Forest mask: ESA WorldCover 2021 tree cover (10m)
// Already-degraded mask: pixels anomalous in 2024 excluded from 2025 alerts
// =============================================================================


// =============================================================================
// 0. CONFIGURATION
// =============================================================================

var CONFIG = {

  archiveStart: '2020-01-01',
  archiveEnd:   '2025-01-01',

  detectStart:  '2025-01-01',
  detectEnd:    '2025-12-31',

  // Growing season — harmonic model fit on full year but detection season-limited
  seasonStart: 5,   // May
  seasonEnd:   9,   // September

  // Harmonic model: number of cycles per year
  // 1 = single annual cycle (captures main seasonal swing)
  // 2 = adds intra-annual variation (recommended for temperate forests)
  numHarmonics: 2,

  // Residual z-score threshold
  // Harmonic residuals have much tighter std than raw NBR — threshold can be
  // looser (less negative) while still being more specific than raw z-score
  // Start at -2.5, inspect NBR residual min layer, tune toward -3.0 if needed
  zThreshResidual: -2.5,

  // Min scenes that must flag a pixel (consecutive confirmation)
  minScenes: 3,

  // Min contiguous patch: 50 px at 10m = 0.5 ha
  minPatchPx: 50,

  // S2 scene-level cloud pre-filter
  maxCloudPct: 50,

  // Export
  exportScale:  20,
  exportCRS:    'EPSG:32633',
  exportFolder: 'GEE_Sumava',
};


// =============================================================================
// 1. AOI
// =============================================================================



print('AOI area (km²):', aoi.area(1).divide(1e6).round());


// =============================================================================
// 2. SENTINEL-2 PREPROCESSING
// =============================================================================

function scaleS2(img) {
  return img.select('B.*').multiply(0.0001)
    .addBands(img.select('SCL'))
    .copyProperties(img, img.propertyNames());
}

function maskClouds(img) {
  var scl = img.select('SCL');
  var bad = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10)).or(scl.eq(11));
  return img.updateMask(bad.not());
}

function addNBR(img) {
  var nir   = img.select('B8');
  var swir2 = img.select('B12');
  return nir.subtract(swir2).divide(nir.add(swir2))
    .rename('NBR')
    .copyProperties(img, ['system:time_start', 'system:index']);
}

// Archive uses FULL year (all months) so the harmonic model fits the complete
// seasonal cycle, not just the growing season
function loadS2Full(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
    .map(scaleS2)
    .map(maskClouds)
    .map(addNBR);
}

// Detection uses season filter — we only score growing season images
function loadS2Season(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.calendarRange(CONFIG.seasonStart, CONFIG.seasonEnd, 'month'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
    .map(scaleS2)
    .map(maskClouds)
    .map(addNBR);
}


// =============================================================================
// 3. FOREST MASK — ESA WorldCover 2021 + NDVI confirmation
// =============================================================================
//
// Two-layer mask:
//   Layer 1: WorldCover 2021 tree cover class (value=10) — coarse spatial mask
//   Layer 2: Peak-summer NDVI > 0.5 from 2021 archive — removes rocks,
//            wet meadows, sparse vegetation incorrectly labelled as tree cover
//
// Without the NDVI layer, bare rocky outcrops and wet areas at forest edges
// produce false alerts because their NBR is structurally low and variable.

// Peak summer NDVI from cloud-free 2021 images (pre-bark-beetle-peak baseline)
var ndviMask = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2021-06-01', '2021-09-01')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(scaleS2)
  .map(maskClouds)
  .map(function(img) {
    return img.select('B8').subtract(img.select('B4'))
      .divide(img.select('B8').add(img.select('B4')))
      .rename('NDVI');
  })
  .median()
  .gt(0.5);    // only pixels with strong summer greenness pass

var coreForest = ee.ImageCollection('ESA/WorldCover/v200')
  .first().select('Map').clip(aoi)
  .eq(10)                                              // tree cover class only
  .and(ndviMask)                                       // must also be vegetated
  .focalMin({radius: 1, kernelType: 'square', units: 'pixels'})  // edge removal
  .eq(1).rename('forest').selfMask().clip(aoi);

print('Core forest area (km²):',
  ee.Number(coreForest.multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 10, maxPixels: 1e10})
    .get('forest')
  ).divide(1e6).round()
);


// =============================================================================
// 4. LOAD COLLECTIONS
// =============================================================================

// Archive: full year (all months) for harmonic fit
var s2Archive   = loadS2Full(CONFIG.archiveStart, CONFIG.archiveEnd);
// Detection: growing season only
var s2Detection = loadS2Season(CONFIG.detectStart, CONFIG.detectEnd);

print('S2 archive images (full year):', s2Archive.size());
print('S2 detection images (season):',  s2Detection.size());


// =============================================================================
// 5. HARMONIC MODEL — fit to archive NBR time series
// =============================================================================
//
// Model: NBR(t) = c0 + c1*t + Σ [a_k*sin(k*2π*t) + b_k*cos(k*2π*t)]
//
//   c0       = intercept (overall NBR level)
//   c1*t     = linear trend (gradual long-term change)
//   sin/cos  = seasonal cycles (annual + semi-annual oscillation)
//   t        = fractional year from reference date
//
// After fitting, for any new image:
//   predicted = model(t)
//   residual  = actual_NBR - predicted
//   z-score   = residual / RMSE
//
// A healthy forest pixel in July 2025 should look like July 2020-2024 →
// residual ≈ 0 → z ≈ 0.
// A clearcut in July 2025 looks nothing like the model prediction →
// residual very negative → z << -3.
//
// Reference: Zhu & Woodcock (2014) Remote Sensing of Environment
//            CCDC — Continuous Change Detection and Classification

var referenceDate = ee.Date('2020-01-01');
var omega = 2.0 * Math.PI;

// Add harmonic predictor bands to each archive image
function addHarmonicBands(img) {
  // t = fractional years since reference date
  var t = ee.Image(
    img.date().difference(referenceDate, 'year')
  ).float().rename('t');

  var bands = [
    ee.Image.constant(1).float().rename('constant'),
    t,
    t.multiply(omega).sin().rename('sin1'),
    t.multiply(omega).cos().rename('cos1'),
  ];

  // Add second harmonic if configured (captures intra-annual variation)
  if (CONFIG.numHarmonics >= 2) {
    bands.push(t.multiply(2 * omega).sin().rename('sin2'));
    bands.push(t.multiply(2 * omega).cos().rename('cos2'));
  }

  return img.addBands(bands);
}

var harmonicArchive = s2Archive.map(addHarmonicBands);

// Predictor band names (must match what addHarmonicBands adds)
var predictors = CONFIG.numHarmonics >= 2
  ? ['constant', 't', 'sin1', 'cos1', 'sin2', 'cos2']
  : ['constant', 't', 'sin1', 'cos1'];

// Fit ordinary least squares harmonic regression
// linearRegression returns coefficients image: shape [numX, numY]
var harmonicFit = harmonicArchive
  .select(predictors.concat(['NBR']))
  .reduce(ee.Reducer.linearRegression({
    numX: predictors.length,
    numY: 1
  }));

// Extract coefficient array and flatten to named bands
// linearRegression appends '_NBR' to band names → rename to plain predictor names
var coefficientsFull = harmonicFit.select('coefficients')
  .arrayFlatten([predictors, ['NBR']]);

// Rename from 'constant_NBR', 't_NBR' etc. → 'constant', 't' etc.
var predictorsNBR = predictors.map(function(p) { return p + '_NBR'; });
var coefficients  = coefficientsFull.select(predictorsNBR, predictors);

print('Harmonic model fitted — predictors:', predictors.length);


// =============================================================================
// 6. COMPUTE RMSE OVER ARCHIVE (per-pixel residual std)
// =============================================================================
//
// RMSE is computed from archive residuals and used as the denominator for
// z-scoring detection residuals. It represents how much natural variability
// the model doesn't explain — typically much smaller than raw NBR std because
// the seasonal curve has been removed.
//
// Low RMSE = model fits well = tight z-scores = high specificity
// High RMSE = noisy pixel (e.g. persistent cloud gaps) = loose z-scores

function predictNBR(img) {
  var t = ee.Image(
    img.date().difference(referenceDate, 'year')
  ).float().rename('t');

  var predictorImg = ee.Image.constant(1).float().rename('constant')
    .addBands(t)
    .addBands(t.multiply(omega).sin().rename('sin1'))
    .addBands(t.multiply(omega).cos().rename('cos1'));

  if (CONFIG.numHarmonics >= 2) {
    predictorImg = predictorImg
      .addBands(t.multiply(2 * omega).sin().rename('sin2'))
      .addBands(t.multiply(2 * omega).cos().rename('cos2'));
  }

  // Dot product of predictor bands with coefficients = predicted NBR
  var predicted = predictorImg.select(predictors)
    .multiply(coefficients.select(predictors))
    .reduce(ee.Reducer.sum())
    .rename('NBR_predicted');

  var residual = img.select('NBR')
    .subtract(predicted)
    .rename('NBR_residual');

  return residual.copyProperties(img, ['system:time_start']);
}

// Compute archive residuals and derive per-pixel RMSE
var archiveResiduals = s2Archive.map(predictNBR);

var rmse = archiveResiduals
  .map(function(img) { return img.pow(2); })
  .mean()
  .sqrt()
  .max(0.005)        // floor RMSE at 0.005 to prevent division explosion
  .rename('RMSE');

print('RMSE computed');


// =============================================================================
// 7. ALREADY-DEGRADED MASK — harmonic trend coefficient
// =============================================================================
//
// Uses the fitted model's own trend coefficient (c1) to identify pixels
// in systematic long-term decline across 2020-2024.
//
// c1 = the linear trend term from the harmonic regression.
// A strongly negative c1 means NBR has been declining year-over-year —
// the signature of multi-year bark-beetle mortality or chronic drought stress.
//
// Advantages over checking specific years:
//   - Uses ALL 5 archive years simultaneously (no extra data loads)
//   - Zero additional memory cost — coefficients already computed in section 5
//   - Catches monotonic decline specifically, not just a single bad year
//   - A pixel that had one bad year but recovered is NOT excluded
//
// Threshold: c1 / RMSE < -0.5 means declining at >0.5 RMSE units per year.
// Tune upward (e.g. -0.3) to exclude more pre-existing stress,
// downward (e.g. -0.8) to be more conservative about exclusions.

var trendCoeff = coefficients.select('t');  // c1: linear trend per pixel

var alreadyDegraded = trendCoeff.divide(rmse)
  .lt(-0.5)
  .rename('already_degraded');

print('Already-degraded mask built (using harmonic trend coefficient c1)');


// =============================================================================
// 8. SCORE DETECTION IMAGES — harmonic residual z-score
// =============================================================================
//
// Key memory optimisation: precompute the harmonic prediction for EACH
// detection image ONCE as a single-band image, store in a collection.
// Then score against it in one pass — avoids recomputing the dot product
// 3× per image (alert + residualZ + firstDay) as the previous version did.
//
// Pipeline per image:
//   predicted  = coefficients · [1, t, sin1, cos1, sin2, cos2]
//   residual_z = (NBR - predicted) / RMSE
//   alert      = residual_z < threshold ? 1 : 0

var detectStartMs = ee.Date(CONFIG.detectStart).millis();

// Step 1: precompute residual_z for every detection image (one pass)
function computeResidualZ(img) {
  var t = ee.Image(
    img.date().difference(referenceDate, 'year')
  ).float();

  // Build predictor stack — same structure as training
  var predictorImg = ee.Image.constant(1).float().rename('constant')
    .addBands(t.rename('t'))
    .addBands(t.multiply(omega).sin().rename('sin1'))
    .addBands(t.multiply(omega).cos().rename('cos1'))
    .addBands(t.multiply(2 * omega).sin().rename('sin2'))
    .addBands(t.multiply(2 * omega).cos().rename('cos2'));

  var predicted = predictorImg.select(predictors)
    .multiply(coefficients)
    .reduce(ee.Reducer.sum());

  return img.select('NBR')
    .subtract(predicted)
    .divide(rmse)
    .rename('residual_z')
    .set('system:time_start', img.get('system:time_start'));
}

var s2ResidZ = s2Detection.map(computeResidualZ);

// Step 2: per-month aggregation — each month reduces independently
var monthlyAlertCounts = [];
var monthlyResidZMins  = [];
var monthlyFirstDays   = [];

for (var sm = CONFIG.seasonStart; sm <= CONFIG.seasonEnd; sm++) {
  var mResidZ = s2ResidZ.filter(ee.Filter.calendarRange(sm, sm, 'month'));

  // Binary alert from residual z-score
  var mAlerts = mResidZ.map(function(img) {
    return img.lt(CONFIG.zThreshResidual).rename('alert').toFloat()
      .set('system:time_start', img.get('system:time_start'));
  });

  // First day flagged
  var mFirstDay = mResidZ.map(function(img) {
    var isAlert = img.lt(CONFIG.zThreshResidual);
    return ee.Image.constant(
      img.date().millis().subtract(detectStartMs).divide(86400000)
    ).toFloat().rename('first_alert_day').updateMask(isAlert);
  });

  monthlyAlertCounts.push(mAlerts.sum().rename('alert_count'));
  monthlyResidZMins.push(mResidZ.min().rename('residual_z_min'));
  monthlyFirstDays.push(mFirstDay.min());
}

var alertCount       = ee.ImageCollection(monthlyAlertCounts).sum().rename('alert_count');
var residZMin        = ee.ImageCollection(monthlyResidZMins).min().rename('residual_z_min');
var firstAlertDayRaw = ee.ImageCollection(monthlyFirstDays).min().rename('first_alert_day');

print('Scoring complete');


// =============================================================================
// 9. PERSISTENCE + FOREST MASK + DEGRADED MASK + PATCH FILTER
// =============================================================================

var persistentAlert = alertCount.gte(CONFIG.minScenes).rename('persistent_alert');
var maskedAlert     = persistentAlert.updateMask(coreForest);
var newAlert        = maskedAlert.updateMask(alreadyDegraded.not());

var patchSize  = newAlert.connectedPixelCount(CONFIG.minPatchPx + 1);
var cleanAlert = newAlert
  .updateMask(patchSize.gte(CONFIG.minPatchPx))
  .rename('clean_alert');

var firstAlertDay = firstAlertDayRaw.updateMask(cleanAlert);


// =============================================================================
// 10. VISUALISATION
// =============================================================================

Map.centerObject(aoi, 11);

var s2TC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(CONFIG.detectStart, CONFIG.detectEnd)
  .filter(ee.Filter.calendarRange(CONFIG.seasonStart, CONFIG.seasonEnd, 'month'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
  .map(scaleS2).map(maskClouds)
  .select(['B4', 'B3', 'B2']).median().clip(aoi);

Map.addLayer(s2TC,
  {min: 0, max: 0.3, gamma: 1.4},
  'S2 true colour (May-Sep 2025)', true);

Map.addLayer(coreForest,
  {palette: ['1a9641'], opacity: 0.3},
  'Forest mask (WorldCover 2021)', false);

Map.addLayer(alreadyDegraded.selfMask().clip(aoi),
  {palette: ['8B4513'], opacity: 0.5},
  'Already degraded in 2024 (excluded)', false);

// Residual z-score — key diagnostic layer
// Harmonic residuals are tighter than raw NBR z-scores
// Red = genuine structural anomaly, white = fits model = healthy or gradual stress
Map.addLayer(residZMin.clip(aoi),
  {min: -5, max: 0, palette: ['d73027', 'f46d43', 'fdae61', 'ffffbf', 'ffffff']},
  'Harmonic residual z-score min (red = anomalous)', false);

Map.addLayer(alertCount.updateMask(alertCount.gt(0)).clip(aoi),
  {min: 1, max: 10, palette: ['ffffcc', 'feb24c', 'f03b20']},
  'Alert scene count (persistence)', false);

Map.addLayer(persistentAlert.selfMask().clip(aoi),
  {palette: ['FFA500'], opacity: 0.6},
  'Candidate alerts (pre filters)', false);

Map.addLayer(cleanAlert.selfMask().clip(aoi),
  {palette: ['FF0000'], opacity: 0.9},
  'FINAL ALERTS — new disturbance 2025', true);

Map.addLayer(firstAlertDay.clip(aoi),
  {min: 0, max: 365, palette: ['d94701', 'fd8d3c', 'ffffcc']},
  'First alert day (orange = earlier)', false);

Map.addLayer(
  ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), 0, 2),
  {palette: ['00ffff']}, 'Sumava NP boundary');


// =============================================================================
// 11. CONSOLE STATS
// =============================================================================

var forestKm2 = ee.Number(
  coreForest.multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 10, maxPixels: 1e10})
    .get('forest')
).divide(1e6).round();

var degradedHa = ee.Number(
  alreadyDegraded.selfMask()
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 200, maxPixels: 1e10, bestEffort: true})
    .get('already_degraded')
).divide(1e4).round();

var persistHa = ee.Number(
  persistentAlert.selfMask()
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 200, maxPixels: 1e10, bestEffort: true})
    .get('persistent_alert')
).divide(1e4).round();

var newAlertHa = ee.Number(
  newAlert.selfMask()
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 200, maxPixels: 1e10, bestEffort: true})
    .get('persistent_alert')
).divide(1e4).round();

var meanScenes = alertCount.updateMask(newAlert)
  .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi,
                 scale: 200, maxPixels: 1e10, bestEffort: true})
  .get('alert_count');

print('--- Stats ---');
print('Core forest area (km²):', forestKm2);
print('Already degraded in 2024, excluded (ha):', degradedHa);
print('Candidate alerts, pre-filters (ha):', persistHa);
print('New disturbance alerts, pre-patch (ha):', newAlertHa);
print('Mean scene count per alert pixel:', meanScenes);
print('Note: exact final area (post patch) from exported polygon file');


// =============================================================================
// 12. EXPORT
// =============================================================================

var yr = CONFIG.detectStart.slice(0, 4);

function exportImg(img, name) {
  Export.image.toDrive({
    image: img.toFloat(), description: name,
    folder: CONFIG.exportFolder, fileNamePrefix: name,
    region: aoi, scale: CONFIG.exportScale,
    crs: CONFIG.exportCRS, maxPixels: 1e10, fileFormat: 'GeoTIFF'
  });
}

exportImg(cleanAlert,                          'sumava_alerts_'       + yr);
exportImg(residZMin.clip(aoi),                 'sumava_residual_z_'   + yr);
exportImg(alertCount.clip(aoi),                'sumava_scene_count_'  + yr);
exportImg(firstAlertDay.unmask(0).clip(aoi),   'sumava_first_day_'    + yr);
exportImg(rmse.clip(aoi),                      'sumava_rmse_'         + yr);
exportImg(alreadyDegraded.selfMask().clip(aoi),'sumava_degraded_2024_'+ yr);

// Polygon export removed — reduceToVectors on the harmonic pipeline
// exceeds GEE memory at any scale due to the deep computation graph.
//
// INSTEAD: vectorise locally in QGIS after downloading the raster:
//   1. Download sumava_alerts_2025.tif from Google Drive
//   2. QGIS → Raster → Conversion → Polygonize (Raster to Vector)
//      Input: sumava_alerts_2025.tif
//      Field name: alert
//      Check "Use 8-connectedness"
//   3. Open attribute table → select features where alert = 1 → save as GeoJSON
//   4. Add area_ha field: open field calculator → $area / 10000
//
// The exported raster (sumava_alerts_2025.tif) is the primary output.
// All accuracy assessment and visual validation can be done from the raster
// directly in QGIS alongside Google Satellite / S2 August 2025 imagery.

print('Note: vectorise sumava_alerts_2025.tif locally in QGIS (see comments above)');

print('=== Exports queued — run from Tasks panel ===');