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
  minScenes: 2,

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

var aoi = ee.Geometry.Polygon([[
  [13.23115,49.12411],[13.23606,49.11372],[13.27640,49.12048],
  [13.28918,49.11864],[13.34418,49.08889],[13.34641,49.08206],
  [13.37049,49.06746],[13.37622,49.05826],[13.39629,49.05154],
  [13.39787,49.04579],[13.39167,49.04215],[13.39966,49.03707],
  [13.40583,49.02385],[13.40029,49.01567],[13.40941,49.00322],
  [13.40227,48.99444],[13.40272,48.98722],[13.42439,48.97741],
  [13.42618,48.97249],[13.45926,48.96267],[13.49577,48.94149],
  [13.50827,48.94214],[13.50706,48.96912],[13.52927,48.97405],
  [13.54806,48.96688],[13.58390,48.96921],[13.59300,48.96089],
  [13.59034,48.95297],[13.61025,48.93862],[13.60792,48.94351],
  [13.62860,48.94924],[13.63112,48.94700],[13.62249,48.93881],
  [13.63802,48.92569],[13.63802,48.91923],[13.65548,48.89358],
  [13.66965,48.89051],[13.67144,48.88015],[13.71687,48.87814],
  [13.73054,48.88712],[13.73794,48.88602],[13.73730,48.87934],
  [13.75061,48.86683],[13.74945,48.85965],[13.76444,48.83448],
  [13.79295,48.83012],[13.78818,48.82485],[13.81525,48.79709],
  [13.80355,48.78086],[13.81320,48.77402],[13.87613,48.76661],
  [13.91022,48.74750],[13.93917,48.72324],[13.95371,48.72074],
  [13.95589,48.71442],[13.98213,48.72058],[13.97977,48.73080],
  [13.97278,48.73298],[13.96853,48.74391],[13.92272,48.76981],
  [13.95167,48.79387],[13.96227,48.79577],[13.96643,48.82083],
  [13.94190,48.84613],[13.91158,48.86257],[13.89953,48.87717],
  [13.89564,48.89016],[13.88363,48.89959],[13.87344,48.90038],
  [13.86671,48.89275],[13.85745,48.89470],[13.84173,48.90626],
  [13.82501,48.90906],[13.82398,48.91639],[13.81244,48.92207],
  [13.79032,48.91356],[13.77428,48.91371],[13.76833,48.90807],
  [13.72096,48.90702],[13.72223,48.91334],[13.70721,48.93111],
  [13.70936,48.94627],[13.71518,48.94707],[13.71385,48.95735],
  [13.70099,48.96714],[13.68369,48.96742],[13.67245,48.97534],
  [13.66393,48.99080],[13.67209,49.00270],[13.66691,49.02621],
  [13.62685,49.03266],[13.62667,49.04386],[13.61972,49.05657],
  [13.60452,49.05952],[13.60669,49.07734],[13.61410,49.08209],
  [13.60473,49.08638],[13.60123,49.09920],[13.58730,49.11700],
  [13.57728,49.12040],[13.57607,49.10765],[13.53489,49.13572],
  [13.52361,49.13990],[13.50946,49.13807],[13.51223,49.14358],
  [13.48910,49.14334],[13.46729,49.13631],[13.44514,49.13775],
  [13.43621,49.14612],[13.44433,49.15502],[13.43555,49.15616],
  [13.40841,49.17376],[13.36871,49.18225],[13.34450,49.17416],
  [13.30995,49.19148],[13.30380,49.18948],[13.30653,49.18582],
  [13.29512,49.17867],[13.29063,49.16805],[13.26024,49.15472],
  [13.24888,49.14181],[13.24774,49.13761],[13.25200,49.13773],
  [13.24738,49.12903],[13.23115,49.12411]
]], null, false);

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
// 3. FOREST MASK — ESA WorldCover 2021
// =============================================================================

var coreForest = ee.ImageCollection('ESA/WorldCover/v200')
  .first().select('Map').clip(aoi)
  .eq(10)
  .focalMin({radius: 1, kernelType: 'square', units: 'pixels'})
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
// 7. ALREADY-DEGRADED MASK
// =============================================================================
//
// Pixels already anomalous in 2024 (z < -1.5) are excluded.
// Uses the same harmonic model — consistent with detection methodology.

var archive2024Residuals = loadS2Season('2024-01-01', '2025-01-01')
  .map(predictNBR);

var alreadyDegraded = archive2024Residuals
  .mean()
  .divide(rmse)
  .lt(-1.5)
  .rename('already_degraded');

print('Already-degraded mask built');


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
                   scale: 100, maxPixels: 1e10, bestEffort: true})
    .get('already_degraded')
).divide(1e4).round();

var persistHa = ee.Number(
  persistentAlert.selfMask()
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 100, maxPixels: 1e10, bestEffort: true})
    .get('persistent_alert')
).divide(1e4).round();

var newAlertHa = ee.Number(
  newAlert.selfMask()
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 100, maxPixels: 1e10, bestEffort: true})
    .get('persistent_alert')
).divide(1e4).round();

var meanScenes = alertCount.updateMask(newAlert)
  .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi,
                 scale: 100, maxPixels: 1e10, bestEffort: true})
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

print('=== Exports queued — run from Tasks panel ===');