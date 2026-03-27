/ =============================================================================
// ŠUMAVA NP — FOREST DISTURBANCE DETECTION
// Harmonic model baseline · Sentinel-2 NBR · Zero external labels
//
// Based on CCDC (Zhu & Woodcock, 2014)
// Forest mask: ESA WorldCover 2021 + NDVI confirmation
// Already-degraded mask: harmonic trend coefficient c1/RMSE < threshold
// Export: 10m resolution (matches S2 native, fixes patch filter mismatch)
// =============================================================================


// =============================================================================
// 0. CONFIGURATION
// =============================================================================

// ← SWITCH HERE: change DETECT_YEAR to 2024 or 2025
var DETECT_YEAR = 2024;

var CONFIG = {
  archiveStart: '2020-01-01',
  archiveEnd:   DETECT_YEAR + '-01-01',  // archive always ends just before detection year

  detectStart:  DETECT_YEAR + '-01-01',
  detectEnd:    DETECT_YEAR + '-12-31',

  // Comparison year for visual validation
  // Shows S2 imagery from year before detection to confirm change is new
  compareYear:  DETECT_YEAR - 1,

  seasonStart: 4,
  seasonEnd:   11,

  numHarmonics:    2,
  zThreshResidual: -2.5,
  minScenes:       3,
  minPatchPx:      50,
  maxCloudPct:     50,

  exportScale:  10,
  exportCRS:    'EPSG:32633',
  exportFolder: 'GEE_Sumava',
};

print('=== Detection year:', DETECT_YEAR, '===');
print('Archive:', CONFIG.archiveStart, '→', CONFIG.archiveEnd);
print('Detection:', CONFIG.detectStart, '→', CONFIG.detectEnd);


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

function loadS2Full(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
    .map(scaleS2)
    .map(maskClouds)
    .map(addNBR);
}

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
  .gt(0.5);

var coreForest = ee.ImageCollection('ESA/WorldCover/v200')
  .first().select('Map').clip(aoi)
  .eq(10)
  .and(ndviMask)
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

var s2Archive   = loadS2Full(CONFIG.archiveStart, CONFIG.archiveEnd);
var s2Detection = loadS2Season(CONFIG.detectStart, CONFIG.detectEnd);

print('S2 archive images (full year):', s2Archive.size());
print('S2 detection images (season):',  s2Detection.size());


// =============================================================================
// 5. HARMONIC MODEL
// =============================================================================

var referenceDate = ee.Date('2020-01-01');
var omega = 2.0 * Math.PI;

function addHarmonicBands(img) {
  var t = ee.Image(
    img.date().difference(referenceDate, 'year')
  ).float().rename('t');

  var bands = [
    ee.Image.constant(1).float().rename('constant'),
    t,
    t.multiply(omega).sin().rename('sin1'),
    t.multiply(omega).cos().rename('cos1'),
  ];

  if (CONFIG.numHarmonics >= 2) {
    bands.push(t.multiply(2 * omega).sin().rename('sin2'));
    bands.push(t.multiply(2 * omega).cos().rename('cos2'));
  }

  return img.addBands(bands);
}

var harmonicArchive = s2Archive.map(addHarmonicBands);

var predictors = CONFIG.numHarmonics >= 2
  ? ['constant', 't', 'sin1', 'cos1', 'sin2', 'cos2']
  : ['constant', 't', 'sin1', 'cos1'];

var harmonicFit = harmonicArchive
  .select(predictors.concat(['NBR']))
  .reduce(ee.Reducer.linearRegression({
    numX: predictors.length,
    numY: 1
  }));

var coefficientsFull = harmonicFit.select('coefficients')
  .arrayFlatten([predictors, ['NBR']]);

var predictorsNBR = predictors.map(function(p) { return p + '_NBR'; });
var coefficients  = coefficientsFull.select(predictorsNBR, predictors);

print('Harmonic model fitted — predictors:', predictors.length);


// =============================================================================
// 6. COMPUTE RMSE OVER ARCHIVE
// =============================================================================

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

  var predicted = predictorImg.select(predictors)
    .multiply(coefficients.select(predictors))
    .reduce(ee.Reducer.sum())
    .rename('NBR_predicted');

  var residual = img.select('NBR')
    .subtract(predicted)
    .rename('NBR_residual');

  return residual.copyProperties(img, ['system:time_start']);
}

var archiveResiduals = s2Archive.map(predictNBR);

var rmse = archiveResiduals
  .map(function(img) { return img.pow(2); })
  .mean()
  .sqrt()
  .max(0.005)
  .rename('RMSE');

print('RMSE computed');


// =============================================================================
// 7. ALREADY-DEGRADED MASK
// =============================================================================

var trendCoeff = coefficients.select('t');

var alreadyDegraded = trendCoeff.divide(rmse)
  .lt(-0.8)
  .rename('already_degraded');

print('Already-degraded mask built (c1/RMSE < -0.8)');


// =============================================================================
// 8. SCORE DETECTION IMAGES
// =============================================================================

var detectStartMs = ee.Date(CONFIG.detectStart).millis();

function computeResidualZ(img) {
  var t = ee.Image(
    img.date().difference(referenceDate, 'year')
  ).float();

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

var monthlyAlertCounts = [];
var monthlyResidZMins  = [];
var monthlyFirstDays   = [];

for (var sm = CONFIG.seasonStart; sm <= CONFIG.seasonEnd; sm++) {
  var mResidZ = s2ResidZ.filter(ee.Filter.calendarRange(sm, sm, 'month'));

  var mAlerts = mResidZ.map(function(img) {
    return img.lt(CONFIG.zThreshResidual).rename('alert').toFloat()
      .set('system:time_start', img.get('system:time_start'));
  });

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

var newAlert = persistentAlert
  .updateMask(coreForest)
  .updateMask(alreadyDegraded.not())
  .rename('new_alert');

var patchSize = newAlert.selfMask()
  .connectedPixelCount({
    maxSize: CONFIG.minPatchPx + 1,
    eightConnected: true
  });

var cleanAlert = newAlert
  .updateMask(patchSize.gte(CONFIG.minPatchPx))
  .rename('clean_alert');

var firstAlertDay = firstAlertDayRaw.updateMask(cleanAlert);

// =============================================================================
// 9b. TEMPORAL SPREAD FILTER — separates abrupt from gradual decline
// =============================================================================

// firstAlertDay: day-of-year relative to detectStart (0 = Apr 1 if seasonStart=4)
// Abrupt events (clearcuts, windthrow): first alert fires early, stays flagged
// Gradual decline (bark-beetle): threshold crossed only mid-to-late season

// Days from season start when alert first fires
// Earlier = more abrupt
var seasonLengthDays = 214; // Apr 1 → Oct 31 approx

// Fraction through season when first alert fires (0 = Apr, 1 = Oct)
var firstAlertFraction = firstAlertDayRaw.divide(seasonLengthDays);

// --- Option B: keep based on how SUSTAINED the drop is
// Deep z-score that appears early = abrupt total loss
// Shallow z-score late = gradual decline
// Combine both signals: early AND deep
var abruptAlert = cleanAlert
  .updateMask(firstAlertFraction.lt(0.55))      // alert fires before mid-season
  .updateMask(residZMin.lt(-3.0))               // and is genuinely deep
  .rename('abrupt_alert');
  
// =============================================================================
// 9b. CLEARCUT CONFIRMATION — low NDVI in detection year
// =============================================================================

var ndviDetect = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(DETECT_YEAR + '-06-01', DETECT_YEAR + '-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(scaleS2).map(maskClouds)
  .map(function(img) {
    return img.select('B8').subtract(img.select('B4'))
      .divide(img.select('B8').add(img.select('B4')))
      .rename('NDVI');
  })
  .median();

var clearcuts = cleanAlert
  .updateMask(ndviDetect.lt(0.4))
  .rename('clearcut_alert');

var firstAlertDayClearcut = firstAlertDay.updateMask(clearcuts);

  
// =============================================================================
// 10. VISUALISATION
// =============================================================================

Map.centerObject(aoi, 11);

// ── Detection year composite ──
var s2TC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(CONFIG.detectStart, CONFIG.detectEnd)
  .filter(ee.Filter.calendarRange(CONFIG.seasonStart, CONFIG.seasonEnd, 'month'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPct))
  .map(scaleS2).map(maskClouds)
  .select(['B4', 'B3', 'B2']).median().clip(aoi);

Map.addLayer(s2TC,
  {min: 0, max: 0.3, gamma: 1.4},
  'S2 true colour ' + DETECT_YEAR + ' (detection year)', true);

// ── Comparison: previous year composite ──
// Load same season from DETECT_YEAR-1 to visually confirm change is new
var s2PrevTC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(CONFIG.compareYear + '-04-01', CONFIG.compareYear + '-11-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(scaleS2).map(maskClouds)
  .select(['B4', 'B3', 'B2']).median().clip(aoi);

Map.addLayer(s2PrevTC,
  {min: 0, max: 0.25, gamma: 1.3},
  'S2 true colour ' + CONFIG.compareYear + ' (year before — compare)', false);

var s2PrevFC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(CONFIG.compareYear + '-04-01', CONFIG.compareYear + '-11-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(scaleS2).map(maskClouds)
  .select(['B8', 'B4', 'B3']).median().clip(aoi);

Map.addLayer(s2PrevFC,
  {min: 0, max: 0.4},
  'S2 false colour ' + CONFIG.compareYear + ' NIR-R-G (compare)', false);

// ── Validation: late season detection year ──
var s2ValTC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(DETECT_YEAR + '-08-01', DETECT_YEAR + '-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(scaleS2).map(maskClouds)
  .select(['B4', 'B3', 'B2']).median().clip(aoi);

Map.addLayer(s2ValTC,
  {min: 0, max: 0.25, gamma: 1.3},
  'S2 true colour Aug-Oct ' + DETECT_YEAR + ' (validation)', false);

var s2ValFC = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(DETECT_YEAR + '-08-01', DETECT_YEAR + '-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(scaleS2).map(maskClouds)
  .select(['B8', 'B4', 'B3']).median().clip(aoi);

Map.addLayer(s2ValFC,
  {min: 0, max: 0.4},
  'S2 false colour Aug-Oct ' + DETECT_YEAR + ' NIR-R-G (validation)', false);

// ── Diagnostic layers ──
Map.addLayer(coreForest,
  {palette: ['1a9641'], opacity: 0.3},
  'Forest mask (WorldCover 2021)', false);

Map.addLayer(alreadyDegraded.selfMask().clip(aoi),
  {palette: ['8B4513'], opacity: 0.5},
  'Already degraded — excluded (c1/RMSE < -0.8)', false);

Map.addLayer(residZMin.clip(aoi),
  {min: -5, max: 0, palette: ['d73027', 'f46d43', 'fdae61', 'ffffbf', 'ffffff']},
  'Harmonic residual z-score min (red = anomalous)', false);

Map.addLayer(alertCount.updateMask(alertCount.gt(0)).clip(aoi),
  {min: 1, max: 10, palette: ['ffffcc', 'feb24c', 'f03b20']},
  'Alert scene count (persistence)', false);

Map.addLayer(persistentAlert.selfMask().clip(aoi),
  {palette: ['FFA500'], opacity: 0.6},
  'Candidate alerts (pre-filters)', false);

Map.addLayer(clearcuts.selfMask().clip(aoi),
  {palette: ['FF0000'], opacity: 0.9},
  'FINAL ALERTS — clearcuts only ' + DETECT_YEAR, true);

Map.addLayer(firstAlertDayClearcut.clip(aoi),
  {min: 0, max: 365, palette: ['d94701', 'fd8d3c', 'ffffcc']},
  'First alert day — clearcuts only', false);
Map.addLayer(firstAlertDay.clip(aoi),
  {min: 0, max: 365, palette: ['d94701', 'fd8d3c', 'ffffcc']},
  'First alert day (orange = earlier)', false);

Map.addLayer(
  ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), 0, 2),
  {palette: ['00ffff']}, 'Sumava NP boundary');


// =============================================================================
// 11. VALIDATION — random sample points
// =============================================================================

var alertPoints = cleanAlert.selfMask()
  .sample({
    region:     aoi,
    scale:      50,
    numPixels:  30,
    seed:       42,
    geometries: true,
    tileScale:  4
  });

var nonAlertPoints = cleanAlert.unmask(0).eq(0)
  .updateMask(coreForest)
  .sample({
    region:     aoi,
    scale:      50,
    numPixels:  30,
    seed:       123,
    geometries: true,
    tileScale:  4
  });

Map.addLayer(alertPoints,
  {color: 'FF0000'},
  'Validation — alert points (TP/FP check)', false);

Map.addLayer(nonAlertPoints,
  {color: '0000FF'},
  'Validation — non-alert points (FN check)', false);

print('Validation points generated — 30 alert, 30 non-alert');
print('For each red point: toggle ' + DETECT_YEAR + ' vs ' + CONFIG.compareYear + ' composites');
print('If canopy present in ' + CONFIG.compareYear + ' and gone in ' + DETECT_YEAR + ' → TP');
print('If canopy present in both years → FP (false alarm)');
print('If canopy gone in ' + CONFIG.compareYear + ' already → already degraded (expected miss)');


// =============================================================================
// 12. CONSOLE STATS
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
    .get('new_alert')
).divide(1e4).round();

var meanScenes = alertCount.updateMask(newAlert)
  .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi,
                 scale: 200, maxPixels: 1e10, bestEffort: true})
  .get('alert_count');

print('--- Stats ---');
print('Detection year:', DETECT_YEAR);
print('Core forest area (km²):', forestKm2);
print('Already degraded, excluded (ha):', degradedHa);
print('Candidate alerts, pre-filters (ha):', persistHa);
print('New disturbance alerts, pre-patch (ha):', newAlertHa);
print('Mean scene count per alert pixel:', meanScenes);
print('Note: exact final area (post-patch) from exported raster at 10m');


// =============================================================================
// 13. EXPORT
// =============================================================================

var yr = String(DETECT_YEAR);

function exportImg(img, name) {
  Export.image.toDrive({
    image:           img.toFloat(),
    description:     name,
    folder:          CONFIG.exportFolder,
    fileNamePrefix:  name,
    region:          aoi,
    scale:           CONFIG.exportScale,
    crs:             CONFIG.exportCRS,
    maxPixels:       1e10,
    fileFormat:      'GeoTIFF'
  });
}

exportImg(cleanAlert,                           'sumava_alerts_'        + yr);
exportImg(residZMin.clip(aoi),                  'sumava_residual_z_'    + yr);
exportImg(alertCount.clip(aoi),                 'sumava_scene_count_'   + yr);
exportImg(firstAlertDay.unmask(0).clip(aoi),    'sumava_first_day_'     + yr);
exportImg(rmse.clip(aoi),                       'sumava_rmse_'          + yr);
exportImg(alreadyDegraded.selfMask().clip(aoi), 'sumava_degraded_mask_' + yr);
exportImg(clearcuts, 'sumava_clearcuts_' + yr);

print('=== Exports queued — run from Tasks panel ===');