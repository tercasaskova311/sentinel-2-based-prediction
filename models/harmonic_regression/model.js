
//HARMONIC REGRESSION MODEL
// to be run in google earth engine console: https://code.earthengine.google.com
// Harmonic model baseline · Sentinel-2 NBR · Zero external labels
// Based on CCDC (Zhu & Woodcock, 2014)

var DETECT_YEAR = 2024;

var CONFIG = {
  archiveStart: '2020-01-01',
  archiveEnd:   DETECT_YEAR + '-01-01',
  detectStart:  DETECT_YEAR + '-01-01',
  detectEnd:    DETECT_YEAR + '-12-31',
  compareYear:  DETECT_YEAR - 1,
  seasonStart:  4,
  seasonEnd:    11,
  numHarmonics:    2,
  zThreshResidual: -2.5,
  minScenes:       2,
  minPatchPx:      30,
  maxCloudPct:     50,
  exportScale:  10,
  exportCRS:    'EPSG:32633',
  exportFolder: 'GEE_Sumava',
};

print('=== Detection year:', DETECT_YEAR, '===');
print('Archive:', CONFIG.archiveStart, '→', CONFIG.archiveEnd);
print('Detection:', CONFIG.detectStart, '→', CONFIG.detectEnd);
print('AOI area (km²):', aoi.area(1).divide(1e6).round());

// 2. SENTINEL-2 PREPROCESSING
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

// 3. FOREST MASK — ESA WorldCover 2021 + NDVI confirmation
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

var s2Archive   = loadS2Full(CONFIG.archiveStart, CONFIG.archiveEnd);
var s2Detection = loadS2Season(CONFIG.detectStart, CONFIG.detectEnd);

print('S2 archive images (full year):', s2Archive.size());
print('S2 detection images (season):',  s2Detection.size());

// 5. HARMONIC MODEL
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

// 6. COMPUTE RMSE OVER ARCHIVE
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


// 8. SCORE DETECTION IMAGES
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


// 9. DETECTION
// 1. Persistent negative anomaly
var persistentAlert = alertCount.gte(CONFIG.minScenes).rename('persistent_alert');

// 2. Forest mask + already-degraded mask
var degradedMask = coefficients.select('t').divide(rmse).lt(-0.5);

var cleanAlert = persistentAlert
  .updateMask(coreForest)
  .updateMask(degradedMask.not())
  .rename('clean_alert');

// 3. Patch filter
var patchSize = cleanAlert.selfMask()
  .connectedPixelCount({ maxSize: 512, eightConnected: true });

var finalAlert = cleanAlert
  .updateMask(patchSize.gte(CONFIG.minPatchPx))
  .rename('final_alert');

print('Detection complete');

// 10. VISUALISATION
Map.centerObject(aoi, 11);

Map.addLayer(coreForest,
  {palette: ['1a9641'], opacity: 0.3},
  'Forest mask', false);


Map.addLayer(residZMin.clip(aoi),
  {min: -5, max: 0, palette: ['d73027', 'f46d43', 'fdae61', 'ffffbf', 'ffffff']},
  'Residual z-score min', false);

Map.addLayer(alertCount.updateMask(alertCount.gt(0)).clip(aoi),
  {min: 1, max: 10, palette: ['ffffcc', 'feb24c', 'f03b20']},
  'Alert scene count', false);

Map.addLayer(persistentAlert.selfMask().clip(aoi),
  {palette: ['FFA500'], opacity: 0.6},
  'Persistent alerts (pre-forest-mask)', false);

Map.addLayer(finalAlert.selfMask().clip(aoi),
  {palette: ['FF6600'], opacity: 0.8},
  'Final alerts (all disturbance) ' + DETECT_YEAR, false);


Map.addLayer(firstAlertDayRaw.updateMask(finalAlert).clip(aoi),
  {min: 0, max: 365, palette: ['d94701', 'fd8d3c', 'ffffcc']},
  'First alert day', false);

Map.addLayer(
  ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), 0, 2),
  {palette: ['00ffff']}, 'Sumava NP boundary');

// 12.  STATS
function countHa(img, bandName, sc) {
  return ee.Number(
    img.selfMask()
      .multiply(ee.Image.pixelArea())
      .reduceRegion({
        reducer:   ee.Reducer.sum(),
        geometry:  aoi,
        scale:     sc || 200,
        maxPixels: 1e10,
        bestEffort: true
      })
      .get(bandName)
  ).divide(1e4).round();
}

print('--- Stats ---');
print('Detection year:', DETECT_YEAR);
print('Core forest area (km²):',
  ee.Number(coreForest.multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e10})
    .get('forest')).divide(1e6).round());

print('Persistent alerts pre-forest-mask (ha):',
  countHa(persistentAlert.rename('persistent_alert'), 'persistent_alert'));
print('Clean alerts post-forest-mask (ha):',
  countHa(cleanAlert, 'clean_alert'));
print('Final alerts post-patch-filter (ha):',
  countHa(finalAlert, 'final_alert'));

print('Mean scene count per final alert pixel:',
  alertCount.updateMask(finalAlert)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi,
                   scale: 200, maxPixels: 1e10, bestEffort: true})
    .get('alert_count'));

// 13. EXPORT
var yr = String(DETECT_YEAR);

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

exportImg(finalAlert,                           'sumava_alerts_'        + yr);
exportImg(residZMin.clip(aoi),                  'sumava_residual_z_'    + yr);
exportImg(alertCount.clip(aoi),                 'sumava_scene_count_'   + yr);
exportImg(firstAlertDayRaw.unmask(0).clip(aoi), 'sumava_first_day_'     + yr);

