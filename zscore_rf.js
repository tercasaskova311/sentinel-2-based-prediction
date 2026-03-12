// to be run in google console
// ═══════════════════════════════════════════════════════════════════════════════
// ŠUMAVA DEFORESTATION ALERT SYSTEM — v1.0
// Two-layer architecture:
//   Layer 1: NBR z-score anomaly detection against 2019-2024 monthly baseline
//   Layer 2: Random Forest trained on z-score-derived labels (self-supervised)
//            using single-image features (spectral + red-edge + texture)
//
// Usage:
//   1. Set TARGET_DATE to any date you want to analyse
//   2. Set CLOUD_THRESHOLD if needed (default 20%)
//   3. Run — results appear as map layers and console metrics
//
// Reference methodology:
//   Senf & Seidl (2021) Nature Communications — spectral anomaly baseline
//   Verhelst et al. (2022) Remote Sensing — RF on change pairs
// ═══════════════════════════════════════════════════════════════════════════════

// ── CONFIG — change these ────────────────────────────────────────────────────
var TARGET_DATE      = '2024-08-15'; // date of new image to analyse
var CLOUD_THRESHOLD  = 20;           // max cloud % for S2 scene selection
var WINDOW_DAYS      = 15;           // ±days around target date to find clean scenes
var Z_MODERATE       = -1.5;         // z-score threshold: moderate disturbance
var Z_SEVERE         = -2.5;         // z-score threshold: severe / complete loss
var BASELINE_YEARS   = [2019, 2020, 2021, 2022, 2023]; // baseline period

// NOTE: target year is automatically excluded from baseline to prevent self-comparison
var RF_TREES         = 100;
var RF_SAMPLES       = 300;          // samples per class for RF training
// ─────────────────────────────────────────────────────────────────────────────

// ── AOI: Šumava NP boundary (125-pt simplified polygon) ──────────────────────
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

Map.setCenter(13.6, 48.95, 11);
Map.addLayer(aoi, {color: 'cyan'}, 'Šumava NP boundary');

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 1: Forest mask
// Hansen treecover2000 > 30% + WorldCover 2021 tree class + no pre-2019 loss
// ═══════════════════════════════════════════════════════════════════════════════
var hansen    = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var treecover = ee.Image(hansen.select('treecover2000'));
var datamask  = ee.Image(hansen.select('datamask'));
var lossYearU = ee.Image(hansen.select('lossyear')).unmask(0);
var wcTree    = ee.Image('ESA/WorldCover/v200/2021').select('Map').eq(10);

var wasForest  = treecover.gt(30);
var preLoss    = lossYearU.gte(1).and(lossYearU.lte(18));
var forestMask = wasForest.and(datamask.eq(1)).and(wcTree).and(preLoss.not());

Map.addLayer(forestMask.selfMask(),
  {palette: ['#1a5c1a']}, 'Forest mask', false);

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 2: Feature extraction from a single S2 image
// All features must be computable from ONE image — this is what makes the
// system real-time capable. No seasonal composite needed.
// Features:
//   - Core indices: NBR, NDVI, NDWI
//   - Red-edge: B5/B6/B7 normalised differences (sensitive to early stress)
//   - SWIR ratio: B11/B12 (moisture loss signal)
//   - Texture: GLCM on NIR (B8) — contrast, entropy, dissimilarity
//   - Raw bands: B4, B5, B6, B7, B8, B11, B12
// ═══════════════════════════════════════════════════════════════════════════════
function extractFeatures(img) {
  // Core spectral indices
  var nbr  = img.normalizedDifference(['B8','B12']).rename('NBR');
  var ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI');
  var ndwi = img.normalizedDifference(['B8','B11']).rename('NDWI');

  // Red-edge indices — highly sensitive to chlorophyll loss before visible browning
  var ndre1 = img.normalizedDifference(['B8','B5']).rename('NDRE1'); // B8-B5
  var ndre2 = img.normalizedDifference(['B8','B6']).rename('NDRE2'); // B8-B6
  var ndre3 = img.normalizedDifference(['B8','B7']).rename('NDRE3'); // B8-B7
  var cire  = img.select('B7').divide(img.select('B5')).subtract(1).rename('CIre'); // chlorophyll index red-edge

  // SWIR ratio — responds to canopy moisture loss after mortality
  var swirRatio = img.select('B11').divide(img.select('B12')).rename('SWIR_ratio');

  // Raw bands (scaled to reflectance)
  var bands = img.select(['B4','B5','B6','B7','B8','B11','B12'])
    .divide(10000);

  // GLCM texture on NIR (B8) — clearcut vs bark-beetle vs windthrow are structurally distinct
  var nir8bit = img.select('B8').divide(10000)
    .unitScale(0, 0.5).multiply(255).toByte();
  var glcm = nir8bit.glcmTexture({size: 3});
  var texture = ee.Image.cat([
    glcm.select('B8_contrast').rename('tex_contrast'),
    glcm.select('B8_ent').rename('tex_entropy'),
    glcm.select('B8_diss').rename('tex_diss'),
    glcm.select('B8_corr').rename('tex_corr')
  ]);

  return ee.Image.cat([
    bands, nbr, ndvi, ndwi,
    ndre1, ndre2, ndre3, cire,
    swirRatio, texture
  ]).updateMask(forestMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 3: Get the target image
// Best available S2 scene within ±WINDOW_DAYS of TARGET_DATE
// ═══════════════════════════════════════════════════════════════════════════════
var targetDate = ee.Date(TARGET_DATE);
var targetCol  = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(
    targetDate.advance(-WINDOW_DAYS, 'day'),
    targetDate.advance( WINDOW_DAYS, 'day'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
  .sort('CLOUDY_PIXEL_PERCENTAGE'); // least cloudy first

var targetImage = ee.Image(targetCol.first());
var targetMonth = targetDate.get('month');

print('Target image date:',
  targetImage.date().format('YYYY-MM-dd'));
print('Target image cloud %:',
  targetImage.get('CLOUDY_PIXEL_PERCENTAGE'));
print('Scenes available in window:', targetCol.size());

var targetFeatures = extractFeatures(targetImage);
var featureBands   = targetFeatures.bandNames();
print('Feature bands:', featureBands);

// Visualise true colour of target image
Map.addLayer(
  targetImage.select(['B4','B8','B11']).divide(10000),
  {min:0, max:0.3}, 'Target image (false colour NIR/Red/SWIR)');

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 4: Build per-pixel monthly NBR baseline (2019–2024)
// For each baseline year, take the same-month composite as the target
// This ensures sun angle and phenology are comparable
// ═══════════════════════════════════════════════════════════════════════════════
// Exclude the target year from baseline to prevent self-comparison
var targetYearJS = parseInt(TARGET_DATE.slice(0, 4));
var baselineYearsFiltered = BASELINE_YEARS.filter(function(y){ return y !== targetYearJS; });
print('Baseline years used:', baselineYearsFiltered);

var baselineNBRs = ee.ImageCollection(
  baselineYearsFiltered.map(function(y) {
    var start = ee.Date.fromYMD(y, targetMonth, 1);
    var end   = start.advance(1, 'month');
    var col   = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate(start, end)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD));
    var nbr = ee.Image(
      ee.Algorithms.If(
        col.size().gt(0),
        col.median().normalizedDifference(['B8','B12']),
        ee.Image.constant(0).selfMask()  // masked if no data
      )
    ).rename('NBR').updateMask(forestMask);
    return nbr;
  })
);

var baselineMedian = baselineNBRs.median().rename('baseline_median');
var baselineStddev = baselineNBRs.reduce(ee.Reducer.stdDev())
  .max(0.02)  // minimum stddev to prevent division explosion
  .rename('baseline_stddev');

Map.addLayer(baselineMedian,
  {min:0.1, max:0.8, palette:['#d73027','#ffffbf','#1a9641']},
  'NBR baseline median', false);

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 5: LAYER 1 — Z-score anomaly detection
// z = (NBR_target - baseline_median) / baseline_stddev
// ═══════════════════════════════════════════════════════════════════════════════
var targetNBR = targetImage.normalizedDifference(['B8','B12'])
  .rename('NBR').updateMask(forestMask);

var zscore = targetNBR.subtract(baselineMedian)
  .divide(baselineStddev)
  .rename('zscore');

// Binary alert layers
var alertModerate = zscore.lt(Z_MODERATE)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .selfMask().updateMask(forestMask);

var alertSevere = zscore.lt(Z_SEVERE)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .selfMask().updateMask(forestMask);

Map.addLayer(zscore,
  {min:-4, max:2, palette:['#d7191c','#fdae61','#ffffbf','#a6d96a','#1a9641']},
  'Z-score (red = anomaly)');
Map.addLayer(alertModerate,
  {palette:['#fd8d3c']}, 'ALERT: moderate disturbance (z < -1.5)');
Map.addLayer(alertSevere,
  {palette:['#cc0000']}, 'ALERT: severe / complete loss (z < -2.5)');

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 6: LAYER 2 — Self-supervised RF
// Labels generated from z-score — no Hansen, no manual digitising
// Positive (loss):   z < -2.5  (severe anomaly = likely complete canopy removal)
// Negative (stable): z > -0.5  (stable forest well within normal range)
// Trained on single-image features → can classify any future single image
// ═══════════════════════════════════════════════════════════════════════════════

// Generate labels from z-score
// Adaptive threshold: start strict, loosen if not enough loss pixels exist
// This prevents the "only one class" RF crash on low-disturbance images
var lossLabel_strict   = zscore.lt(-2.5).and(forestMask);
var lossLabel_moderate = zscore.lt(-1.5).and(forestMask);
var lossLabel_loose    = zscore.lt(-1.0).and(forestMask);

// Count loss pixels at each threshold to pick the right one
var lossCount_strict   = lossLabel_strict.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: aoi, scale: 100, maxPixels: 1e9
});
var lossCount_moderate = lossLabel_moderate.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: aoi, scale: 100, maxPixels: 1e9
});

// Use strict if enough pixels, fall back to moderate, then loose
var enoughStrict   = ee.Number(lossCount_strict.values().get(0)).gte(50);
var enoughModerate = ee.Number(lossCount_moderate.values().get(0)).gte(50);

var lossLabel = ee.Image(
  ee.Algorithms.If(enoughStrict,   lossLabel_strict,
  ee.Algorithms.If(enoughModerate, lossLabel_moderate,
                                   lossLabel_loose))
);

print('Loss label threshold used (strict=z<-2.5, moderate=z<-1.5, loose=z<-1.0):',
  ee.Algorithms.If(enoughStrict, 'strict',
  ee.Algorithms.If(enoughModerate, 'moderate', 'loose')));

var stableLabel = zscore.gt(-0.5).and(forestMask);

// Sample training points
var lossSamples = targetFeatures
  .updateMask(lossLabel)
  .addBands(ee.Image.constant(1).rename('label'))
  .stratifiedSample({
    numPoints:  RF_SAMPLES,
    classBand:  'label',
    region:     aoi,
    scale:      20,
    seed:       42,
    geometries: false
  });

var stableSamples = targetFeatures
  .updateMask(stableLabel)
  .addBands(ee.Image.constant(0).rename('label'))
  .stratifiedSample({
    numPoints:  RF_SAMPLES,
    classBand:  'label',
    region:     aoi,
    scale:      20,
    seed:       42,
    geometries: false
  });

var allSamples = lossSamples.merge(stableSamples)
  .randomColumn('rand', 99);
var trainSamples = allSamples.filter(ee.Filter.lt('rand', 0.8));
var valSamples   = allSamples.filter(ee.Filter.gte('rand', 0.8));

print('── Layer 2 RF training ─────────────────────');
print('Loss samples:', lossSamples.size());
print('Stable samples:', stableSamples.size());
print('Train/val split:', trainSamples.size(), '/', valSamples.size());

// Train RF
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees:     RF_TREES,
  minLeafPopulation: 5,
  variablesPerSplit: 5,  // ~sqrt(20 features)
  seed:              42
}).train({
  features:        trainSamples,
  classProperty:   'label',
  inputProperties: featureBands
});

var rfProb = ee.Classifier.smileRandomForest({
  numberOfTrees:     RF_TREES,
  minLeafPopulation: 5,
  variablesPerSplit: 5,
  seed:              42
}).setOutputMode('PROBABILITY')
  .train({
    features:        trainSamples,
    classProperty:   'label',
    inputProperties: featureBands
  });

// Internal validation
var confMatrix = valSamples.classify(rf)
  .errorMatrix('label', 'classification');
print('RF confusion matrix:', confMatrix);
print('RF overall accuracy:', confMatrix.accuracy());
print('RF kappa:', confMatrix.kappa());
print('RF feature importances:', rf.explain().get('importance'));

// Classify target image
var rfProbImg = targetFeatures.classify(rfProb).updateMask(forestMask);
var rfAlert   = rfProbImg.gt(0.5)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .selfMask();

Map.addLayer(rfProbImg,
  {min:0, max:1, palette:['#1a9641','#ffffbf','#d7191c']},
  'RF: loss probability', false);
Map.addLayer(rfAlert,
  {palette:['#990000']}, 'RF: loss alert (p > 0.5)');

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 7: Combined alert — agreement between Layer 1 and Layer 2
// Only flag pixels where BOTH z-score severe AND RF agree
// This minimises false positives at forest edges
// ═══════════════════════════════════════════════════════════════════════════════
var combinedAlert = alertSevere.unmask(0)
  .and(rfAlert.unmask(0))
  .selfMask()
  .updateMask(forestMask);

Map.addLayer(combinedAlert,
  {palette:['#ff0000']}, '★ COMBINED ALERT (z-score + RF agreement)');

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 8: Validation against Hansen 2024 (if target year is 2024)
// Skip this block if analysing 2025 — no Hansen ground truth yet
// ═══════════════════════════════════════════════════════════════════════════════
var targetYear = targetDate.get('year');
var hansenLossYear = lossYearU.eq(ee.Number(targetYear).subtract(2000))
  .and(wasForest).and(datamask.eq(1));

function validateLayer(detected, label) {
  var agr = hansenLossYear.unmask(0).updateMask(forestMask)
    .multiply(2).add(detected.unmask(0)).rename('agreement');
  var stats = agr.reduceRegion({
    reducer:   ee.Reducer.frequencyHistogram(),
    geometry:  aoi,
    scale:     30,
    maxPixels: 1e10
  });
  var hist = ee.Dictionary(ee.Dictionary(stats).get('agreement'));
  var tp   = ee.Number(hist.get('3', 0));
  var fp   = ee.Number(hist.get('1', 0));
  var fn   = ee.Number(hist.get('2', 0));
  var precision = tp.divide(tp.add(fp));
  var recall    = tp.divide(tp.add(fn));
  var f1        = precision.multiply(recall).multiply(2).divide(precision.add(recall));
  print('── ' + label + ' vs Hansen ' + TARGET_DATE.slice(0,4) + ' ──');
  print('  precision:', precision);
  print('  recall:',    recall);
  print('  F1:',        f1);
}

validateLayer(alertSevere.unmask(0),   'Z-score severe [z<-2.5]');
validateLayer(rfAlert.unmask(0),       'RF alert [p>0.5]');
validateLayer(combinedAlert.unmask(0), 'Combined alert');

// ═══════════════════════════════════════════════════════════════════════════════
// STEP 9: Area summaries
// ═══════════════════════════════════════════════════════════════════════════════
function areaHa(img, label) {
  var a = img.unmask(0).eq(1).multiply(ee.Image.pixelArea()).rename('area')
    .reduceRegion({
      reducer:   ee.Reducer.sum(),
      geometry:  aoi,
      scale:     20,
      maxPixels: 1e10
    });
  print(label + ' (ha):',
    ee.Number(a.get('area')).divide(10000).format('%.1f'));
}

print('── Area summaries ──────────────────────────');
areaHa(alertModerate,  'Z-score moderate [z<-1.5]');
areaHa(alertSevere,    'Z-score severe   [z<-2.5]');
areaHa(rfAlert,        'RF alert [p>0.5]');
areaHa(combinedAlert,  'Combined alert (both agree)');
areaHa(hansenLossYear.updateMask(forestMask), 'Hansen reference ' + TARGET_DATE.slice(0,4));

print('Done ✓');
print('Target date analysed:', TARGET_DATE);
print('To analyse a different date, change TARGET_DATE at the top of the script.');