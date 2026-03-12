// ═══════════════════════════════════════════════════════════════════════════════
// ŠUMAVA DEFORESTATION DETECTOR — v4.1  (all GEE API bugs fixed)
// ═══════════════════════════════════════════════════════════════════════════════
// to be run in Google earth engine console
// ═══════════════════════════════════════════════════════════════════════════════

// ── AOI: exact Šumava NP boundary (inline, 125-pt simplification) ─────────────
var aoi = ee.Geometry.Polygon([[
  [13.23115, 49.12411],[13.23606, 49.11372],[13.27640, 49.12048],
  [13.28918, 49.11864],[13.34418, 49.08889],[13.34641, 49.08206],
  [13.37049, 49.06746],[13.37622, 49.05826],[13.39629, 49.05154],
  [13.39787, 49.04579],[13.39167, 49.04215],[13.39966, 49.03707],
  [13.40583, 49.02385],[13.40029, 49.01567],[13.40941, 49.00322],
  [13.40227, 48.99444],[13.40272, 48.98722],[13.42439, 48.97741],
  [13.42618, 48.97249],[13.45926, 48.96267],[13.49577, 48.94149],
  [13.50827, 48.94214],[13.50706, 48.96912],[13.52927, 48.97405],
  [13.54806, 48.96688],[13.58390, 48.96921],[13.59300, 48.96089],
  [13.59034, 48.95297],[13.61025, 48.93862],[13.60792, 48.94351],
  [13.62860, 48.94924],[13.63112, 48.94700],[13.62249, 48.93881],
  [13.63802, 48.92569],[13.63802, 48.91923],[13.65548, 48.89358],
  [13.66965, 48.89051],[13.67144, 48.88015],[13.71687, 48.87814],
  [13.73054, 48.88712],[13.73794, 48.88602],[13.73730, 48.87934],
  [13.75061, 48.86683],[13.74945, 48.85965],[13.76444, 48.83448],
  [13.79295, 48.83012],[13.78818, 48.82485],[13.81525, 48.79709],
  [13.80355, 48.78086],[13.81320, 48.77402],[13.87613, 48.76661],
  [13.91022, 48.74750],[13.93917, 48.72324],[13.95371, 48.72074],
  [13.95589, 48.71442],[13.98213, 48.72058],[13.97977, 48.73080],
  [13.97278, 48.73298],[13.96853, 48.74391],[13.92272, 48.76981],
  [13.95167, 48.79387],[13.96227, 48.79577],[13.96643, 48.82083],
  [13.94190, 48.84613],[13.91158, 48.86257],[13.89953, 48.87717],
  [13.89564, 48.89016],[13.88363, 48.89959],[13.87344, 48.90038],
  [13.86671, 48.89275],[13.85745, 48.89470],[13.84173, 48.90626],
  [13.82501, 48.90906],[13.82398, 48.91639],[13.81244, 48.92207],
  [13.79032, 48.91356],[13.77428, 48.91371],[13.76833, 48.90807],
  [13.72096, 48.90702],[13.72223, 48.91334],[13.70721, 48.93111],
  [13.70936, 48.94627],[13.71518, 48.94707],[13.71385, 48.95735],
  [13.70099, 48.96714],[13.68369, 48.96742],[13.67245, 48.97534],
  [13.66393, 48.99080],[13.67209, 49.00270],[13.66691, 49.02621],
  [13.62685, 49.03266],[13.62667, 49.04386],[13.61972, 49.05657],
  [13.60452, 49.05952],[13.60669, 49.07734],[13.61410, 49.08209],
  [13.60473, 49.08638],[13.60123, 49.09920],[13.58730, 49.11700],
  [13.57728, 49.12040],[13.57607, 49.10765],[13.53489, 49.13572],
  [13.52361, 49.13990],[13.50946, 49.13807],[13.51223, 49.14358],
  [13.48910, 49.14334],[13.46729, 49.13631],[13.44514, 49.13775],
  [13.43621, 49.14612],[13.44433, 49.15502],[13.43555, 49.15616],
  [13.40841, 49.17376],[13.36871, 49.18225],[13.34450, 49.17416],
  [13.30995, 49.19148],[13.30380, 49.18948],[13.30653, 49.18582],
  [13.29512, 49.17867],[13.29063, 49.16805],[13.26024, 49.15472],
  [13.24888, 49.14181],[13.24774, 49.13761],[13.25200, 49.13773],
  [13.24738, 49.12903],[13.23115, 49.12411]
]], null, false);

// G3: use setCenter instead of centerObject to avoid ErrorMargin bug
Map.setCenter(13.6, 48.95, 10);
Map.addLayer(aoi, {color: 'cyan'}, 'Šumava NP boundary');

// ═══════════════════════════════════════════════
// STEP 1: Hansen labels
// ═══════════════════════════════════════════════
var hansen    = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var lossYear  = ee.Image(hansen.select('lossyear'));
var treecover = ee.Image(hansen.select('treecover2000'));
var datamask  = ee.Image(hansen.select('datamask'));
var lossYearU = lossYear.unmask(0);

var wcTree  = ee.Image('ESA/WorldCover/v200/2021').select('Map').eq(10);  // moved up

var wasForest  = treecover.gt(30);
var hadAnyLoss = lossYearU.gte(1);

var stableMask    = hadAnyLoss.not().and(wasForest).and(datamask.eq(1)).and(wcTree);
var trainLossMask = lossYearU.gte(20).and(lossYearU.lte(23)).and(wasForest);
var hansen2024Loss = lossYearU.eq(24).and(wasForest).and(datamask.eq(1));

Map.addLayer(stableMask.selfMask(),
  {palette:['#2d6a2d']}, 'Hansen: Stable Forest (train)', false);
Map.addLayer(trainLossMask.selfMask(),
  {palette:['orange']}, 'Hansen: Loss 2020-2023 (train)', false);
Map.addLayer(hansen2024Loss.selfMask(),
  {palette:['#ff6600']}, 'Hansen: Confirmed Loss 2024 (val)', false);

// ═══════════════════════════════════════════════
// STEP 2: Forest mask
// ═══════════════════════════════════════════════
var wcTree  = ee.Image('ESA/WorldCover/v200/2021').select('Map').eq(10);
var preLoss = lossYearU.gte(1).and(lossYearU.lte(19));
var forestMask = wasForest.and(datamask.eq(1)).and(wcTree).and(preLoss.not());

Map.addLayer(forestMask.selfMask(),
  {palette:['#1a5c1a']}, 'Forest Mask', false);

// ═══════════════════════════════════════════════
// STEP 3: Monthly composite builder (May–Oct)
// G5: returns ee.Image.cat() of per-month bands — NOT ee.Image(JS array)
// ═══════════════════════════════════════════════
var MONTHS   = [5, 6, 7, 8, 9, 10];
var S2_BANDS = ['B4','B8','B11','B12'];

function safeMonthComposite(year, m) {
  var start = ee.Date.fromYMD(year, m, 1);
  var end   = start.advance(1, 'month');
  var s     = '_M' + m;

  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

  var fallback = ee.Image.constant([0,0,0,0]).rename(S2_BANDS).selfMask();

  var median = ee.Image(
    ee.Algorithms.If(col.size().gt(0),
      col.median().select(S2_BANDS),
      fallback)
  );

  // G5: build per-month block as a single multi-band image via addBands
  return ee.Image.cat([
    median.select('B4').rename('B4'  + s),
    median.select('B8').rename('B8'  + s),
    median.select('B11').rename('B11'+ s),
    median.normalizedDifference(['B8','B4']).rename('NDVI' + s),
    median.normalizedDifference(['B8','B12']).rename('NBR' + s),
    median.normalizedDifference(['B8','B11']).rename('NDWI'+ s)
  ]);
}

function monthlyComposite(year) {
  // G5: ee.Image.cat() correctly merges an array of ee.Image into one multi-band image
  return ee.Image.cat(
    MONTHS.map(function(m){ return safeMonthComposite(year, m); })
  );
}

// ═══════════════════════════════════════════════
// STEP 4: Delta features
// ═══════════════════════════════════════════════
function addDeltas(img) {
  return img.addBands([
    img.select('NDVI_M8').subtract(img.select('NDVI_M5')).rename('dNDVI_M8_M5'),
    img.select('NBR_M8').subtract(img.select('NBR_M5')).rename('dNBR_M8_M5'),
    img.select('NBR_M9').subtract(img.select('NBR_M6')).rename('dNBR_M9_M6'),
    img.select('NDVI_M9').subtract(img.select('NDVI_M7')).rename('dNDVI_M9_M7'),
    img.select('NDVI_M8').subtract(img.select('NDVI_M6')).rename('dNDVI_M8_M6'),
    img.select('NBR_M10').subtract(img.select('NBR_M7')).rename('dNBR_M10_M7')
  ]);
}

// ═══════════════════════════════════════════════
// STEP 5: Texture features (GLCM on B8 July)
// ═══════════════════════════════════════════════
function addTexture(img) {
  var b8july = img.select('B8_M7').unmask(0)
    .unitScale(0, 5000).multiply(255).toByte();
  var glcm = b8july.glcmTexture({size: 3});
  return img.addBands([
    glcm.select('B8_M7_contrast').rename('texture_contrast'),
    glcm.select('B8_M7_ent').rename('texture_entropy'),
    glcm.select('B8_M7_corr').rename('texture_corr'),
    glcm.select('B8_M7_diss').rename('texture_diss')
  ]);
}

// ═══════════════════════════════════════════════
// STEP 6: DEM features
// G1: ee.Terrain.slope/aspect require ee.Image() — compute BEFORE clip
// ═══════════════════════════════════════════════
// G1 REAL FIX: ee.Terrain.slope/aspect unreliable in Code Editor
// Use ee.Algorithms.Terrain() — the stable API that always works
var demRaw     = ee.Image('USGS/SRTMGL1_003');
var terrain    = ee.Algorithms.Terrain(demRaw);
var dem        = terrain.select('elevation').clip(aoi);
var slope      = terrain.select('slope').clip(aoi);
var aspect     = terrain.select('aspect').clip(aoi);
var demFeatures = ee.Image.cat([dem, slope, aspect]);

// ═══════════════════════════════════════════════
// STEP 7: Build training stacks
// ═══════════════════════════════════════════════
var TRAIN_YEARS = [2020, 2021, 2022, 2023];

var stableStack = ee.ImageCollection(
  TRAIN_YEARS.map(function(y){ return monthlyComposite(y); })
).mean();
stableStack = addDeltas(addTexture(stableStack)).addBands(demFeatures);

var lossStacks = TRAIN_YEARS.map(function(y) {
  // y is a client-side JS number so y-2000 is safe plain arithmetic
  var yearMask = lossYearU.eq(y - 2000).and(wasForest);
  var stack    = addDeltas(addTexture(monthlyComposite(y))).addBands(demFeatures);
  return stack.updateMask(yearMask);
});

var bandNames = stableStack.bandNames();
print('Feature band count:', bandNames.size());
print('Feature band names:', bandNames);

// ═══════════════════════════════════════════════
// STEP 8: Sample training points
// ═══════════════════════════════════════════════
var stableSamples = stableStack
  .updateMask(stableMask)
  .addBands(ee.Image.constant(0).rename('label'))
  .stratifiedSample({
    numPoints:  800,
    classBand:  'label',
    region:     aoi,
    scale:      30,
    seed:       42,
    geometries: false
  });

var lossSamplesList = TRAIN_YEARS.map(function(y, i) {
  return lossStacks[i]
    .addBands(ee.Image.constant(1).rename('label'))
    .stratifiedSample({
      numPoints:  200,
      classBand:  'label',
      region:     aoi,
      scale:      30,
      seed:       42,
      geometries: false
    });
});

var lossSamples = ee.FeatureCollection(lossSamplesList[0])
  .merge(lossSamplesList[1])
  .merge(lossSamplesList[2])
  .merge(lossSamplesList[3]);

print('Stable samples:', stableSamples.size());
print('Loss samples (2020-2023):', lossSamples.size());

var allSamples = stableSamples.merge(lossSamples).randomColumn('rand', 123);
var trainSet   = allSamples.filter(ee.Filter.lt('rand', 0.8));
var valSet     = allSamples.filter(ee.Filter.gte('rand', 0.8));
print('Train:', trainSet.size(), '| Val:', valSet.size());

// ═══════════════════════════════════════════════
// STEP 9: Train Random Forest
// ═══════════════════════════════════════════════
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees:     100,  // reduced from 200 to stay within GEE compute limits
  minLeafPopulation: 5,
  variablesPerSplit: 7,    // ~sqrt(49 bands) — speeds up each tree
  seed:              42
}).train({
  features:        trainSet,
  classProperty:   'label',
  inputProperties: bandNames
});

var probClassifier = classifier.setOutputMode('PROBABILITY');
print('RF trained ✓');
print('Feature importances:', classifier.explain());

// ═══════════════════════════════════════════════
// STEP 10: Internal validation (20% holdout)
// ═══════════════════════════════════════════════
var confMatrix = valSet.classify(classifier)
  .errorMatrix('label', 'classification');
print('— Internal validation (20% holdout) —');
print('Confusion matrix:', confMatrix);
print('Overall accuracy:', confMatrix.accuracy());
print('Kappa:', confMatrix.kappa());
print('Producer accuracy (recall):', confMatrix.producersAccuracy());
print('Consumer accuracy (precision):', confMatrix.consumersAccuracy());

// ═══════════════════════════════════════════════
// STEP 11: Predict 2024 — temporal holdout
// ═══════════════════════════════════════════════
var features2024     = addDeltas(addTexture(monthlyComposite(2024))).addBands(demFeatures);
var rfProb2024       = features2024.classify(probClassifier).updateMask(forestMask);
// Sieve: require 3×3 kernel majority to suppress isolated salt-and-pepper FPs
// A predicted-loss pixel is only kept if its neighbourhood is also mostly loss
var rfBinary2024_t50 = rfProb2024.gt(0.50)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .rename('pred_t50');
var rfBinary2024_t40 = rfProb2024.gt(0.40)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .rename('pred_t40');
var rfBinary2024_t35 = rfProb2024.gt(0.35)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .rename('pred_t35');

Map.addLayer(rfProb2024,
  {min:0, max:1, palette:['#1a9641','#ffffbf','#d7191c']},
  'RF: Loss Probability 2024');
Map.addLayer(rfBinary2024_t50.selfMask(), {palette:['red']},
  'RF: Predicted Loss 2024 (t=0.50)', false);
Map.addLayer(rfBinary2024_t40.selfMask(), {palette:['#cc3300']},
  'RF: Predicted Loss 2024 (t=0.40)', false);
Map.addLayer(rfBinary2024_t35.selfMask(), {palette:['#990000']},
  'RF: Predicted Loss 2024 (t=0.35)', false);

// Agreement: (hansen×2)+rf → 0=TN 1=FP 2=FN 3=TP
function agreementMap(rfBinary, label) {
  var hansenImg = hansen2024Loss.unmask(0).updateMask(forestMask);
  var agr = hansenImg.multiply(2).add(rfBinary.unmask(0)).rename('agreement');

  Map.addLayer(agr.updateMask(forestMask).selfMask(),
    {min:0, max:3, palette:['#cccccc','#f4a321','#3182bd','#cc0000']},
    label + '  grey=TN  orange=FP  blue=FN  red=TP');

  var stats = agr.updateMask(forestMask).reduceRegion({
    reducer:   ee.Reducer.frequencyHistogram(),
    geometry:  aoi,
    scale:     30,
    maxPixels: 1e10
  });
  print(label + ' histogram:', stats);

  var hist = ee.Dictionary(ee.Dictionary(stats).get('agreement'));
  var tp   = ee.Number(hist.get('3', 0));
  var fp   = ee.Number(hist.get('1', 0));
  var fn   = ee.Number(hist.get('2', 0));
  var tn   = ee.Number(hist.get('0', 0));
  var precision = tp.divide(tp.add(fp));
  var recall    = tp.divide(tp.add(fn));
  var f1        = precision.multiply(recall).multiply(2).divide(precision.add(recall));
  var iou       = tp.divide(tp.add(fp).add(fn));
  print(label + ' precision:', precision);
  print(label + ' recall:',    recall);
  print(label + ' F1:',        f1);
  print(label + ' IoU:',       iou);
  print(label + ' TN/FP/FN/TP:', ee.List([tn, fp, fn, tp]));
}

agreementMap(rfBinary2024_t50, 'RF vs Hansen 2024 [t=0.50]');
agreementMap(rfBinary2024_t40, 'RF vs Hansen 2024 [t=0.40]');
agreementMap(rfBinary2024_t35, 'RF vs Hansen 2024 [t=0.35]');

// ═══════════════════════════════════════════════
// STEP 12: Predict 2025
// ═══════════════════════════════════════════════
var monthMasks2025 = ee.ImageCollection(
  MONTHS.map(function(m) {
    // .size() returns a scalar; use constant image so result is always 1-band
    var n = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate(
        ee.Date.fromYMD(2025, m, 1),
        ee.Date.fromYMD(2025, m, 1).advance(1, 'month'))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .size();
    // Returns a guaranteed single-band constant image: 1 if scenes exist, else 0
    return ee.Image(ee.Algorithms.If(n.gt(0),
      ee.Image.constant(1), ee.Image.constant(0)
    )).rename('has_data').toFloat();
  })
).sum().rename('months_available');

var highConf2025 = monthMasks2025.gte(4).updateMask(forestMask);

print('2025 S2 scenes in AOI:',
  ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filterDate('2025-05-01','2025-10-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).size());

var features2025 = addDeltas(addTexture(monthlyComposite(2025))).addBands(demFeatures);
var rfProb2025   = features2025.classify(probClassifier).updateMask(forestMask);
var rfBinary2025 = rfProb2025.gt(0.40)
  .focal_min({radius:1, kernelType:'square', units:'pixels'})
  .focal_max({radius:1, kernelType:'square', units:'pixels'})
  .updateMask(forestMask);

Map.addLayer(rfProb2025,
  {min:0, max:1, palette:['#1a9641','#ffffbf','#d7191c']},
  'RF: Loss Probability 2025');
Map.addLayer(rfBinary2025.selfMask(), {palette:['red']},
  'RF: Predicted Loss 2025 (t=0.40)');
Map.addLayer(rfBinary2025.updateMask(highConf2025).selfMask(), {palette:['#880000']},
  'RF: Predicted Loss 2025 — HIGH CONFIDENCE ✓');
Map.addLayer(monthMasks2025.updateMask(forestMask),
  {min:0, max:6, palette:['#d73027','#fdae61','#a6d96a','#1a9641']},
  'DEBUG: 2025 month coverage (0–6)', false);

var riskDelta = rfProb2025.subtract(rfProb2024).updateMask(forestMask);
Map.addLayer(riskDelta,
  {min:-0.4, max:0.4, palette:['#2166ac','#f7f7f7','#d6604d']},
  'Risk Delta: 2025 − 2024 (red = emerging risk)');

// ═══════════════════════════════════════════════
// STEP 13: Area summaries
// G4: use named band key instead of .values().get(0) to avoid null crash
// ═══════════════════════════════════════════════
function lossAreaHa(binaryImg, label) {
  // Rename to known key so .get() is safe regardless of band order
  var areaImg = binaryImg.eq(1).multiply(ee.Image.pixelArea()).rename('area');
  var result  = areaImg.reduceRegion({
    reducer:   ee.Reducer.sum(),
    geometry:  aoi,
    scale:     30,
    maxPixels: 1e10
  });
  // G4: .get('area') is safe — band was explicitly renamed above
  print(label + ' loss area (ha):',
    ee.Number(result.get('area')).divide(10000).format('%.1f'));
}

print('── Area summaries ──────────────────────────');
lossAreaHa(rfBinary2024_t50,                      'RF 2024 [t=0.50]');
lossAreaHa(rfBinary2024_t40,                      'RF 2024 [t=0.40]');
lossAreaHa(rfBinary2024_t35,                      'RF 2024 [t=0.35]');
lossAreaHa(rfBinary2025,                          'RF 2025 [t=0.40]');
lossAreaHa(rfBinary2025.updateMask(highConf2025), 'RF 2025 [t=0.40, high-conf]');
lossAreaHa(hansen2024Loss.unmask(0),              'Hansen 2024 reference');

print('Done ✓');