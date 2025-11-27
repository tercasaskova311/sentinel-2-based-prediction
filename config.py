
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

class DataPaths:
    """All data file paths"""
    # Raw data
    HISTORICAL_DIR = DATA_DIR / "historical"
    BOUNDARIES_DIR = DATA_DIR / "boundaries"
    
    # Input files
    AOI_GEOJSON = BOUNDARIES_DIR / "sumava_aoi_clean.geojson"
    GFC_LOSSYEAR = DATA_DIR / "gfc_lossyear.tif"
    
    # Processed data outputs
    PREPROCESSED_CSV = OUTPUT_DIR / "01_preprocessed_timeseries.csv"
    PREPROCESSED_PKL = OUTPUT_DIR / "01_preprocessed_timeseries.pkl"  # Faster loading
    
    # ML-ready data
    ML_DATASET = OUTPUT_DIR / "02_ml_dataset_labeled.csv"
    ML_DATASET_PKL = OUTPUT_DIR / "02_ml_dataset_labeled.pkl"
    
    TRAIN_DATA = OUTPUT_DIR / "03_train_data.csv"
    TEST_DATA = OUTPUT_DIR / "03_test_data.csv"
    FEATURE_COLS = OUTPUT_DIR / "03_feature_columns.txt"
    
    # Analysis outputs
    FIGURES_DIR = OUTPUT_DIR / "figures"
    FIGURES_DIR.mkdir(exist_ok=True)
    
    PREPROCESSING_PLOT = FIGURES_DIR / "01_preprocessing_timeseries.png"
    LABEL_ANALYSIS_PLOT = FIGURES_DIR / "02_label_analysis.png"
    FEATURE_IMPORTANCE_PLOT = FIGURES_DIR / "04_feature_importance.png"


# PREPROCESSING PARAMETERS
class PreprocessingConfig:
    """Settings for data preprocessing"""
    
    # Sentinel-2 normalization
    S2_SCALE_FACTOR = 10000.0
    
    # Band indices (in the stacked GeoTIFF)
    NDVI_BAND = 0
    NBR_BAND = 1
    NDMI_BAND = 2
    
    # Valid value ranges (after normalization to 0-1)
    VALID_RANGE = (-1.0, 1.0)
    
    # Cloud/artifact filtering
    OUTLIER_THRESHOLD_STD = 3.0  # Remove values > 3 std devs from mean
    MIN_VALID_PIXELS_FRACTION = 0.5  # At least 50% valid pixels required
    
    # Date extraction patterns
    DATE_PATTERNS = [
        r"(\d{4})[-_](\d{2})[-_](\d{2})",  # YYYY-MM-DD or YYYY_MM_DD
        r"(\d{4})(\d{2})(\d{2})",           # YYYYMMDD
    ]
    
    # Files to skip
    SKIP_PATTERNS = [
        r"\(\d+\)",      # Duplicate markers (1), (2)
        r"copy",         # File copies
        r"^\.",          # Hidden files
        r"^\._",         # macOS resource forks
    ]

# LABELING PARAMETERS
# ============================================================================
class LabelingConfig:
    """Settings for disturbance labeling"""
    
    # GFC (Global Forest Change) labeling
    GFC_MIN_LOSS_FRACTION = 0.005  # Min 0.5% of AOI lost to label as disturbance
    GFC_CONFIDENCE_SCALE = 10.0    # Scale fraction to confidence (0-1)
    
    # NBR anomaly detection
    NBR_DROP_THRESHOLD = 0.15      # Minimum NBR drop to consider
    NBR_ANOMALY_STD_THRESHOLD = 2.0  # Std devs from seasonal baseline
    NBR_ROLLING_WINDOW = 5         # Number of observations for baseline
    
    # Spatial anomaly detection
    SPATIAL_STD_THRESHOLD = 0.15   # High spatial variation indicates disturbance
    SPATIAL_MIN_AFFECTED = 0.1     # Min 10% of area must show low values
    
    # Label confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4

# FEATURE ENGINEERING PARAMETERS
# ============================================================================
class FeatureConfig:
    """Settings for feature engineering"""
    
    # Rolling window sizes for temporal features
    ROLLING_WINDOWS = [3, 5, 10]
    
    # Spatial percentiles to compute
    SPATIAL_PERCENTILES = [10, 25, 50, 75, 90]
    
    # Feature groups (for feature selection)
    BASE_FEATURES = ['NDVI_mean', 'NBR_mean', 'NDMI_mean']
    
    SPATIAL_FEATURES = [
        'NDVI_std', 'NBR_std', 'NDMI_std',
        'NDVI_p10', 'NDVI_p90', 
        'NBR_p10', 'NBR_p90',
        'NDMI_p10', 'NDMI_p90'
    ]
    
    TEMPORAL_FEATURES = [
        'day_of_year', 'month', 'season',
        'days_since_start', 'days_since_prev'
    ]

# ML TRAINING PARAMETERS
# ============================================================================
class MLConfig:
    """Settings for ML model training"""
    
    # Train/test split
    TEST_YEAR = 2024  # Use 2024 for testing, earlier years for training
    VALIDATION_FRACTION = 0.2  # 20% of training data for validation
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Class imbalance handling
    USE_CLASS_WEIGHTS = True
    USE_SMOTE = False  # Use SMOTE oversampling if True
    
    # Model hyperparameters (defaults)
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10,  # Adjust based on class imbalance
            'random_state': RANDOM_SEED
        }
    }

# VISUALIZATION PARAMETERS
# ============================================================================
class VisualizationConfig:
    """Settings for plots and figures"""
    
    FIGURE_DPI = 150
    FIGURE_FORMAT = 'png'
    
    # Color scheme
    COLORS = {
        'normal': '#2ecc71',      # Green
        'disturbed': '#e74c3c',   # Red
        'warning': '#f39c12',     # Orange
        'info': '#3498db'         # Blue
    }
    
    # Plot sizes
    SMALL_PLOT = (10, 6)
    MEDIUM_PLOT = (14, 8)
    LARGE_PLOT = (18, 10)

# LOGGING & REPORTING
# ============================================================================
class ReportingConfig:
    """Settings for logging and reports"""
    
    VERBOSE = True
    SAVE_INTERMEDIATE_RESULTS = True
    
    # Report files
    PREPROCESSING_REPORT = OUTPUT_DIR / "report_01_preprocessing.txt"
    LABELING_REPORT = OUTPUT_DIR / "report_02_labeling.txt"
    TRAINING_REPORT = OUTPUT_DIR / "report_03_training.txt"

# VALIDATION CHECKS
# ============================================================================
def validate_configuration():
    """Check if all required paths exist"""
    missing = []
    
    if not DataPaths.HISTORICAL_DIR.exists():
        missing.append(f"Historical data directory: {DataPaths.HISTORICAL_DIR}")
    
    if not DataPaths.AOI_GEOJSON.exists():
        missing.append(f"AOI GeoJSON: {DataPaths.AOI_GEOJSON}")
    
    if not DataPaths.GFC_LOSSYEAR.exists():
        missing.append(f"GFC loss year raster: {DataPaths.GFC_LOSSYEAR}")
    
    if missing:
        print("⚠️  Missing required files/directories:")
        for item in missing:
            print(f"   - {item}")
        return False
    
    return True


# HELPER FUNCTIONS
# ============================================================================
def print_config_summary():
    """Print current configuration"""
    print("="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\n📁 Data Directories:")
    print(f"   Historical: {DataPaths.HISTORICAL_DIR}")
    print(f"   Output:     {OUTPUT_DIR}")
    
    print(f"\n🏷️  Labeling Settings:")
    print(f"   GFC threshold:      {LabelingConfig.GFC_MIN_LOSS_FRACTION*100:.1f}%")
    print(f"   NBR drop threshold: {LabelingConfig.NBR_DROP_THRESHOLD:.2f}")
    
    print(f"\n🤖 ML Settings:")
    print(f"   Test year:    {MLConfig.TEST_YEAR}")
    print(f"   Random seed:  {MLConfig.RANDOM_SEED}")
    print(f"   Class weights: {MLConfig.USE_CLASS_WEIGHTS}")
    
    print("="*70)

if __name__ == "__main__":
    print_config_summary()
    
    if validate_configuration():
        print("\n Configuration validated successfully!")
    else:
        print("\n Configuration validation failed!")