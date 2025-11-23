## Automated Deforestation Monitoring in Šumava Using - Sentinel2: Image Processing and Change Detection


# STRUCTURE
- Data Acquisition & Preprocessing
- Feature Extraction (Indices)
- Image Change Detection
- Deforestation Classification + Visualization

# Automated Data Download
- Sentinel Hub API (recommended)
- Bands: B02 (Blue), B03, B04 (Red), B08 (NIR), B8A, B11, B12
SCL (Scene Classification Layer) for cloud masking

# Preprocessing Pipeline
- Cloud Masking: Use SCL band (classes: clouds, shadows, snow, vegetation).
- Resample to 10 m
- Normalize bands

# FEATURE EXTRACTION (Classic Image Processing)
- NDVI
- NBR (Normalized Burn Ratio) — best for clear-cut logging
- NDMI (Moisture Index)
- NDWI (Water Index) – useful to ignore wet areas

# Compute Texture Features 
- Using Grey-Level Co-occurrence Matrix (GLCM)

# CHANGE DETECTION
- Temporal Pairing

- Change Detection Methods: Method A — Simple Differencing, Method B — Otsu Thresholding on ΔNBR

# CLASSIFICATION 
- Create Training Data
- Extract Pixel-Level Features


# Train a Simple ML Model
- Random Forest or LightGBM

# Create Time-Series Graphs
- For the entire Šumava region: area lost per month, number of disturbance patches, severity index

# Evaluate the System
- precision (how many detected disturbances are correct)
- recall (how many actual disturbances were caught)
- IoU for segmentation mask
- ROC curve (for ML classifier)




