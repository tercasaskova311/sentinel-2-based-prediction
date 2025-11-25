# ------------------------------------------------------------
# Sentinel-2 Preprocessing – Robust Version
# ------------------------------------------------------------
import os
import re
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ------------------------------
# 1) Settings
# ------------------------------
root = "data/historical"
gfc_loss_path = "data/gfc_lossyear.tif"

# ------------------------------
# 2) Load Global Forest Change (GFC) loss year layer
# ------------------------------
with rasterio.open(gfc_loss_path) as src:
    gfc_loss_patch = src.read(1)

# ------------------------------
# 3) Helper function: extract date robustly from filename
# ------------------------------
def extract_date_from_filename(fname):
    """
    Extracts year, month, day from a filename.
    Works for messy names like:
        sumava_2020-10-25.tif
        sumava_2021_04_17(1).tif
        regionA_sumava_2022_07_02 copy.tif
    """
    # Find all 4-digit years in the filename
    years = re.findall(r"(\d{4})", fname)
    for y in years:
        # Look for month/day after this year
        pattern = rf"{y}[-_]?(\d{{2}})[-_]?(\d{{2}})"
        match = re.search(pattern, fname)
        if match:
            m, d = map(int, match.groups())
            return datetime(int(y), m, d)
    return None

# ------------------------------
# 4) Preprocess TIFFs
# ------------------------------
records = []

for year_folder in sorted(os.listdir(root)):
    folder = os.path.join(root, year_folder)
    if not os.path.isdir(folder):
        continue

    for tif in sorted(os.listdir(folder)):
        path = os.path.join(folder, tif)

        # Skip non-TIFFs and hidden/macOS files
        if not tif.lower().endswith(".tif") or tif.startswith(".") or tif.startswith("._"):
            continue

        # Skip duplicate/messy files
        if re.search(r"\(\d+\)", tif) or "copy" in tif.lower() or " " in tif[:-4]:
            print("Skipping duplicate/messy file:", tif)
            continue

        # Extract date
        date = extract_date_from_filename(tif)
        if date is None:
            print("Skipping bad filename (no date found):", tif)
            continue

        # Read raster and normalize
        with rasterio.open(path) as src:
            arr = src.read().astype("float32")
            arr /= 10000.0  # Sentinel-2 normalization 0-1

            ndvi = arr[0]
            nbr  = arr[1]
            ndmi = arr[2]

        records.append({
            "date": date,
            "year": date.year,
            "NDVI_mean": float(np.nanmean(ndvi)),
            "NBR_mean":  float(np.nanmean(nbr)),
            "NDMI_mean": float(np.nanmean(ndmi)),
        })

# ------------------------------
# 5) Build DataFrame
# ------------------------------
if not records:
    raise RuntimeError("No TIFF files were processed! Check filenames and paths.")

df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
print("Loaded records:", len(df))
print(df.head())

# ------------------------------
# 6) Plot Time Series
# ------------------------------
plt.figure(figsize=(14,6))
plt.plot(df["date"], df["NDVI_mean"], label="NDVI")
plt.plot(df["date"], df["NBR_mean"], label="NBR")
plt.plot(df["date"], df["NDMI_mean"], label="NDMI")
plt.legend()
plt.title("Vegetation Indices Time Series – Šumava")
plt.xlabel("Date")
plt.ylabel("Index value")
plt.grid()
plt.show()

# ------------------------------
# 7) Load individual NBR for change detection
# ------------------------------
def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(2).astype("float32")  # band 2 = NBR

# Example usage:
# before = "data/historical/2021/2021-06-12.tif"
# after  = "data/historical/2021/2021-07-27.tif"
# delta_nbr = load_tif(before) - load_tif(after)

# ------------------------------
# 8) Label disturbances
# ------------------------------
def label_disturbances(df, gfc_loss_patch, nbr_drop_threshold=0.15, gfc_fraction_threshold=0.0):
    df = df.copy()
    df["disturbed"] = False
    df["gfc_label"] = 0

    # Sudden NBR drops
    for i in range(1, len(df)):
        prev_nbr = df.iloc[i-1]["NBR_mean"]
        curr_nbr = df.iloc[i]["NBR_mean"]
        if np.isnan(prev_nbr) or np.isnan(curr_nbr):
            continue
        if prev_nbr - curr_nbr > nbr_drop_threshold:
            df.loc[df.index[i], "disturbed"] = True

    # GFC labels
    for i, row in df.iterrows():
        year = row["date"].year
        gfc_code = year - 2000
        if 1 <= gfc_code <= 21:
            fraction_lost = np.nanmean(gfc_loss_patch == gfc_code)
            if fraction_lost > gfc_fraction_threshold:
                df.loc[i, "gfc_label"] = 1

    return df
