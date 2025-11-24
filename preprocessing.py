#normalize the rectangels from sentinel2 images in order to train them? 
import geopandas as gpd
import os
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

root = "data/historical"

with rasterio.open("data/gfc_lossyear.tif") as src:
    gfc_loss_patch = src.read(1)

records = []

for year in sorted(os.listdir(root)):
    folder = os.path.join(root, year)
    if not os.path.isdir(folder):
        continue
    
    for tif in sorted(os.listdir(folder)):
        if not tif.endswith(".tif"):
            continue
        
        date_str = tif.replace(".tif","")
        date = datetime.strptime(date_str, "%Y-%m-%d")
        path = os.path.join(folder, tif)

        with rasterio.open(path) as src:
            arr = src.read().astype("float32")  # shape: [3 bands, H, W]
            ndvi = arr[0]
            nbr  = arr[1]
            ndmi = arr[2]

            records.append({
                "date": date,
                "year": int(year),
                "NDVI_mean": np.nanmean(ndvi),
                "NBR_mean": np.nanmean(nbr),
                "NDMI_mean": np.nanmean(ndmi),
            })

df = pd.DataFrame(records).sort_values("date")
df.head()

plt.figure(figsize=(14,6))
plt.plot(df["date"], df["NDVI_mean"], label="NDVI")
plt.plot(df["date"], df["NBR_mean"],  label="NBR")
plt.plot(df["date"], df["NDMI_mean"], label="NDMI")
plt.legend()
plt.title("Vegetation Indices Time Series – Šumava")
plt.xlabel("Date")
plt.ylabel("Index value")
plt.grid()
plt.show()

def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(2).astype("float32")  # band 2 = NBR in your exports

# choose pair
before = "data/historical/2021/2021-06-12.tif"
after  = "data/historical/2021/2021-07-27.tif"

nbr_before = load_tif(before)
nbr_after  = load_tif(after)

delta_nbr = nbr_before - nbr_after  # positive = disturbance

plt.figure(figsize=(8,6))
plt.imshow(delta_nbr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(label="ΔNBR (loss)")
plt.title("Change Detection (ΔNBR) – Šumava")
plt.show()


def label_disturbances(df, gfc_loss_patch, nbr_drop_threshold=0.15, gfc_fraction_threshold=0.0):
    """
    Label forest disturbances in a Sentinel-2 time-series dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'date' (datetime) and 'NBR_mean' (mean NBR for patch).
    gfc_loss_patch : np.array or pd.Series
        Array of Hansen GFC loss years for the patch (same spatial extent as patch).
        Values: 0=no loss, 1..21 = loss year (2001-2021).
    nbr_drop_threshold : float
        Threshold for sudden NBR drop to flag disturbance.
    gfc_fraction_threshold : float
        Minimum fraction of pixels lost in patch to consider label=1.

    Returns:
    --------
    df_labeled : pd.DataFrame
        Original dataframe with two new columns:
        - 'disturbed': True if sudden NBR drop
        - 'gfc_label': 0=healthy, 1=loss according to GFC
    """

    df = df.copy()
    df["disturbed"] = False
    df["gfc_label"] = 0

    #Spectral change detection
    for i in range(1, len(df)):
        prev_nbr = df.iloc[i-1]["NBR_mean"]
        curr_nbr = df.iloc[i]["NBR_mean"]
        if np.isnan(prev_nbr) or np.isnan(curr_nbr):
            continue
        if prev_nbr - curr_nbr > nbr_drop_threshold:
            df.loc[df.index[i], "disturbed"] = True

    # Hansen GFC labeling
    for i, row in df.iterrows():
        year = row["date"].year
        gfc_code = year - 2000  # Map calendar year to GFC lossyear (1=2001)
        if 1 <= gfc_code <= 21:
            # Fraction of pixels in patch lost this year
            fraction_lost = np.nanmean(gfc_loss_patch == gfc_code)
            if fraction_lost > gfc_fraction_threshold:
                df.loc[i, "gfc_label"] = 1

    return df

