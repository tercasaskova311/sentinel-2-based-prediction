import pandas as pd
import geopandas as gpd

df = gpd.read_file('/Users/terezasaskova/Downloads/labeled_alerts_24.geojson')

print("Columns:", df.columns.tolist())
print("Total features:", len(df))
print("\n--- Label distribution ---")
print(df['label'].value_counts())

# Counts
tp = df[df['label'] == 'tp']
fp = df[df['label'] == 'fp']

n_tp = len(tp)
n_fp = len(fp)
n_total = len(df)

# Accuracy metrics
precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0

print(f"\n--- Accuracy ---")
print(f"TP: {n_tp}")
print(f"FP: {n_fp}")
print(f"Precision (user's accuracy): {precision:.2%}")

# Area stats
print(f"\n--- Area (ha) ---")
print(f"Total alerted area:  {df['area_ha'].sum():.1f} ha")
print(f"TP area:             {tp['area_ha'].sum():.1f} ha")
print(f"FP area:             {fp['area_ha'].sum():.1f} ha")
print(f"Mean TP patch size:  {tp['area_ha'].mean():.2f} ha")
print(f"Mean FP patch size:  {fp['area_ha'].mean():.2f} ha")

# Alert scene count if available
if 'alerts' in df.columns:
    print(f"\n--- Scene count ---")
    print(f"Mean alert scenes (TP): {tp['alerts'].mean():.2f}")
    print(f"Mean alert scenes (FP): {fp['alerts'].mean():.2f}")