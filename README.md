# sentinel-2-based-prediction
Sentinel-2 Based Deforestation/Bark Bettle regrowth + Suspicious Logging Detection in Šumava


Sentinel-2 Based Deforestation/Bark Bettle regrowth + Suspicious Logging Detection in Šumava


STRUCTURE
detect forest disturbance and classify some of it as “potentially suspicious” using:
✔ Sentinel-2 10 m yearly composites
✔ Hansen Global Forest Change (for training)
✔ Šumava NP zone maps (Zone 1 = no logging allowed)
✔ Forest road maps (OpenStreetMap)
✔ Patch-based ML model (ChangeFormer or UNet)

and produce:
● A forest-disturbance probability map
● A “suspicious disturbance” map with categories:
disturbance in Zone 1
disturbance > 30m from forest road
patch too small (<0.25 ha)
disturbance hugging zone boundaries (<20m)
disturbance not in Hansen (extra detections)

