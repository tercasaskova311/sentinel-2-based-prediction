# STRUCTURE

rn there is:

- data/ boundaries
    - sumava_aio_clean_proj.geojson: basically a aio from QGIS 
    - gfc_lossyear.tif: GFC projected on aio Šumava...

- output: savig the preprocessed data

- scr:
    - download_aio_osm: download from OSM + projection 
    - download_gee: take aio and download given indices for our aio from gee to google drive
    - preprocessing: take imaginery with filtered indices and filte only if +30% pixels usable, create composites, compute some basics stats
