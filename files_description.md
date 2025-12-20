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
        * here I have changed the logic with composites. -- check it out - I think it can partly solve our proble with cloud masking - becuase composite basically means I take all imagies I have for given 10 days window and I take median - I do this by pixels - this is super important for next steps - becuase with pixels approach - we will have spatial detection(meaning we perform give statistics on given pixel only, which allow us to perform the detection later on...)
