"""Baixa a série de um pixel do Sentinel-2 L2A (Planetary Computer) e grava em CSV."""
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform

LAT, LON = -13.375151012548173, -54.47550071228342   # talhão de soja no MT (MapBiomas classe 39)
OUT = sys.argv[1]

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
items = list(catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [LON, LAT]},
    datetime="2019-01-01/2025-12-31",
).items())
print("cenas encontradas:", len(items))
print(pd.Series([i.properties["s2:mgrs_tile"] for i in items]).value_counts().to_string())


def read_pixel(item):
    out = {"data": item.datetime.date(), "tile": item.properties["s2:mgrs_tile"],
           "baseline": item.properties.get("s2:processing_baseline"),
           "nuvem_cena": item.properties.get("eo:cloud_cover")}
    for band in ["B04", "B08", "SCL"]:
        with rasterio.open(item.assets[band].href) as src:
            xs, ys = transform("EPSG:4326", src.crs, [LON], [LAT])
            r, c = src.index(xs[0], ys[0])
            out[band] = int(src.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
    return out


with ThreadPoolExecutor(16) as pool:
    rows = list(pool.map(read_pixel, items))
df = pd.DataFrame(rows).sort_values("data")
df.to_csv(OUT, index=False)
print(df.head().to_string(), "\n", len(df))
