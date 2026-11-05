"""Baixa um cubo 32x32 (10 m) do Sentinel-2 L2A centrado no pixel da série e grava em .npz."""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform
from rasterio.windows import Window

LAT, LON = -13.375151012548173, -54.47550071228342
SIDE, HALF = 32, 16          # pixel da série em [16, 16]
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
print("cenas:", len(items))


def read_patch(item):
    out = {}
    with rasterio.open(item.assets["B04"].href) as src:          # grade de 10 m de referência
        xs, ys = transform("EPSG:4326", src.crs, [LON], [LAT])
        r, c = src.index(xs[0], ys[0])
        win10 = Window(c - HALF, r - HALF, SIDE, SIDE)
        out["B04"] = src.read(1, window=win10)
        bounds = rasterio.windows.bounds(win10, src.transform)
    with rasterio.open(item.assets["B08"].href) as src:
        out["B08"] = src.read(1, window=win10)
    with rasterio.open(item.assets["SCL"].href) as src:           # 20 m, reamostrada para 10 m
        win20 = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        out["SCL"] = src.read(1, window=win20, out_shape=(SIDE, SIDE), resampling=Resampling.nearest)
    return (str(item.datetime.date()), float(item.properties["s2:processing_baseline"]),
            out["B04"], out["B08"], out["SCL"])


def read_with_retry(item, attempts=5):
    import time
    for k in range(attempts):
        try:
            return read_patch(item)
        except Exception as exc:          # falha transitória de rede no COG
            if k == attempts - 1:
                raise
            print(f"nova tentativa {k + 1} em {item.id}: {str(exc)[:60]}", flush=True)
            time.sleep(2 * (k + 1))


with ThreadPoolExecutor(16) as pool:
    rows = sorted(pool.map(read_with_retry, items), key=lambda x: (x[0], x[1]))
np.savez_compressed(
    OUT,
    data=np.array([r[0] for r in rows]),
    baseline=np.array([r[1] for r in rows]),
    B04=np.stack([r[2] for r in rows]),
    B08=np.stack([r[3] for r in rows]),
    SCL=np.stack([r[4] for r in rows]),
)
print("ok", np.stack([r[2] for r in rows]).shape)
