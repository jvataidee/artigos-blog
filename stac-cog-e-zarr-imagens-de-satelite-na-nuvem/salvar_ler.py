"""Passo 4: guardar o resultado em formatos nativos da nuvem e ler só o pedaço necessário.

- lê do Zarr local (saida/ndvi_recife_2025.zarr) a série de um ponto e conta os chunks tocados;
- grava o NDVI de um mês como COG (saida/ndvi_recife_2025_10.tif) e confere blocos e overviews.

Grava resultados/zarr_cog.json.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registra o acessor .rio)
import xarray as xr
import zarr
from pyproj import Transformer

PASTA = Path(__file__).parent
ZARR = PASTA / "saida" / "ndvi_recife_2025.zarr"
PONTO = (-34.936, -8.024)  # mata densa na região de Dois Irmãos (lon, lat)


def tamanho_pasta(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main():
    resumo = {}

    # 1. estrutura do Zarr: metadados + um arquivo por chunk
    grupo = zarr.open_group(ZARR, mode="r")
    arr = grupo["ndvi"]
    arquivos_chunk = [f for f in (ZARR / "ndvi").rglob("*") if f.is_file() and f.name != "zarr.json"]
    print("ndvi:", arr.shape, arr.dtype, "| chunks de", arr.chunks)
    print(f"{len(arquivos_chunk)} arquivos de chunk, {tamanho_pasta(ZARR) / 1024**2:.1f} MB no disco "
          f"(sem compressão seriam {arr.nbytes / 1024**2:.1f} MB)")
    resumo["zarr"] = {"shape": list(arr.shape), "chunks": list(arr.chunks), "arquivos_chunk": len(arquivos_chunk),
                      "mb_disco": round(tamanho_pasta(ZARR) / 1024**2, 1), "mb_bruto": round(arr.nbytes / 1024**2, 1),
                      "compressores": [str(c) for c in arr.compressors]}

    # 2. série temporal de um ponto: quantos chunks precisamos abrir?
    ds = xr.open_zarr(ZARR)
    crs = ds["ndvi"].attrs.get("crs") or "EPSG:32725"
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*PONTO)
    serie = ds["ndvi"].sel(x=x, y=y, method="nearest")
    ix = int(np.abs(ds.x.values - x).argmin())
    iy = int(np.abs(ds.y.values - y).argmin())
    tocados = {(t, iy // arr.chunks[1], ix // arr.chunks[2]) for t in range(arr.shape[0])}
    total = int(np.prod([int(np.ceil(s / c)) for s, c in zip(arr.shape, arr.chunks)]))
    valores = serie.values
    print(f"\nponto {PONTO}: chunks lidos {len(tocados)} de {total}")
    for t, v in zip(ds.time.values, valores):
        print(str(t)[:7], "nan" if np.isnan(v) else f"{v:.3f}")
    resumo["serie_ponto"] = {"lonlat": PONTO, "chunks_lidos": len(tocados), "chunks_total": total,
                             "ndvi": {str(t)[:7]: (None if np.isnan(v) else round(float(v), 3))
                                      for t, v in zip(ds.time.values, valores)}}

    # 3. um mês como COG: blocos internos + overviews
    outubro = ds["ndvi"].sel(time="2025-10").squeeze("time", drop=True)
    outubro = outubro.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs(crs)
    cog = PASTA / "saida" / "ndvi_recife_2025_10.tif"
    outubro.rio.to_raster(cog, driver="COG", compress="DEFLATE", predictor=3, blocksize=512,
                         overview_resampling="average")
    with rasterio.open(cog) as src:
        info = {"px": [src.width, src.height], "bloco": list(src.block_shapes[0]), "overviews": src.overviews(1),
                "compressao": src.compression.name, "mb": round(cog.stat().st_size / 1024**2, 2),
                "layout": src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")}
    print("\nCOG de outubro:", info)
    resumo["cog_outubro"] = info

    (PASTA / "resultados" / "zarr_cog.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
