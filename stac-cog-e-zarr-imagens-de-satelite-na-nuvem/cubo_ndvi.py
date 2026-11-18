"""Passo 3: montar o cubo B04, B08 e SCL de Recife em 2025 e calcular o NDVI mensal.

Lê só a janela de Recife de cada COG, com odc-stac e dask. Grava o NDVI mensal em
Zarr (saida/ndvi_recife_2025.zarr) e o resumo em resultados/cubo.json. A figura sai do figuras.py.
"""

import json
import time
from pathlib import Path

import odc.stac
import planetary_computer
import pystac
import requests

PASTA = Path(__file__).parent
BBOX = [-35.02, -8.15, -34.86, -7.95]
BANDAS = ["B04", "B08", "SCL"]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def tamanho_total(itens):
    """Soma o tamanho dos arquivos completos das bandas usadas (HEAD em cada asset)."""
    sessao = requests.Session()
    total = 0
    for item in itens:
        for banda in BANDAS:
            resp = sessao.head(item.assets[banda].href, allow_redirects=True, timeout=60)
            resp.raise_for_status()
            total += int(resp.headers["Content-Length"])
    return total


def main():
    itens = pystac.ItemCollection.from_file(str(PASTA / "dados" / "itens_recife_2025.json"))
    # só as datas com pelo menos 50% de Recife limpa (triagem_nuvens.py)
    datas = set(json.loads((PASTA / "resultados" / "triagem.json").read_text(encoding="utf-8"))["selecionadas"])
    itens = [planetary_computer.sign(i) for i in itens if i.datetime.date().isoformat() in datas]
    print(len(itens), "itens em", len(datas), "datas")

    # a partir da baseline 04.00 (jan/2022) o L2A soma 1000 aos números digitais
    baselines = sorted({i.properties["s2:processing_baseline"] for i in itens})
    assert all(float(b) >= 4.0 for b in baselines), baselines

    cubo = odc.stac.load(
        itens,
        bands=BANDAS,
        bbox=BBOX,
        resolution=10,
        groupby="solar_day",
        chunks={"x": 1024, "y": 1024},  # leitura preguiçosa com dask
    )
    print(cubo)
    print(f"\ncubo em memória (se carregado inteiro): {cubo.nbytes / 1024**2:.0f} MB")

    arquivos = tamanho_total(itens)
    print(f"arquivos completos de B04, B08 e SCL das {len(itens)} cenas: {arquivos / 1024**3:.2f} GB")

    # SCL: 4 vegetação, 5 solo exposto, 6 água (o resto é nuvem, sombra, sem dado etc.)
    validos = cubo.SCL.isin([4, 5, 6]) & (cubo.B04 > 0)
    red = (cubo.B04.where(validos) - 1000) / 10000
    nir = (cubo.B08.where(validos) - 1000) / 10000
    soma = nir + red
    ndvi = ((nir - red) / soma).where(soma > 0)  # água escura pode zerar o denominador

    t0 = time.perf_counter()
    ndvi_mensal = ndvi.resample(time="1MS").median().compute()
    frac_valida = validos.resample(time="1MS").max().mean(dim=["x", "y"]).compute()
    seg = time.perf_counter() - t0
    print(f"\nNDVI mensal calculado em {seg:.0f} s")

    por_mes = []
    for t, fv in zip(frac_valida.time.values, frac_valida.values):
        mes = int(str(t)[5:7])
        n = int((ndvi.time.dt.month == mes).sum())
        media = float(ndvi_mensal.sel(time=t).mean())
        por_mes.append({"mes": MESES[mes - 1], "datas": n, "pct_area_com_dado": round(100 * float(fv), 1),
                        "ndvi_medio": round(media, 3)})
        print(f"{MESES[mes - 1]}: {n} data(s), {100 * float(fv):5.1f}% da área com ao menos um pixel limpo, "
              f"NDVI médio {media:.3f}")

    # Zarr: um chunk por mês, blocos de 512 x 512
    saida = PASTA / "saida"
    saida.mkdir(exist_ok=True)
    ds = ndvi_mensal.astype("float32").to_dataset(name="ndvi")
    ds = ds.drop_vars([v for v in ds.coords if v == "spatial_ref"], errors="ignore")
    ds["ndvi"].encoding = {}
    ds["ndvi"].attrs = {"crs": str(cubo.odc.crs), "descricao": "NDVI, mediana mensal, SCL 4/5/6"}
    ds.chunk({"time": 1, "y": 512, "x": 512}).to_zarr(saida / "ndvi_recife_2025.zarr", mode="w")

    resumo = {
        "cenas": len(itens),
        "datas_solares": int(cubo.sizes["time"]),
        "grade_px": [int(cubo.sizes["x"]), int(cubo.sizes["y"])],
        "crs": str(cubo.odc.crs),
        "baselines": baselines,
        "cubo_mb": round(cubo.nbytes / 1024**2, 1),
        "arquivos_completos_gb": round(arquivos / 1024**3, 2),
        "segundos_ndvi": round(seg, 1),
        "por_mes": por_mes,
    }
    (PASTA / "resultados" / "cubo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
