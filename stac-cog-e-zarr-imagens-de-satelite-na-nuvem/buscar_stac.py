"""Passo 1: buscar cenas Sentinel-2 L2A sobre Recife em 2025 no STAC do Planetary Computer.

Grava o resultado da busca (só metadados, nenhum pixel) em dados/itens_recife_2025.json
e um resumo em resultados/busca.json.
"""

import json
import time
from pathlib import Path

import pystac_client

PASTA = Path(__file__).parent
CATALOGO = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [-35.02, -8.15, -34.86, -7.95]  # Recife (lon/lat)
PERIODO = "2025-01-01/2025-12-31"


def buscar(nuvens_max=None):
    catalogo = pystac_client.Client.open(CATALOGO)
    filtro = {"eo:cloud_cover": {"lt": nuvens_max}} if nuvens_max is not None else None
    busca = catalogo.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=PERIODO,
        query=filtro,
        sortby="datetime",
    )
    return busca.item_collection()


def main():
    (PASTA / "dados").mkdir(exist_ok=True)
    (PASTA / "resultados").mkdir(exist_ok=True)

    t0 = time.perf_counter()
    todas = buscar()
    itens = buscar(nuvens_max=20)
    segundos = time.perf_counter() - t0

    print(len(todas), "cenas em 2025 sobre Recife")
    print(len(itens), "cenas com menos de 20% de nuvens")

    item = itens[0]
    print("\nprimeiro item:", item.id)
    print("data:", item.datetime.isoformat())
    print("nuvens (%):", item.properties["eo:cloud_cover"])
    print("tile MGRS:", item.properties["s2:mgrs_tile"])
    print("EPSG:", item.properties["proj:code"] if "proj:code" in item.properties else item.properties.get("proj:epsg"))
    print("assets:", ", ".join(item.assets))

    b04 = item.assets["B04"]
    print("\nB04 ->", b04.media_type)
    print(b04.href)

    # guardamos todas as cenas; a triagem de nuvens é feita na área de Recife (triagem_nuvens.py)
    todas.save_object(str(PASTA / "dados" / "itens_recife_2025.json"))

    nuvens = [i.properties["eo:cloud_cover"] for i in todas]
    por_limite = {f"lt{t}": sum(c < t for c in nuvens) for t in (20, 40, 60, 80)}
    print("\ncenas por limite de eo:cloud_cover:", por_limite)

    resumo = {
        "catalogo": CATALOGO,
        "colecao": "sentinel-2-l2a",
        "bbox": BBOX,
        "periodo": PERIODO,
        "cenas_total": len(todas),
        "cenas_nuvens_lt20": len(itens),
        "datas_total": len({i.datetime.date() for i in todas}),
        "cenas_por_limite_nuvens": por_limite,
        "tiles": sorted({i.properties["s2:mgrs_tile"] for i in todas}),
        "datas_lt20": [i.datetime.date().isoformat() for i in itens],
        "segundos_duas_buscas": round(segundos, 1),
    }
    (PASTA / "resultados" / "busca.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
