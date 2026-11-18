"""Passo 5: a mesma busca em outro catálogo STAC (Earth Search, da Element 84, na AWS).

Só muda a URL do catálogo e o nome da coleção. Grava resultados/portabilidade.json.
"""

import json
from pathlib import Path

import pystac_client

PASTA = Path(__file__).parent
BBOX = [-35.02, -8.15, -34.86, -7.95]

CATALOGOS = {
    "Planetary Computer": ("https://planetarycomputer.microsoft.com/api/stac/v1", "sentinel-2-l2a"),
    "Earth Search": ("https://earth-search.aws.element84.com/v1", "sentinel-2-c1-l2a"),
}


def main():
    resumo = {}
    for nome, (url, colecao) in CATALOGOS.items():
        itens = pystac_client.Client.open(url).search(
            collections=[colecao],
            bbox=BBOX,
            datetime="2025-01-01/2025-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
        ).item_collection()
        item = itens[0]
        bandas = sorted(item.assets)
        datas = sorted({i.datetime.date().isoformat() for i in itens})
        print(f"{nome}: {len(itens)} itens, {len(datas)} datas, coleção {colecao}")
        print("  bandas do primeiro item:", ", ".join(bandas[:12]), "...")
        resumo[nome] = {"url": url, "colecao": colecao, "itens": len(itens), "datas": datas,
                        "assets_exemplo": bandas}

    a, b = (set(r["datas"]) for r in resumo.values())
    print(f"\ndatas em comum: {len(a & b)} | só no Planetary Computer: {len(a - b)} | só no Earth Search: {len(b - a)}")
    resumo["datas_em_comum"] = len(a & b)
    (PASTA / "resultados" / "portabilidade.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")


if __name__ == "__main__":
    main()
