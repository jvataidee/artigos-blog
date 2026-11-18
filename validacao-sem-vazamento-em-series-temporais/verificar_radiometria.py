"""Confere uma cena arquivada por baseline contra XML e COG oficiais, sem alterar dados.

Requer rede. A auditoria é amostral e não valida automaticamente todos os itens.
URLs assinadas são usadas apenas em memória e nunca gravadas no relatório.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import planetary_computer
import rasterio
import requests
from rasterio.warp import transform

HERE = Path(__file__).parent
API = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
LAT, LON = -13.375151012548173, -54.47550071228342


def main():
    raw = pd.read_csv(HERE / "dados" / "pixel_soja_s2.csv", parse_dates=["data"])
    samples = raw.sort_values("data").groupby("baseline").first().reset_index()
    records = []
    for _, row in samples.iterrows():
        date = row["data"]
        query = {"collections": ["sentinel-2-l2a"],
                 "bbox": [LON - 0.00001, LAT - 0.00001, LON + 0.00001, LAT + 0.00001],
                 "datetime": f"{date:%Y-%m-%d}T00:00:00Z/{date:%Y-%m-%d}T23:59:59Z", "limit": 100}
        response = requests.post(API, json=query, timeout=60)
        response.raise_for_status()
        items = response.json()["features"]
        matches = [f for f in items if float(f["properties"]["s2:processing_baseline"]) == row["baseline"]]
        record = {"data": str(date.date()), "baseline": float(row["baseline"])}
        if not matches:
            record["status"] = "baseline arquivada não encontrada no catálogo atual"
            records.append(record)
            print(record, flush=True)
            continue
        item = matches[0]
        assets = item["assets"]
        metadata_key = next((key for key in assets if "product" in key and "metadata" in key), None)
        if metadata_key is None:
            raise ValueError(f"XML do produto ausente: {item['id']}")
        metadata_url = assets[metadata_key]["href"]
        response = requests.get(planetary_computer.sign(metadata_url), timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        offsets = {e.attrib["band_id"]: float(e.text) for e in root.iter()
                   if e.tag.split("}")[-1] == "BOA_ADD_OFFSET"}
        quant = [float(e.text) for e in root.iter()
                 if e.tag.split("}")[-1] in ("BOA_QUANTIFICATION_VALUE", "L2A_BOA_QUANTIFICATION_VALUE")]
        if not quant:
            raise ValueError(f"Quantificação BOA ausente: {item['id']}")
        expected_offset = -1000.0 if row["baseline"] >= 4 else 0.0
        bands = {}
        for name, band_id in [("B04", "3"), ("B08", "7")]:
            with rasterio.open(planetary_computer.sign(assets[name]["href"])) as src:
                xs, ys = transform("EPSG:4326", src.crs, [LON], [LAT])
                r, c = src.index(xs[0], ys[0])
                dn = int(src.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
                bands[name] = {"dn_cog": dn, "dn_arquivado": int(row[name]),
                               "dn_igual": dn == int(row[name]), "escala_cog": src.scales[0],
                               "offset_cog": src.offsets[0], "offset_xml": offsets.get(band_id, 0.0)}
        valid = all(b["dn_igual"] and b["offset_xml"] == expected_offset
                    and b["escala_cog"] == 1 and b["offset_cog"] == 0 for b in bands.values())
        valid = valid and all(q == 10000 for q in quant)
        record.update(item=item["id"], metadados_produto=metadata_url, quantificacao=quant,
                      bandas=bands, status="conferido" if valid else "divergência a investigar")
        records.append(record)
        print({k: v for k, v in record.items() if k not in ("metadados_produto", "bandas")}, flush=True)
    out = HERE / "resultados"
    out.mkdir(exist_ok=True)
    report = {"escopo": "Uma cena por baseline; conferência amostral, sem alteração dos dados.",
              "amostras": records, "todas_as_amostras_conferidas": all(r["status"] == "conferido" for r in records)}
    (out / "radiometria.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(r["status"] == "divergência a investigar" for r in records):
        raise SystemExit("Foram encontradas divergências; consulte resultados/radiometria.json.")


if __name__ == "__main__":
    main()
