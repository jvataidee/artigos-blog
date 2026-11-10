"""Passo 2: abrir um COG direto na nuvem e medir quantos bytes cada leitura transfere.

Usa a banda B04 do primeiro item da busca (dados/itens_recife_2025.json). O arquivo é
aberto por um leitor HTTP próprio, que faz requisições com cabeçalho Range e conta os
bytes recebidos. Grava o resumo em resultados/cog.json.
"""

import io
import json
import time
from pathlib import Path

import numpy as np
import planetary_computer
import pystac
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

PASTA = Path(__file__).parent
BBOX = [-35.02, -8.15, -34.86, -7.95]


class ArquivoHTTP(io.RawIOBase):
    """Arquivo remoto lido por requisições HTTP Range, com contador de bytes."""

    def __init__(self, url, sessao):
        self.url, self.sessao, self.pos = url, sessao, 0
        resp = sessao.head(url, allow_redirects=True, timeout=60)
        resp.raise_for_status()
        self.tamanho = int(resp.headers["Content-Length"])
        self.bytes_lidos = 0
        self.requisicoes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.tamanho}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.tamanho - self.pos
        n = min(n, self.tamanho - self.pos)
        if n <= 0:
            return b""
        cab = {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        resp = self.sessao.get(self.url, headers=cab, timeout=120)
        resp.raise_for_status()
        dados = resp.content
        self.pos += len(dados)
        self.bytes_lidos += len(dados)
        self.requisicoes += 1
        return dados

    def readinto(self, b):
        dados = self.read(len(b))
        b[: len(dados)] = dados
        return len(dados)


def mb(n):
    return n / 1024**2


def medir(url, sessao, leitura):
    """Abre o COG pelo `opener` do rasterio: o GDAL pede trechos e o ArquivoHTTP conta os bytes."""
    abertos = []

    def opener(caminho, mode="rb", **kwds):
        if not str(caminho).endswith("b04.tif"):
            raise FileNotFoundError(caminho)  # arquivos auxiliares (.aux.xml, .ovr) não existem
        arquivo = ArquivoHTTP(url, sessao)
        abertos.append(arquivo)
        return arquivo

    t0 = time.perf_counter()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open("b04.tif", opener=opener) as src:
        resultado = leitura(src)
    seg = time.perf_counter() - t0
    total = Contador(sum(a.bytes_lidos for a in abertos), sum(a.requisicoes for a in abertos), abertos[0].tamanho)
    return resultado, total, seg


class Contador:
    def __init__(self, bytes_lidos, requisicoes, tamanho):
        self.bytes_lidos, self.requisicoes, self.tamanho = bytes_lidos, requisicoes, tamanho


def main():
    itens = pystac.ItemCollection.from_file(str(PASTA / "dados" / "itens_recife_2025.json"))
    # a quadrícula 25MBM cobre a maior parte de Recife; pegamos a cena dela com menos nuvens
    mbm = [i for i in itens if i.properties["s2:mgrs_tile"] == "25MBM"]
    item = planetary_computer.sign(min(mbm, key=lambda i: i.properties["eo:cloud_cover"]))
    url = item.assets["B04"].href
    sessao = requests.Session()

    # 1. só o cabeçalho: metadados, grade, blocos e overviews
    def cabecalho(src):
        return {
            "largura": src.width,
            "altura": src.height,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "bloco": list(src.block_shapes[0]),
            "overviews": src.overviews(1),
            "compressao": src.compression.name if src.compression else None,
            "blocos_total": len(list(src.block_windows(1))),
        }

    info, arq, seg = medir(url, sessao, cabecalho)
    print("item:", item.id)
    print(f"arquivo B04: {mb(arq.tamanho):.1f} MB, {info['largura']} x {info['altura']} px, {info['dtype']}")
    print("blocos:", info["bloco"], "| overviews:", info["overviews"], "| compressão:", info["compressao"])
    print(f"só o cabeçalho: {arq.bytes_lidos / 1024:.0f} KB em {arq.requisicoes} requisições")
    tamanho = arq.tamanho
    resumo = {"item": item.id, "arquivo_mb": round(mb(tamanho), 1), **info,
              "cabecalho": {"kb": round(arq.bytes_lidos / 1024, 1), "requisicoes": arq.requisicoes}}

    # 2. a janela de Recife na resolução cheia
    def janela(src):
        limites = transform_bounds("EPSG:4326", src.crs, *BBOX)
        jan = from_bounds(*limites, transform=src.transform).round_offsets().round_lengths()
        jan = jan.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        arr = src.read(1, window=jan)
        return arr, jan

    (arr, jan), arq, seg = medir(url, sessao, janela)
    frac_area = (jan.width * jan.height) / (info["largura"] * info["altura"])
    print(f"\njanela de Recife: {int(jan.width)} x {int(jan.height)} px ({100 * frac_area:.1f}% da área da cena)")
    print(f"transferido: {mb(arq.bytes_lidos):.1f} MB ({100 * arq.bytes_lidos / tamanho:.1f}% do arquivo) "
          f"em {arq.requisicoes} requisições, {seg:.1f} s")
    resumo["janela_recife"] = {
        "px": [int(jan.width), int(jan.height)],
        "pct_area": round(100 * frac_area, 2),
        "mb": round(mb(arq.bytes_lidos), 2),
        "pct_arquivo": round(100 * arq.bytes_lidos / tamanho, 2),
        "requisicoes": arq.requisicoes,
        "segundos": round(seg, 1),
        "media_nd": float(np.mean(arr)),
    }

    # 3. a cena inteira, mas numa versão reduzida (overview 1/16)
    def reduzida(src):
        return src.read(1, out_shape=(src.height // 16, src.width // 16), resampling=Resampling.nearest)

    arr, arq, seg = medir(url, sessao, reduzida)
    print(f"\ncena inteira em 1/16: {arr.shape[1]} x {arr.shape[0]} px")
    print(f"transferido: {mb(arq.bytes_lidos):.1f} MB ({100 * arq.bytes_lidos / tamanho:.1f}% do arquivo) "
          f"em {arq.requisicoes} requisições, {seg:.1f} s")
    resumo["overview_16"] = {
        "px": [arr.shape[1], arr.shape[0]],
        "mb": round(mb(arq.bytes_lidos), 2),
        "pct_arquivo": round(100 * arq.bytes_lidos / tamanho, 2),
        "requisicoes": arq.requisicoes,
        "segundos": round(seg, 1),
    }

    # 4. as mesmas leituras pelo leitor HTTP nativo do GDAL (/vsicurl/), só para medir o tempo
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        t0 = time.perf_counter()
        with rasterio.open(url) as src:
            janela(src)
        t_janela = time.perf_counter() - t0
        t0 = time.perf_counter()
        with rasterio.open(url) as src:
            reduzida(src)
        t_reduzida = time.perf_counter() - t0
    print(f"\nGDAL nativo: janela de Recife em {t_janela:.1f} s, cena em 1/16 em {t_reduzida:.1f} s")
    resumo["gdal_nativo_segundos"] = {"janela_recife": round(t_janela, 1), "overview_16": round(t_reduzida, 1)}

    (PASTA / "resultados" / "cog.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
