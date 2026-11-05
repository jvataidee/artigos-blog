# Validação sem vazamento em séries temporais

Dados e código do artigo [Validação sem vazamento em séries temporais: um pixel de soja do Sentinel-2, walk-forward e teste selado](https://www.joaoataide.com/blog/validacao-sem-vazamento-em-series-temporais/).

Um pixel do Sentinel-2 L2A num talhão de soja no Mato Grosso (-13,3752; -54,4755), de 2019 a 2025, lido do catálogo STAC do Planetary Computer.

## Arquivos

| Arquivo | O que é |
|---|---|
| `dados/pixel_soja_s2.csv` | Uma linha por cena: data, *tile*, *baseline* de processamento, cobertura de nuvem da cena e os valores do pixel em B04, B08 e SCL |
| `dados/ndvi_decendial.csv` | NDVI decendial do pixel (máximo limpo de cada decêndio; vazio quando não há cena limpa) |
| `dados/cubo_soja_s2.npz` | Janela de 32×32 pixels de 10 m centrada no ponto, com B04, B08 e SCL de cada cena (o pixel da série fica em `[16, 16]`) |
| `baixar_pixel.py` | Baixa a série do pixel e grava o CSV |
| `baixar_cubo.py` | Baixa a janela de 32×32 pixels e grava o `.npz` |
| `experimento.py` | Todo o experimento do artigo: NDVI, decêndios, MASE, KFold, *walk-forward*, teste selado e figuras |
| `figura_cubo.py` | Figura do cubo com o pixel da série marcado |
| `analise_extra.py` | Análises de robustez: KFold só com o passado, 10 sementes, bootstrap por blocos, erro por fase da cultura e correlação do talhão |

## Como rodar

```bash
pip install numpy pandas scikit-learn matplotlib rasterio pystac-client planetary-computer

python experimento.py          # usa dados/pixel_soja_s2.csv e grava as figuras em fig/
python figura_cubo.py          # usa dados/cubo_soja_s2.npz e dados/ndvi_decendial.csv
python analise_extra.py        # depende dos dois anteriores estarem nesta pasta
```

Para baixar os dados de novo (cerca de 5 minutos cada):

```bash
python baixar_pixel.py dados/pixel_soja_s2.csv
python baixar_cubo.py dados/cubo_soja_s2.npz
```

O catálogo do Planetary Computer pode mudar com reprocessamentos da ESA, e um *download* novo pode trazer um número diferente de cenas. Os arquivos em `dados/` são os usados no artigo.
