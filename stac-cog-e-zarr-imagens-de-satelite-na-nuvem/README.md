# STAC, COG e Zarr: imagens de satélite na nuvem sem baixar terabytes

Dados e código do [artigo de João Ataíde](https://www.joaoataide.com/blog/stac-cog-e-zarr-imagens-de-satelite-na-nuvem/).

Sentinel-2 L2A sobre Recife (bbox `-35.02, -8.15, -34.86, -7.95`) em 2025, buscado no catálogo STAC do Microsoft Planetary Computer. O exemplo mostra a busca no catálogo, a triagem de nuvens na área de interesse, a leitura parcial de um COG com contagem de bytes, um cubo de NDVI mensal com `odc-stac` e o resultado gravado em Zarr e COG.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `buscar_stac.py` | Busca STAC (todas as cenas e o filtro `eo:cloud_cover < 20`), grava os itens em `dados/` |
| `triagem_nuvens.py` | Lê a SCL a 200 m (overviews) e mede a fração de Recife limpa em cada data |
| `ler_cog.py` | Abre a B04 de uma cena por um leitor HTTP próprio e conta os bytes de cada leitura |
| `cubo_ndvi.py` | Cubo B04/B08/SCL das datas selecionadas, NDVI mensal e gravação em Zarr |
| `salvar_ler.py` | Série de um ponto lida do Zarr (chunks tocados) e um mês gravado como COG |
| `portabilidade.py` | A mesma busca no Earth Search (Element 84, AWS) |
| `figuras.py` | Painel do NDVI mensal (figura 3 do artigo), a partir dos arquivos locais |
| `dados/itens_recife_2025.json` | Os 202 itens STAC da busca (só metadados, links sem token) |
| `resultados/` | Resumos em JSON e CSV de cada passo, com os números citados no artigo |

## Como reproduzir

Com Python 3.12, na pasta deste artigo:

```bash
python -m pip install -r requirements.txt
python buscar_stac.py
python triagem_nuvens.py
python ler_cog.py
python cubo_ndvi.py
python salvar_ler.py
python portabilidade.py
python figuras.py
```

Todos os passos, menos `salvar_ler.py` e `figuras.py`, acessam a rede. O `cubo_ndvi.py` levou cerca de quatro minutos numa conexão doméstica. Rasters e Zarr vão para `saida/` e as figuras para `fig/`, as duas pastas fora do Git.

O `planetary_computer.sign` acrescenta aos links um token SAS temporário. Os itens em `dados/` foram gravados antes da assinatura e não contêm tokens.

## Decisões e limites

- **Triagem de nuvens.** Uma data entra no cubo se pelo menos 50% dos pixels com dado na área de Recife forem das classes SCL 4, 5 ou 6 (vegetação, solo exposto, água). Para comparar com o filtro clássico, a data passa no `eo:cloud_cover < 20` se ao menos um dos seus itens passar (Recife fica na divisa das quadrículas 25LBL e 25MBM).
- **Radiometria.** Todos os 202 itens têm processing baseline 05.11. No Planetary Computer, os números digitais do L2A vêm com deslocamento de +1000, removido antes de dividir por 10.000. No Earth Search (`sentinel-2-c1-l2a`), a escala e o deslocamento estão declarados em `raster:bands`.
- **Contagem de bytes.** O `ler_cog.py` conta os bytes entregues ao GDAL pelo leitor em Python, que faz uma requisição por trecho. Os tempos desse leitor não representam o GDAL nativo, medido à parte no mesmo script.
- **NDVI.** Calculado só onde vermelho + infravermelho > 0. A mediana mensal ignora pixels mascarados. Fevereiro e agosto não têm data selecionada.
- **Reprodutibilidade.** O catálogo muda com reprocessamentos. Uma busca nova pode devolver itens diferentes dos arquivados em `dados/`.
