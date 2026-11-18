# Análise de dados do Airbnb em Toronto (setembro de 2026)

Dados e código do [artigo de João Ataíde](https://www.joaoataide.com/blog/analise-de-dados-do-airbnb-toronto-canada/).

Anúncios de Toronto do Inside Airbnb, coletados em 17 e 18/09/2026 (22.050 anúncios, 16.694 usados). O estudo descreve o preço por tipo de quarto e por bairro e ajusta um modelo de preço (log do preço) com validação cruzada por blocos espaciais, teste lacrado em 20% das células de uma grade de 1,5 km e comparação com a divisão aleatória. Compara uma referência (mediana por tipo e hóspedes), regressão linear, LightGBM e CatBoost.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa `listings.csv.gz` do Inside Airbnb para `dados/` (fora do Git) |
| `airbnb.py` | Exploração, modelos, validação por blocos, teste lacrado e divisão aleatória. Grava `resultados/` |

## Como reproduzir

Com Python 3.10 ou superior, na pasta deste artigo:

```bash
python -m pip install -r requirements.txt
python baixar_dados.py
python airbnb.py                 # cerca de 1 minuto
```

## Notas

- **Dados:** o preço é o publicado no dia da coleta, e 24% dos anúncios não têm preço. Os números valem para Toronto em setembro de 2026.
- **Blocos:** o teste usa um único sorteio de 20% das células, então a comparação com a divisão aleatória é indicativa. Repetir o sorteio de blocos seria o próximo passo.
