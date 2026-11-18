# Risco de crédito: inadimplência

Código do [artigo de João Ataíde](https://www.joaoataide.com/blog/avaliacao-de-risco-de-credito/).

41.741 pedidos de crédito com desfecho conhecido, de um conjunto anonimizado. Compara regressão logística, LightGBM e CatBoost, com e sem variáveis derivadas, em validação cruzada estratificada de 5 dobras repetida 3 vezes, com calibração isotônica, e avalia a decisão de crédito por custo (aprovar todos, recusar 20%, corte ótimo). Inclui o experimento de balancear a base antes de dividir.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa `acquisition_train.csv` para `dados/` |
| `credito.py` | Validação repetida, derivadas, calibração, decisão por custo e o experimento do balanceamento. Grava `resultados/` |

## Como rodar

```bash
pip install -r requirements.txt
python baixar_dados.py
python credito.py                      # cerca de 15 minutos
python credito.py --so-balanceamento   # só o experimento do balanceamento
```

## Limites

- A base só tem pedidos com desfecho conhecido, então a política é avaliada numa população já filtrada.
- A razão de custo é uma hipótese, e não um dado. Não há coluna de data, e a divisão é aleatória.
