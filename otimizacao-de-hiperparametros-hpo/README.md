# Otimização de hiperparâmetros: busca aleatória, TPE, CMA-ES e algoritmo genético

Código do artigo do blog [joaoataide.com](https://joaoataide.com) que compara quatro métodos de HPO com o mesmo orçamento (60 avaliações) e 10 sementes cada, ajustando 7 hiperparâmetros de um LightGBM na base Telco Customer Churn.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `dados/telco_churn.csv` | A base Telco Customer Churn (amostra pública da IBM, 7.043 clientes). É o mesmo arquivo do projeto `evasao-de-clientes-telecom` |
| `hpo.py` | Os quatro métodos, o objetivo (PR-AUC em validação cruzada de 3 dobras no treino) e o teste lacrado. Grava `resultados/` |
| `resumo.py` | Médias, diferenças pareadas contra a busca aleatória e velocidade de convergência, a partir de `resultados/` |

## Como rodar

```bash
pip install -r requirements.txt
python hpo.py            # cerca de 40 minutos com 10 sementes
python resumo.py
```

Para um teste rápido, `python hpo.py --sementes 1 --avaliacoes 20`.

## Limites

- Uma base, um modelo e um espaço de 7 hiperparâmetros. O resultado não vale para outros problemas.
- A busca aleatória e o algoritmo genético compartilham as 10 primeiras avaliações de cada semente.
- Os hiperparâmetros do algoritmo genético (população 10, torneio 3, mutação 0,2) não foram ajustados.
- O tempo de relógio não foi comparado.
