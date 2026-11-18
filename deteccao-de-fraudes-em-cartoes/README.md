# Detecção de fraudes em cartões de crédito

Código do [artigo de João Ataíde](https://www.joaoataide.com/blog/deteccao-de-fraudes-em-cartoes-de-credito/).

Transações de cartões europeus em dois dias de 2013 (284.807 transações, 492 fraudes). O treino é sempre o passado, e a avaliação usa quatro janelas de 6 horas com orçamento de 0,2% de alertas. Compara regressão logística, Random Forest, LightGBM e CatBoost, com e sem a coluna `Time`, e dois rankings (probabilidade, e probabilidade vezes o valor da transação).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa `creditcard.csv` para `dados/` |
| `fraude.py` | Janelas de tempo, modelos, recall no orçamento e valor recuperado. Grava `resultados/` |

## Como rodar

```bash
pip install -r requirements.txt
python baixar_dados.py
python fraude.py
```

## Limites

- Só dois dias de dados, com variáveis anonimizadas (componentes principais).
- O valor da transação não é o prejuízo real.
- Não há teste lacrado, e o orçamento fixo limita o recall quando a taxa de fraude da janela é alta.
