# Boi gordo e ciclo pecuário

Código do [artigo de João Ataíde](https://www.joaoataide.com/blog/analise-financeira-para-pecuaria-de-corte/).

Analisa a sazonalidade do preço real do boi gordo, o abate de fêmeas por trimestre e o ciclo pecuário, e testa se a participação de fêmeas melhora a previsão do preço a 4 trimestres. Dados abertos: preço do Paraná (IPEADATA), abates por tipo de rebanho (IBGE/SIDRA, tabela 1092), IPCA e dólar (Banco Central). Compara referência sazonal, ARIMA e ETS e o ARIMA com fêmeas, dólar e as duas, em 40 origens móveis e num teste de 4 trimestres.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa as séries das quatro fontes para `dados/` |
| `analise.py` | Análise descritiva: sazonalidade do preço (STL), abate de fêmeas por trimestre e pontos de virada do ciclo. Grava `resultados/` |
| `boi.py` | Monta a série trimestral, calcula a correlação defasada e roda o teste de previsão com o AutoHybTS. Grava `resultados/` |

## Como rodar

```bash
pip install -r requirements.txt
python baixar_dados.py
python boi.py        # teste de previsão (opcional)
python analise.py    # análise descritiva
```

## Limites

- O preço é do Paraná, e não o nacional (o indicador do CEPEA não permite download automático).
- Série curta, com 118 trimestres e poucos ciclos completos.
