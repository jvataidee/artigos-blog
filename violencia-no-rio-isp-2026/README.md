# Análise da violência no estado do Rio de Janeiro

Código do [artigo de João Ataíde](https://www.joaoataide.com/blog/analise-da-violencia-no-rio-de-janeiro/).

Ocorrências registradas por município e mês no estado do Rio de Janeiro (ISP-RJ), de janeiro de 2014 a outubro de 2026, em 92 municípios. O estudo descreve a queda da letalidade violenta e dos roubos, a evolução por região e a sazonalidade, e compara seis modelos de previsão de 12 meses (referência sazonal, ETS, ARIMA, ARIMA e Prophet com eventos, e a combinação) em 84 origens mensais, com a pandemia e a intervenção federal de 2018 como eventos.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa `BaseMunicipioMensal.csv` do ISP-RJ para `dados/municipio.csv` |
| `violencia.py` | Séries, sazonalidade e previsão com origens móveis e bootstrap em blocos. Grava `resultados/` |

## Como rodar

```bash
pip install -r requirements.txt
python baixar_dados.py
python violencia.py
```

## Limites

- São ocorrências registradas, e não a violência total, e as contagens não têm população.
- Não há teste lacrado, e as origens vizinhas compartilham dados.
- As datas dos eventos foram escolhidas pelo autor.
