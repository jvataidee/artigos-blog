# Predição de evasão de clientes de telecomunicações

Dados e código do [artigo de João Ataíde](https://www.joaoataide.com/blog/predicao-de-evasao-de-clientes-em-empresas-de-telecomunicacoes/).

O estudo prevê a evasão (*churn*) na base Telco Customer Churn, uma amostra pública da IBM com 7.043 clientes. Separa 20% dos clientes como teste lacrado, ajusta todo o pré-processamento dentro de um `Pipeline`, compara regressão logística, Random Forest, gradient boosting, LightGBM e CatBoost com uma referência (validação cruzada de 5 dobras no treino) e escolhe o limiar de decisão nas previsões fora da dobra, só então aplicando-o ao teste. O TabPFN entra se a variável `TABPFN_TOKEN` estiver definida.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `dados/telco_churn.csv` | A base Telco Customer Churn (amostra pública da IBM, fixa) |
| `churn.py` | Protocolo, modelos, validação, limiar, teste lacrado e importância por permutação. Grava `resultados/` |

## Como reproduzir

Com Python 3.10 ou superior, na pasta deste artigo:

```bash
python -m pip install -r requirements.txt
python churn.py                  # cerca de 1 minuto
```

## Notas

- **Base fixa:** a amostra da IBM não tem versão mais nova, então os números valem para ela, e não para uma operadora real.
- **TabPFN:** exige aceitar a licença em https://ux.priorlabs.ai e definir `TABPFN_TOKEN` com a sua chave. Sem a variável, o script o ignora.
